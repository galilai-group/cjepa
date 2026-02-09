from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torchvision
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from loguru import logger as logging
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel
import wandb
from model.dinowm_causal_AP_node import CausalWM_AP
from model.cjepa_predictor import MaskedSlotPredictor
from model.custom_codes.custom_dataset import ClevrerVideoDataset

import os
import gdown

# import sys, importlib; sys.modules["videosaur"] = importlib.import_module("videosaur.videosaur")
from third_party.videosaur.videosaur import  models

DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches


# ============================================================================
# Data Setup
# ============================================================================
def get_data(cfg):
    """Setup dataset with image transforms and normalization."""

    def get_img_pipeline(key, target, img_size=224):
        return spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                **spt.data.dataset_stats.ImageNet,
                source=key,
                target=target,
            ),
            spt.data.transforms.Resize(img_size, source=key, target=target),
            spt.data.transforms.CenterCrop(img_size, source=key, target=target),
        )
    def get_img_pipeline_minimal(key, target, img_size=224):
        return spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                **spt.data.dataset_stats.ImageNet,
                source=key,
                target=target,
            ),
            spt.data.transforms.Resize((img_size, img_size), source=key, target=target),
        )    

    def norm_col_transform(dataset, col="pixels"):
        """Normalize column to zero mean, unit variance."""
        data = dataset[col][:]
        mean = data.mean(0).unsqueeze(0)
        std = data.std(0).unsqueeze(0)
        return lambda x: (x - mean) / std

    if "clevrer" in cfg.dataset_name:
        train_set = ClevrerVideoDataset(
            cfg.dataset_name + "_train",
            num_steps=cfg.n_steps,
            frameskip=cfg.frameskip,
            transform=None,
            cache_dir=cfg.get("cache_dir", None),

        )    
        val_set = ClevrerVideoDataset(
            cfg.dataset_name + "_val",
            num_steps=cfg.n_steps,
            frameskip=cfg.frameskip,
            transform=None,
            cache_dir=cfg.get("cache_dir", None),
            idx_offset=10000,  # to avoid episode index conflict with train set
        )    
    else :
        train_set = swm.data.VideoDataset(
            cfg.dataset_name + "_train",
            num_steps=cfg.n_steps,
            frameskip=cfg.frameskip,
            transform=None,
            cache_dir=cfg.get("cache_dir", None),
        )    
        val_set = swm.data.VideoDataset(
            cfg.dataset_name + "_val",
            num_steps=cfg.n_steps,
            frameskip=cfg.frameskip,
            transform=None,
            cache_dir=cfg.get("cache_dir", None),
        )

    # Image size must be multiple of DINO patch size (14)
    img_size = (cfg.image_size // cfg.patch_size) * DINO_PATCH_SIZE



    # Apply transforms to all steps
    if "clevrer" in cfg.dataset_name:
        transform = spt.data.transforms.Compose(
            *[get_img_pipeline_minimal(f"{col}.{i}", f"{col}.{i}", img_size) for col in ["pixels"] for i in range(cfg.n_steps)],
        )
        train_set.transform = transform
        val_set.transform = transform
    else :
        train_transform = spt.data.transforms.Compose(
            *[get_img_pipeline(f"{col}.{i}", f"{col}.{i}", img_size) for col in ["pixels"] for i in range(cfg.n_steps)],
            spt.data.transforms.WrapTorchTransform(
                norm_col_transform(train_set.dataset, "action"),
                source="action", 
                target="action",
            ),
            spt.data.transforms.WrapTorchTransform(
                norm_col_transform(train_set.dataset, "proprio"),
                source="proprio",
                target="proprio",
            ),
        )
        val_transform = spt.data.transforms.Compose(
            *[get_img_pipeline(f"{col}.{i}", f"{col}.{i}", img_size) for col in ["pixels"] for i in range(cfg.n_steps)],
            spt.data.transforms.WrapTorchTransform(
                norm_col_transform(val_set.dataset, "action"),
                source="action",
                target="action",
            ),
            spt.data.transforms.WrapTorchTransform(
                norm_col_transform(val_set.dataset, "proprio"),
                source="proprio",
                target="proprio",
            ),
        )
        train_set.transform = train_transform
        val_set.transform = val_transform


    rnd_gen = torch.Generator().manual_seed(cfg.seed)

    logging.info(f"Train: {len(train_set)}, Val: {len(val_set)}")

    train = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    # in video, sample keys : 'pixels' (bs,T,C,H,W), 'goal' (=pixels), 'action' 0*(bs, t, 1), 'episode_idx' (bs), 'step_idx'(bs, t), 'episode_len' 128*(bs)
    val = DataLoader(val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)

    return spt.data.DataModule(train=train, val=val)


# ============================================================================
# Model Architecture
# ============================================================================
def get_world_model(cfg):
    """Build world model: frozen videosaur encoder + masked slot predictor."""

    def forward(self, batch, stage):
        """Forward: encode observations, predict next states, compute losses."""

        proprio_key = "proprio" if "proprio" in batch else None

        # Replace NaN values with 0 (occurs at sequence boundaries)
        if proprio_key is not None:
            batch[proprio_key] = torch.nan_to_num(batch[proprio_key], 0.0)
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        # Encode all timesteps into latent embeddings
        if "clevrer" in cfg.dataset_name:
            batch = self.model.encode(
                batch,
                target="embed",
                pixels_key="pixels"
            )
        elif "pusht" in cfg.dataset_name:
            batch = self.model.encode(
                batch,
                target="embed",
                pixels_key="pixels",
                proprio_key=proprio_key,
                action_key="action",
            )
        # Use history to predict next states
        embedding = batch["embed"][:, : cfg.dinowm.history_size, :, :]  # (B, history_size, S+2, 64)
        

        # Request mask information for selective loss
        pred_output = self.model.predict(embedding)
        pixels_dim = batch["pixels_embed"].shape[-1]
        
        if len(pred_output[1]) > 0:  # mask_indices available
            pred_embedding, mask_indices = pred_output
            target_embedding = batch["embed"][:, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]  # (B, num_pred, S, 64)
            
            pred_history = pred_embedding[:, :cfg.dinowm.history_size, :, :]      # (B, T, S, 64)
            pred_future = pred_embedding[:, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]       # (B, num_pred, S, 64)
            
            # Loss 1: Masked slots in history (what was masked should be recovered)
            # Only compute loss on masked slots
            gt_history = embedding[:, :, :, :]  # Ground truth history (unmasked)
            loss_masked_history = F.mse_loss(
                pred_history[:, :, mask_indices, :pixels_dim],
                gt_history[:, :, mask_indices, :pixels_dim].detach()
            )
            loss_future = F.mse_loss(pred_future[..., :pixels_dim], target_embedding[..., :pixels_dim].detach())

 
            batch["loss"] = loss_masked_history + loss_future
            batch["loss_masked_history"] = loss_masked_history
            batch["loss_future"] = loss_future
        else :
            pred_embedding = pred_output[0]
            pred_future = pred_embedding[:, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]       # (B, num_pred, S, 64)
            target_embedding = batch["embed"][:, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]  # (B, num_pred, S, 64)
            loss_future = F.mse_loss(pred_future[..., :pixels_dim], target_embedding[..., :pixels_dim].detach())
            batch["loss"] = loss_future 
            


        
        # Flatten predictions for RankMe: (B, T, S, D) or (B, num_pred, S, D) -> (B*T, S*D) or (B*num_pred, S*D)
        if isinstance(pred_output, tuple) and len(pred_output) > 0:
            # (B, T, S, D) -> (B*T, S*D)
            B, T, S, D = pred_output[0].shape
            pred_flat = pred_output[0].reshape(B*T, S*D)
        else:
            # (B, num_pred, S, D) -> (B*num_pred, S*D)
            B, num_pred, S, D = pred_embedding.shape
            pred_flat = pred_embedding.reshape(B*num_pred, S*D)
        batch["predictor_embed"] = pred_flat

        # Log all losses
        prefix = "train/" if self.training else "val/"
        losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "loss" in k}
        self.log_dict(losses_dict, on_step=True, sync_dist=True)

        return batch

    model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = model.encoder 
    slot_attention = model.processor 
    initializer = model.initializer
    embedding_dim = cfg.videosaur.SLOT_DIM 

    # num_patches = (cfg.image_size // cfg.patch_size) ** 2
    num_patches = cfg.videosaur.NUM_SLOTS
    logging.info(f"Patches: {num_patches}, Embedding dim: {embedding_dim}")

    # Build masked slot predictor (V-JEPA style)
    predictor = MaskedSlotPredictor(
        num_slots=num_patches + 2,  # number of slots + action_node + proprio_node
        slot_dim=embedding_dim,  # 64 or higher if action/proprio included
        history_frames=cfg.dinowm.history_size,  # T: history length
        pred_frames=cfg.dinowm.num_preds,  # number of future frames to predict
        num_masked_slots=cfg.get("num_masked_slots", 2),  # M: number of slots to mask
        seed=cfg.seed,  # for reproducible masking
        depth=cfg.predictor.get("depth", 6),
        heads=cfg.predictor.get("heads", 16),
        dim_head=cfg.predictor.get("dim_head", 64),
        mlp_dim=cfg.predictor.get("mlp_dim", 2048),
        dropout=cfg.predictor.get("dropout", 0.1),
    )
    # Build action and proprioception encoders
    if "clevrer" in cfg.dataset_name:    
        action_encoder = None
        proprio_encoder = None

        logging.info(f"[Video Only] Action encoder: None, Proprio encoder: None")
    elif "pusht" in cfg.dataset_name:
        effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
        action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=cfg.videosaur.SLOT_DIM)
        proprio_encoder = swm.wm.dinowm.Embedder(in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.videosaur.SLOT_DIM)

        logging.info(f"Action dim: {effective_act_dim}, Proprio dim: {cfg.dinowm.proprio_dim}")

    # Assemble world model
    world_model = CausalWM_AP(
        encoder=spt.backbone.EvalOnly(encoder),
        slot_attention=spt.backbone.EvalOnly(slot_attention),
        initializer = spt.backbone.EvalOnly(initializer),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
    )

    # Wrap in stable_spt Module with separate optimizers for each component
    def add_opt(module_name, lr):
        return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}

    world_model = spt.Module(
        model=world_model,
        forward=forward,
        optim={
            "predictor_opt": add_opt("model.predictor", cfg.predictor_lr),
            "proprio_opt": add_opt("model.proprio_encoder", cfg.proprio_encoder_lr),
            "action_opt": add_opt("model.action_encoder", cfg.action_encoder_lr),
        },
    )
    return world_model


# ============================================================================
# Training Setup
# ============================================================================
def setup_pl_logger(cfg):
    if not cfg.wandb.enable:
        return None
    # try:
    wandb_run_id = cfg.wandb.get("run_id", None)
    wandb_logger = WandbLogger(
        name="dino_wm",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        resume="allow" if wandb_run_id else None,
        id=wandb_run_id,
        log_model=False,
    )

    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return wandb_logger


class ModelObjectCallBack(Callback):
    """Callback to pickle model after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                output_path = Path(
                    self.dirpath,
                    f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt",
                )
                torch.save(pl_module, output_path)
                logging.info(f"Saved world model object to {output_path}")
            # Additionally, save at final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                final_path = self.dirpath / f"{self.filename}_object.ckpt"
                torch.save(pl_module, final_path)
                logging.info(f"Saved final world model object to {final_path}")


# ============================================================================
# Main Entry Point
# ============================================================================
@hydra.main(version_base=None, config_path="../configs", config_name="config_train_causal")
def run(cfg):
    """Run training of predictor"""

    wandb_logger = setup_pl_logger(cfg)
    data = get_data(cfg)
    world_model = get_world_model(cfg)

    cache_dir = swm.data.utils.get_cache_dir() if cfg.cache_dir is None else cfg.cache_dir
    dump_object_callback = ModelObjectCallBack(
        dirpath=cache_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )
    
    # Setup RankMe callback for monitoring predictor embedding quality
    callbacks = [dump_object_callback]
    if cfg.get("monitor_rankme", False):
        target_shape = (cfg.videosaur.NUM_SLOTS+2) * cfg.videosaur.SLOT_DIM
        # RankMe uses a queue to track embeddings and compute effective rank
        rankme_callback = spt.callbacks.RankMe(
            name="rankme/predictor",
            target="predictor_embed",
            queue_length=cfg.get("rankme_queue_length", 2048),
            target_shape=target_shape,  # S * D flattened
        )
        callbacks.append(rankme_callback)
        logging.info(
            f"RankMe monitoring enabled (queue_length={cfg.get('rankme_queue_length', 2048)}, "
            f"target_shape={target_shape})"
        )


    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=wandb_logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data,
        ckpt_path=f"{cache_dir}/{cfg.output_model_name}_weights.ckpt",
        seed=cfg.seed
    )
    manager()


if __name__ == "__main__":
    run()
