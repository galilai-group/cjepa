from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torchvision
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel
import wandb
from videosaur import  models
from data import VideoStepsDataset



DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches


# ============================================================================
# Data Setup
# ============================================================================
def get_data(cfg):
    """Setup dataset with image transforms and normalization."""
    
    # Image size must be multiple of DINO patch size (14)
    # img_size = (cfg.image_size // cfg.patch_size) * DINO_PATCH_SIZE
    img_size = cfg.image_size

    transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.CenterCrop(img_size),
                torchvision.transforms.Resize(img_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    train_set = VideoStepsDataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        cache_dir=None,
        split="train",
        transform=transform
    )

    val_set = VideoStepsDataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        cache_dir=None,
        split="validation",
        transform=transform
    )
    
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
    """Build world model: frozen DINO encoder + trainable causal predictor."""

    def forward(self, batch, stage):
        """Forward: encode observations, predict next states, compute losses."""

        #[8, 4, 3, 518, 518]
        # Encode all timesteps into latent embeddings
        batch = self.model.encode(
            batch,
            target="embed",
            pixels_key="pixels"
        )

        # Use history to predict next states
        embedding = batch["embed"][:, : cfg.dinowm.history_size, :, :]  # (B, T-1, patches, dim)
        pred_embedding = self.model.predict(embedding)
        target_embedding = batch["embed"][:, -cfg.dinowm.num_preds :, :, :]  # (B, T-1, patches, dim)

        # Compute pixel latent prediction loss
        pixels_dim = batch["pixels_embed"].shape[-1]
        batch["loss"] = F.mse_loss(pred_embedding, target_embedding.detach())

        # Log all losses
        prefix = "train/" if self.training else "val/"
        losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "_loss" in k}
        self.log_dict(losses_dict, on_step=True, sync_dist=True)  # , on_epoch=True, sync_dist=True)

        return batch

    # Load frozen DINO encoder
    model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = model.encoder # Mapovertime > FrameEncoder > {backbone, output_transform}         # B T C H W > # B T N D
    slot_attention = model.processor # ScanOverTime > LatentProcessor > {Corrector, Predictor}      
    initializer = model.initializer
    embedding_dim = cfg.videosaur.SLOT_DIM #encoder.config.hidden_size

    # num_patches = (cfg.image_size // cfg.patch_size) ** 2
    num_patches = cfg.videosaur.NUM_SLOTS

    if cfg.training_type == "wm":
        embedding_dim += cfg.dinowm.proprio_embed_dim + cfg.dinowm.action_embed_dim  # Total embedding size

    logging.info(f"Patches: {num_patches}, Embedding dim: {embedding_dim}")

    # Build causal predictor (transformer that predicts next latent states)
    predictor = swm.wm.dinowm.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim,
        **cfg.predictor,
    )

    # Build action and proprioception encoders
    if cfg.training_type == "video":    
        action_encoder = None
        proprio_encoder = None

        logging.info(f"[Video Only] Action encoder: None, Proprio encoder: None")
    else :
        effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
        action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=cfg.dinowm.action_embed_dim)
        proprio_encoder = swm.wm.dinowm.Embedder(in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.dinowm.proprio_embed_dim)

        logging.info(f"Action dim: {effective_act_dim}, Proprio dim: {cfg.dinowm.proprio_dim}")

    # Assemble world model
    world_model = swm.wm.OCWM(
        encoder=encoder,
        slot_attention=slot_attention,
        initializer = initializer,
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
@hydra.main(version_base=None, config_path="./", config_name="config_oc")
def run(cfg):
    """Run training of predictor"""

    wandb_logger = setup_pl_logger(cfg)
    data = get_data(cfg)
    world_model = get_world_model(cfg)

    cache_dir = swm.data.get_cache_dir()
    dump_object_callback = ModelObjectCallBack(
        dirpath=cache_dir,
        filename=cfg.output_model_name,
        epoch_interval=10,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[dump_object_callback],
        num_sanity_val_steps=1,
        logger=wandb_logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data,
        ckpt_path=f"{cache_dir}/{cfg.output_model_name}_weights.ckpt",
    )
    manager()


if __name__ == "__main__":
    run()