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
from torch.utils.data import Dataset, DataLoader
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.nn import functional as F
from einops import rearrange, repeat
from custom_models.cjepa_predictor import MaskedSlotPredictor
from custom_models.dinowm_oc import OCWM
from videosaur.videosaur import models

import pickle as pkl
import numpy as np

import os


# ============================================================================
# Dataset for Pre-extracted Slot Representations
# ============================================================================
class PushTSlotDataset(Dataset):
    """
    Dataset for pre-extracted slot representations from PushT.
    
    This class mirrors the behavior of swm.data.VideoDataset to ensure
    identical data processing. Key behaviors:
    - Window stride of 1 (not frameskip) for sample indices
    - Action is reshaped to (T, action_dim * frameskip) by VideoDataset
    - Normalization uses mean/std without clamping (same as WrapTorchTransform)
    - nan_to_num is only applied in forward pass, not in dataset
    
    Each sample contains:
    - pixels_embed: Pre-extracted slot embeddings (T, num_slots, slot_dim)
    - action: Action sequence (T, action_dim * frameskip)
    - proprio: Proprioception sequence (T, proprio_dim)
    - state: State sequence (T, state_dim) [optional, for evaluation]
    
    Args:
        slot_data: Dict mapping video_id to slot embeddings
        split: 'train' or 'val'
        history_size: Number of history frames
        num_preds: Number of future frames to predict
        action_dir: Path to action pickle file
        proprio_dir: Path to proprioception pickle file
        state_dir: Path to state pickle file (optional)
        frameskip: Frame skip factor (affects action reshaping)
        seed: Random seed for sampling
    """
    
    def __init__(
        self,
        slot_data: dict,
        split: str,
        history_size: int,
        num_preds: int,
        action_dir: str,
        proprio_dir: str,
        state_dir: str = None,
        frameskip: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.slot_data = slot_data
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.frameskip = frameskip
        self.n_steps = history_size + num_preds
        self.seed = seed
        
        # Load action and proprio data
        with open(action_dir, "rb") as f:
            action_data = pkl.load(f)
        self.action_data = action_data[split]
        
        with open(proprio_dir, "rb") as f:
            proprio_data = pkl.load(f)
        self.proprio_data = proprio_data[split]
        
        # State is optional (used for evaluation)
        self.state_data = None
        if state_dir is not None:
            with open(state_dir, "rb") as f:
                state_data = pkl.load(f)
            self.state_data = state_data[split]
        
        # Build index: list of (video_id, start_frame) tuples
        self.samples = self._build_sample_index()
        
        # Compute normalization statistics (matching WrapTorchTransform behavior)
        self._compute_normalization_stats()
        
        logging.info(f"[{split}] Created dataset with {len(self.samples)} samples from {len(self.slot_data)} videos")
    
    def _build_sample_index(self):
        """
        Build list of valid (video_id, start_frame) samples.
        
        Matches VideoDataset behavior: stride of 1, not frameskip.
        VideoDataset uses: episode_max_end = max(0, len(ep) - clip_len + 1)
        and iterates over all start positions with stride 1.
        """
        samples = []
        clip_len = self.n_steps * self.frameskip
        
        for video_id, slots in self.slot_data.items():
            num_frames = slots.shape[0]
            # max_start is inclusive, so we can start at positions 0 to max_start
            max_start = num_frames - clip_len
            
            if max_start < 0:
                continue
            
            # Stride 1 matching VideoDataset behavior
            for start_idx in range(0, max_start + 1):
                samples.append((video_id, start_idx))
        
        return samples
    
    def _compute_normalization_stats(self):
        """
        Compute mean and std for action and proprio normalization.
        
        Matches WrapTorchTransform(norm_col_transform(dataset, col)) behavior:
        - Computes stats over the RESHAPED action column (T, action_dim * frameskip)
        - No clamping of std (WrapTorchTransform doesn't clamp)
        - Uses tensor mean/std with unsqueeze(0)
        
        Note: VideoDataset reshapes action to (T, -1) before transform is applied.
        """
        # Collect all actions and proprios in their RESHAPED form
        # This matches how VideoDataset provides data to the transform
        all_actions = []
        all_proprios = []
        
        for video_id in self.action_data.keys():
            action_raw = self.action_data[video_id]  # (num_frames, action_dim)
            # Reshape to match VideoDataset's reshape: (T, action_dim * frameskip)
            # VideoDataset does: steps["action"].reshape(act_shape, -1)
            # where act_shape = num_steps and the raw actions are clip_len = n_steps * frameskip
            # So each T gets frameskip consecutive actions flattened
            num_frames = action_raw.shape[0]
            clip_len = self.n_steps * self.frameskip
            
            # Iterate over all possible clips (stride 1, matching _build_sample_index)
            for start_idx in range(0, num_frames - clip_len + 1):
                # Get clip_len consecutive raw actions
                action_clip = action_raw[start_idx:start_idx + clip_len]  # (clip_len, action_dim)
                # Reshape to (n_steps, action_dim * frameskip) - matching VideoDataset
                action_reshaped = action_clip.reshape(self.n_steps, -1)
                all_actions.append(action_reshaped)
        
        for video_id in self.proprio_data.keys():
            proprio_raw = self.proprio_data[video_id]  # (num_frames, proprio_dim)
            num_frames = proprio_raw.shape[0]
            clip_len = self.n_steps * self.frameskip
            
            for start_idx in range(0, num_frames - clip_len + 1):
                # Get frames with frameskip (matching VideoDataset: data[::frameskip])
                frame_indices = [start_idx + i * self.frameskip for i in range(self.n_steps)]
                if frame_indices[-1] < num_frames:
                    proprio_clip = proprio_raw[frame_indices]  # (n_steps, proprio_dim)
                    all_proprios.append(proprio_clip)
        
        # Stack and compute stats matching norm_col_transform:
        # data.mean(0).unsqueeze(0), data.std(0).unsqueeze(0)
        all_actions = torch.from_numpy(np.concatenate(all_actions, axis=0)).float()  # (N*T, action_dim*frameskip)
        all_proprios = torch.from_numpy(np.concatenate(all_proprios, axis=0)).float()  # (N*T, proprio_dim)
        
        # Match norm_col_transform: mean(0).unsqueeze(0), std(0).unsqueeze(0)
        self.action_mean = all_actions.mean(0).unsqueeze(0)  # (1, action_dim * frameskip)
        self.action_std = all_actions.std(0).unsqueeze(0)    # (1, action_dim * frameskip)
        
        self.proprio_mean = all_proprios.mean(0).unsqueeze(0)  # (1, proprio_dim)
        self.proprio_std = all_proprios.std(0).unsqueeze(0)    # (1, proprio_dim)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_id, start_idx = self.samples[idx]
        
        # clip_len = n_steps * frameskip raw frames
        clip_len = self.n_steps * self.frameskip
        
        # Get frame indices with frameskip for slots (matching VideoDataset: data[::frameskip])
        frame_indices = [start_idx + i * self.frameskip for i in range(self.n_steps)]
        
        # Extract slot embeddings: (n_steps, num_slots, slot_dim)
        slots = self.slot_data[video_id]
        pixels_embed = torch.from_numpy(slots[frame_indices]).float()
        
        # Extract and reshape actions (matching VideoDataset behavior)
        # VideoDataset gets clip_len consecutive raw actions, then reshapes to (n_steps, -1)
        action_raw = self.action_data[video_id]
        action_clip = action_raw[start_idx:start_idx + clip_len]  # (clip_len, action_dim)
        # Reshape to (n_steps, action_dim * frameskip) - matching VideoDataset's reshape
        action = torch.from_numpy(action_clip.reshape(self.n_steps, -1)).float()
        
        # Extract proprio with frameskip (matching VideoDataset: data[::frameskip])
        proprio_raw = self.proprio_data[video_id]
        proprio = torch.from_numpy(proprio_raw[frame_indices]).float()
        
        # Normalize action and proprio (matching WrapTorchTransform behavior)
        # Note: No nan_to_num here - that's done in forward pass like train_causalwm.py
        action = (action - self.action_mean) / self.action_std
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        
        sample = {
            "pixels_embed": pixels_embed,  # (T, S, D)
            "action": action,              # (T, action_dim * frameskip)
            "proprio": proprio,            # (T, proprio_dim)
        }
        
        # Optionally include state
        if self.state_data is not None:
            state_raw = self.state_data[video_id]
            state = torch.from_numpy(state_raw[frame_indices]).float()
            sample["state"] = state
        
        return sample


# ============================================================================
# Data Setup
# ============================================================================
def get_data(cfg):
    """Setup dataset with pre-extracted slot representations."""
    
    # Load pre-extracted slot embeddings
    with open(cfg.embedding_dir, "rb") as f:
        slot_data = pkl.load(f)
    
    logging.info(f"Loaded slot embeddings from {cfg.embedding_dir}")
    
    train_dataset = PushTSlotDataset(
        slot_data=slot_data["train"],
        split="train",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        action_dir=cfg.action_dir,
        proprio_dir=cfg.proprio_dir,
        state_dir=cfg.get("state_dir", None),
        frameskip=cfg.frameskip,
        seed=cfg.seed,
    )
    
    val_dataset = PushTSlotDataset(
        slot_data=slot_data["val"],
        split="val",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        action_dir=cfg.action_dir,
        proprio_dir=cfg.proprio_dir,
        state_dir=cfg.get("state_dir", None),
        frameskip=cfg.frameskip,
        seed=cfg.seed,
    )
    
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    
    logging.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    
    return spt.data.DataModule(train=train_loader, val=val_loader)


# ============================================================================
# Model Architecture
# ============================================================================
def get_world_model(cfg):
    """
    Build world model: masked slot predictor with action/proprio encoders.
    
    Unlike train_causalwm.py, we don't need the DINO encoder and slot attention
    since we're using pre-extracted slots. However, we create placeholder modules
    to maintain checkpoint compatibility.
    """
    
    def forward(self, batch, stage):
        """
        Forward pass using pre-extracted slot embeddings.
        
        This mirrors the forward in train_causalwm.py but skips encoding.
        """
        proprio_key = "proprio" if "proprio" in batch else None
        
        # Replace NaN values with 0 (occurs at sequence boundaries)
        # This matches train_causalwm.py behavior
        if proprio_key is not None:
            batch[proprio_key] = torch.nan_to_num(batch[proprio_key], 0.0)
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        
        # Pre-extracted slots are already in the batch as 'pixels_embed'
        # Shape: (B, T, S, D) where D is slot_dim
        pixels_embed = batch["pixels_embed"]  # Pre-extracted slots
        B, T, S, slot_dim = pixels_embed.shape
        
        batch["pixels_embed"] = pixels_embed
        
        # Encode action and proprio (still need to train these)
        embedding = pixels_embed
        n_patches = S
        
        if proprio_key is not None:
            proprio = batch[proprio_key].float()  # (B, T, proprio_dim)
            proprio_embed = self.model.proprio_encoder(proprio)  # (B, T, proprio_embed_dim)
            batch["proprio_embed"] = proprio_embed
            
            # Tile proprio across slots
            proprio_tiled = repeat(proprio_embed.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches)
            embedding = torch.cat([embedding, proprio_tiled], dim=-1)
        if "action" in batch:
            action = batch["action"].float()  # (B, T, action_dim * frameskip)
            action_embed = self.model.action_encoder(action)  # (B, T, action_embed_dim)
            batch["action_embed"] = action_embed
            
            # Tile action across slots
            action_tiled = repeat(action_embed.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches)
            embedding = torch.cat([embedding, action_tiled], dim=-1)
        
        
        batch["embed"] = embedding  # (B, T, S, D_total)
        
        # Use history to predict next states
        history_embed = embedding[:, :cfg.dinowm.history_size, :, :]  # (B, history_size, S, D_total)
        
        # Predict with masking
        pred_embedding = self.model.predict(history_embed)
        # target_embedding = batch["embed"][:, cfg.dinowm.num_preds :, :, :]  # (B, T-1, patches, dim)
        target_embedding = batch["embed"][:, cfg.dinowm.num_preds :, :, :]

        # Compute pixel latent prediction loss
        pixels_dim = batch["pixels_embed"].shape[-1]
        if "clevrer" in cfg.dataset_name:
            batch["loss"] = F.mse_loss(pred_embedding, target_embedding.detach())
            B, num_pred, S, D = pred_embedding.shape
            pred_flat = pred_embedding.reshape(B*num_pred, S*D)
            batch["predictor_embed"] = pred_flat
        elif "pusht" in cfg.dataset_name:
            pixels_loss = F.mse_loss(pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim].detach())
            batch["pixels_loss"] = pixels_loss

            # Add proprioception loss if available
            if proprio_key is not None:
                proprio_dim = batch["proprio_embed"].shape[-1]
                proprio_loss = F.mse_loss(
                    pred_embedding[..., pixels_dim : pixels_dim + proprio_dim],
                    target_embedding[..., pixels_dim : pixels_dim + proprio_dim].detach(),
                )
                batch["proprio_loss"] = proprio_loss

            batch["loss"] = F.mse_loss(
                pred_embedding[..., : pixels_dim + proprio_dim],
                target_embedding[..., : pixels_dim + proprio_dim].detach(),
            )
            B, num_pred, S, D = pred_embedding[..., : pixels_dim].shape
            pred_flat = pred_embedding[..., : pixels_dim].reshape(B*num_pred, S*D)
            batch["predictor_embed"] = pred_flat
        # Log losses
        prefix = "train/" if self.training else "val/"
        losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "loss" in k}
        self.log_dict(losses_dict, on_step=True, sync_dist=True)
        
        return batch

    model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = model.encoder
    slot_attention = model.processor
    initializer = model.initializer
    
    # Slot dimension from config
    slot_dim = cfg.videosaur.SLOT_DIM
    num_slots = cfg.videosaur.NUM_SLOTS
    
    # Total embedding dimension (slot + action + proprio)
    embedding_dim = slot_dim + cfg.dinowm.proprio_embed_dim + cfg.dinowm.action_embed_dim
    
    logging.info(f"Num slots: {num_slots}, Slot dim: {slot_dim}, Total embedding dim: {embedding_dim}")

    # Build causal predictor (transformer that predicts next latent states)
    predictor = swm.wm.dinowm.CausalPredictor(
        num_patches=num_slots,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim,
        **cfg.predictor,
    )
    
    # Build action and proprioception encoders (will be trained)
    effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
    action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=cfg.dinowm.action_embed_dim)
    proprio_encoder = swm.wm.dinowm.Embedder(in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.dinowm.proprio_embed_dim)

    logging.info(f"Action dim: {effective_act_dim}, Proprio dim: {cfg.dinowm.proprio_dim}")

    # Assemble world model
    world_model = OCWM(
        encoder=spt.backbone.EvalOnly(encoder),
        slot_attention=spt.backbone.EvalOnly(slot_attention),
        initializer=spt.backbone.EvalOnly(initializer),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
    )
    
    # Wrap in spt.Module with separate optimizers for each trainable component
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
    """Setup WandB logger for PyTorch Lightning."""
    if not cfg.wandb.enable:
        return None
    
    wandb_run_id = cfg.wandb.get("run_id", None)
    wandb_logger = WandbLogger(
        name="oc_wm_slot",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        resume="allow" if wandb_run_id else None,
        id=wandb_run_id,
        log_model=False,
    )
    
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return wandb_logger


class ModelObjectCallBack(Callback):
    """Callback to save model object after each epoch (same as train_causalwm.py)."""
    
    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)
        
        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                output_path = self.dirpath / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
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
@hydra.main(version_base=None, config_path="../configs", config_name="config_train_oc_pusht_slot")
def run(cfg):
    """Run training of predictor using pre-extracted slot representations."""
    
    # Setup cache directory
    cache_dir = Path(swm.data.utils.get_cache_dir() if cfg.cache_dir is None else cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    wandb_logger = setup_pl_logger(cfg)
    
    # Load data
    data = get_data(cfg)
    
    # Build world model
    world_model = get_world_model(cfg)
    
    # Setup callbacks
    dump_object_callback = ModelObjectCallBack(
        dirpath=cache_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )
    
    callbacks = [dump_object_callback]
    

    
    # Setup trainer
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=wandb_logger,
        enable_checkpointing=True,
    )
    
    # Run training
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data,
        ckpt_path=str(cache_dir / f"{cfg.output_model_name}_weights.ckpt"),
        seed=cfg.seed,
    )
    manager()


if __name__ == "__main__":
    run()
