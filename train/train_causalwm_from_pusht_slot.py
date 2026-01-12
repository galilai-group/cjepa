from pathlib import Path

import hydra
# import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.distributed as dist
# from lightning.pytorch.callbacks import Callback, ModelCheckpoint
# from lightning.pytorch.loggers import WandbLogger
# from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_from_disk
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.nn import functional as F
from tqdm import tqdm
import wandb
from custom_models.cjepa_predictor import MaskedSlotPredictor
from einops import rearrange, repeat
from custom_models.dinowm_causal import CausalWM
from videosaur.videosaur import  models

import pickle as pkl
import numpy as np

import os


DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches
OBS_FRAMES = 128
TARGET_LEN = 160


def setup_distributed():
    """Setup distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is the main process."""
    return rank == 0


class PushTSlotDataset(torch.utils.data.Dataset):
    """
    Dataset for pre-extracted slot embeddings from CLEVRER videos.

    Args:
        data: Dict mapping video names to slot tensors, e.g., {'0_pixels.mp4': slots, ...}
              where each slots tensor has shape [num_frames, num_slots, slot_dim] (e.g., [128, 7, 128])
        split: 'train' or 'val'
    """
    def __init__(self, data, split, history_size, num_preds, action_dir, proprio_dir, state_dir, frameskip=1):
        super().__init__()
        self.data = data  # {'0_pixels.mp4': [128, 7, 128], ...}
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.frameskip = frameskip


        # Total number of frames needed per clip (before frameskip)
        self.num_steps = history_size + num_preds
        self.clip_len = self.frameskip * self.num_steps

        with open(action_dir, "rb") as f:
            self.action_meta = pkl.load(f)
        with open(proprio_dir, "rb") as f:
            self.proprio_meta = pkl.load(f)
        with open(state_dir, "rb") as f:
            self.state_meta = pkl.load(f)

        # Compute normalization statistics for action and proprio
        # This matches train_causalwm.py's norm_col_transform behavior
        self._compute_normalization_stats()

        # Build index mapping: list of (video_key, start_frame) tuples
        self.video_keys = list(self.data.keys())
        self.samples = []  # List of (video_key, start_frame) for each valid sample

        for video_key in self.video_keys:
            slots = self.data[video_key]
            num_frames = slots.shape[0]
            # Number of valid starting positions for this video
            # Can start from 0 to (num_frames - clip_len) inclusive
            num_valid_starts = max(0, num_frames - self.clip_len + 1)

            for start in range(num_valid_starts):
                self.samples.append((video_key, start))

    def _compute_normalization_stats(self):
        """Compute mean and std for action and proprio normalization (matching train_causalwm.py).
        
        Key: Compute statistics on RESHAPED action (not raw), matching how VideoDataset.action 
        is used in train_causalwm.py's norm_col_transform.
        """
        # Collect all reshaped actions and proprios from the split
        all_actions_reshaped = []
        all_proprios = []
        
        for video_key in self.action_meta[self.split].keys():
            actions = self.action_meta[self.split][video_key]
            proprios = self.proprio_meta[self.split][video_key]
            
            if isinstance(actions, np.ndarray):
                actions = actions
            else:
                actions = np.array(actions)
                
            if isinstance(proprios, np.ndarray):
                proprios = proprios
            else:
                proprios = np.array(proprios)
            
            # Reshape action to match VideoDataset behavior: (clip_len, action_dim) -> (num_clips, num_steps, frameskip*action_dim)
            # For each video, create all valid clips and reshape their actions
            num_frames = actions.shape[0]
            for start in range(max(0, num_frames - self.clip_len + 1)):
                end_frame = start + self.clip_len
                raw_action = actions[start:end_frame]  # (clip_len, action_dim)
                reshaped_action = raw_action.reshape(self.num_steps, -1)  # (num_steps, frameskip*action_dim)
                all_actions_reshaped.append(reshaped_action)
                
            # For proprio: subsample with frameskip (same as final output)
            for start in range(max(0, num_frames - self.clip_len + 1)):
                end_frame = start + self.clip_len
                clip_proprio = proprios[start:end_frame:self.frameskip]  # (num_steps, proprio_dim)
                all_proprios.append(clip_proprio)
        
        # Stack and compute statistics on RESHAPED forms (matching train_causalwm.py)
        all_actions_reshaped = np.concatenate(all_actions_reshaped, axis=0)  # (total_clips*num_steps, frameskip*action_dim)
        all_proprios = np.concatenate(all_proprios, axis=0)  # (total_clips*num_steps, proprio_dim)
        
        self.action_mean = all_actions_reshaped.mean(axis=0, keepdims=True)  # (1, frameskip*action_dim)
        self.action_std = all_actions_reshaped.std(axis=0, keepdims=True) + 1e-8  # (1, frameskip*action_dim)
        
        self.proprio_mean = all_proprios.mean(axis=0, keepdims=True)  # (1, proprio_dim)
        self.proprio_std = all_proprios.std(axis=0, keepdims=True) + 1e-8  # (1, proprio_dim)
        
        logging.info(f"[{self.split}] Action stats - mean: {self.action_mean.flatten()}, std: {self.action_std.flatten()}")
        logging.info(f"[{self.split}] Proprio stats - mean: {self.proprio_mean.flatten()}, std: {self.proprio_std.flatten()}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_key, start_frame = self.samples[idx]
        slots = self.data[video_key]  # [num_frames, num_slots, slot_dim]
        actions = self.action_meta[self.split][video_key]  # [num_frames, action_dim]
        proprios = self.proprio_meta[self.split][video_key]  # [num_frames, proprio_dim]
        states = self.state_meta[self.split][video_key]  # [num_frames, state_dim]

        # Extract clip with frameskip: frames at start, start+frameskip, start+2*frameskip, ...
        end_frame = start_frame + self.clip_len
        clip_slots = slots[start_frame:end_frame:self.frameskip]  # [num_steps, num_slots, slot_dim]

        # For action: concatenate frameskip consecutive actions into one
        # This matches swm.data.VideoDataset behavior: action.reshape(num_steps, -1)
        # Raw action: (clip_len, action_dim) -> (num_steps, frameskip * action_dim)
        raw_action = actions[start_frame:end_frame]  # [clip_len, action_dim]
        clip_action = raw_action.reshape(self.num_steps, -1)  # [num_steps, frameskip * action_dim]

        # For proprio/state: subsample with frameskip (same as slots)
        clip_proprio = proprios[start_frame:end_frame:self.frameskip]  # [num_steps, proprio_dim]
        clip_state = states[start_frame:end_frame:self.frameskip]  # [num_steps, state_dim]

        # Normalize action and proprio (matching train_causalwm.py behavior)
        # Now action_mean/std are already in reshaped form (1, frameskip*action_dim)
        clip_action = (clip_action - self.action_mean) / self.action_std
        clip_proprio = (clip_proprio - self.proprio_mean) / self.proprio_std

        # Ensure tensor type for slots
        if not isinstance(clip_slots, torch.Tensor):
            clip_slots = torch.tensor(clip_slots, dtype=torch.float32)
        
        clip_action = torch.tensor(clip_action, dtype=torch.float32)
        clip_proprio = torch.tensor(clip_proprio, dtype=torch.float32)
        clip_state = torch.tensor(clip_state, dtype=torch.float32)

        return {"embed": clip_slots,
                "action": clip_action,
                "proprio": clip_proprio,
                "state": clip_state
                }  # [history_size + num_preds, num_slots, slot_dim]


# ============================================================================
# Data Setup
# ============================================================================
def get_data(cfg, is_ddp, world_size, rank):

    # open pickle file to get train and val splits
    with open(cfg.embedding_dir, "rb") as f:
        data = pkl.load(f)  # data is slot embedding, shape of frame x num_slots x slot_dim per video

    train_dataset = PushTSlotDataset(
        data=data["train"],
        split="train",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        action_dir=cfg.action_dir,
        proprio_dir=cfg.proprio_dir,
        state_dir=cfg.state_dir,
        frameskip=cfg.frameskip
    )

    val_dataset = PushTSlotDataset(
        data=data["val"],
        split="val",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        action_dir=cfg.action_dir,
        proprio_dir=cfg.proprio_dir,
        state_dir=cfg.state_dir,
        frameskip=cfg.frameskip
    )

    # Setup samplers for DDP
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, data, train_sampler


# ============================================================================
# Model Architecture
# ============================================================================
def get_world_model(cfg):
    """Build world model: masked slot predictor."""

    effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
    action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=cfg.dinowm.action_embed_dim)
    proprio_encoder = swm.wm.dinowm.Embedder(in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.dinowm.proprio_embed_dim)

    embedding_dim = cfg.videosaur.SLOT_DIM + cfg.dinowm.action_embed_dim + cfg.dinowm.proprio_embed_dim
    logging.info(f"Total embedding dimension (slot + action + proprio): {embedding_dim}")
    predictor = MaskedSlotPredictor(
        num_slots=cfg.videosaur.NUM_SLOTS,  # S: number of slots
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

    return action_encoder, proprio_encoder, predictor


# ============================================================================
# Training Setup
# ============================================================================
def setup_wandb(cfg, rank):
    """Setup wandb logger (only on main process)."""
    if not cfg.wandb.enable or not is_main_process(rank):
        return None

    wandb_run_id = cfg.wandb.get("run_id", None)
    wandb.init(
        name=cfg.wandb.get("name", "causalwm_from_slot"),
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        resume="allow" if wandb_run_id else None,
        id=wandb_run_id,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return wandb


def compute_loss(predictor, action_enc, proprio_enc, batch, cfg, device, inference=False):
    """Compute loss for a batch."""
    embed = batch["embed"].to(device)  # (B, T, S, D)
    pixels_dim = embed.shape[-1]

    actions = batch["action"].to(device)  # (B, T, action_dim)
    proprios = batch["proprio"].to(device)  # (B, T, proprio_dim)
    action_embeddings = action_enc(actions)  # (B, T, action_embed_dim)
    proprio_embeddings = proprio_enc(proprios)  # (B, T, proprio_embed_dim)
    proprio_dim = proprio_embeddings.shape[-1]

    n_patches = embed.shape[2]  # number of slots

    proprio_tiled = repeat(proprio_embeddings.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches)
    embed = torch.cat([embed, proprio_tiled], dim=3)

    action_tiled = repeat(action_embeddings.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches)
    embed = torch.cat([embed, action_tiled], dim=3)

    # Split into history and target
    history = embed[:, :cfg.dinowm.history_size, :, :]  # (B, history_size, S, D)
    target = embed[:, cfg.dinowm.history_size:cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]  # (B, num_preds, S, D)

    # Forward pass
    if inference:
        pred_output = predictor.inference(history) # only future prediction
        # Loss on future prediction
        loss_future = F.mse_loss(pred_output[..., :pixels_dim], target[..., :pixels_dim].detach())

        losses = {}
        losses["loss_future"] = loss_future

        loss_masked_history = torch.tensor(0.0, device=device)
        losses["loss_masked_history"] = loss_masked_history

        proprio_loss = F.mse_loss(
            pred_output[..., pixels_dim : pixels_dim + proprio_dim],
            target[..., pixels_dim : pixels_dim + proprio_dim].detach(),
        )
        losses["loss_proprio"] = proprio_loss

        # Total loss
        total_loss = loss_masked_history + loss_future + proprio_loss
        losses["loss"] = total_loss
        
    else:
        pred_output = predictor(history)
        pred_embedding, mask_indices = pred_output

        # pred_embedding: (B, history_size + num_preds, S, D)
        pred_history = pred_embedding[:, :cfg.dinowm.history_size, :, :]
        pred_future = pred_embedding[:, cfg.dinowm.history_size:cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]

        losses = {}

        if len(mask_indices) > 0:
            # Loss on masked slots in history
            loss_masked_history = F.mse_loss(
                pred_history[:, :, mask_indices, :pixels_dim],
                history[:, :, mask_indices, :pixels_dim].detach()
            )
            losses["loss_masked_history"] = loss_masked_history
        else:
            loss_masked_history = torch.tensor(0.0, device=device)
            losses["loss_masked_history"] = loss_masked_history

        # Loss on future prediction
        loss_future = F.mse_loss(pred_future[..., :pixels_dim], target[..., :pixels_dim].detach())
        losses["loss_future"] = loss_future

        proprio_loss = F.mse_loss(
            pred_future[..., pixels_dim : pixels_dim + proprio_dim],
            target[..., pixels_dim : pixels_dim + proprio_dim].detach(),
        )
        losses["loss_proprio"] = proprio_loss

        # Total loss
        total_loss = loss_masked_history + loss_future + proprio_loss
        losses["loss"] = total_loss

    return losses


@torch.no_grad()
def validate(predictor, action_enc, proprio_enc, val_loader, cfg, device, world_size):
    """Run validation and return average loss."""
    predictor.eval()
    total_loss = 0.0
    total_loss_future = 0.0
    total_loss_masked = 0.0
    total_loss_proprio = 0.0
    num_batches = 0

    for batch in val_loader:
        losses = compute_loss(predictor, action_enc, proprio_enc, batch, cfg, device, inference=True)
        total_loss += losses["loss"].item()
        total_loss_future += losses["loss_future"].item()
        total_loss_masked += losses["loss_masked_history"].item()
        total_loss_proprio += losses["loss_proprio"].item()

        num_batches += 1

    # Average across batches
    avg_loss = total_loss / max(num_batches, 1)
    avg_loss_future = total_loss_future / max(num_batches, 1)
    avg_loss_masked = total_loss_masked / max(num_batches, 1)
    avg_loss_proprio = total_loss_proprio / max(num_batches, 1)

    # Reduce across processes if DDP
    if world_size > 1:
        loss_tensor = torch.tensor([avg_loss, avg_loss_future, avg_loss_masked, avg_loss_proprio, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor[0].item() / world_size
        avg_loss_future = loss_tensor[1].item() / world_size
        avg_loss_masked = loss_tensor[2].item() / world_size
        avg_loss_proprio = loss_tensor[3].item() / world_size

    return {
        "val/loss": avg_loss,
        "val/loss_future": avg_loss_future,
        "val/loss_masked_history": avg_loss_masked,
        "val/loss_proprio": avg_loss_proprio,
    }


# ============================================================================
# Main Entry Point
# ============================================================================
@hydra.main(version_base=None, config_path="../configs", config_name="config_train_causal_pusht_slot")
def run(cfg):
    """Run training of predictor"""

    ############### to save trained model in swm format later ##############
    model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = model.encoder 
    slot_attention = model.processor 
    initializer = model.initializer
    #########################################################################

    # Setup distributed training
    is_ddp, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        logging.info(f"DDP: {is_ddp}, Rank: {rank}, World Size: {world_size}, Device: {device}")

    # Setup cache directory
    cache_dir = swm.data.utils.get_cache_dir() if cfg.cache_dir is None else cfg.cache_dir

    # Setup wandb (only on main process)
    wandb_logger = setup_wandb(cfg, rank)

    # Get data
    train_loader, val_loader, data, train_sampler = get_data(cfg, is_ddp, world_size, rank)

    if is_main_process(rank):
        logging.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Build model
    action_encoder, proprio_encoder, predictor = get_world_model(cfg)
    action_encoder = action_encoder.to(device)
    proprio_encoder = proprio_encoder.to(device)
    predictor = predictor.to(device)

    # Wrap with DDP if needed
    if is_ddp:
        action_encoder = torch.nn.parallel.DistributedDataParallel(
            action_encoder, device_ids=[local_rank], output_device=local_rank
        )
        proprio_encoder = torch.nn.parallel.DistributedDataParallel(
            proprio_encoder, device_ids=[local_rank], output_device=local_rank
        )
        predictor = torch.nn.parallel.DistributedDataParallel(
            predictor, device_ids=[local_rank], output_device=local_rank
        )

    # Setup optimizer
    model_params = predictor.module.parameters() if is_ddp else predictor.parameters()
    optimizer = torch.optim.AdamW(model_params, lr=cfg.predictor_lr)

    action_encoder_params = action_encoder.module.parameters() if is_ddp else action_encoder.parameters()
    proprio_encoder_params = proprio_encoder.module.parameters() if is_ddp else proprio_encoder.parameters()
    action_optimizer = torch.optim.AdamW(action_encoder_params, lr=cfg.action_encoder_lr)
    proprio_optimizer = torch.optim.AdamW(proprio_encoder_params, lr=cfg.proprio_encoder_lr)

   
    # Training loop
    log_every_n_epochs = cfg.get("log_every_n_epochs", 1)
    global_step = 0

    for epoch in range(cfg.trainer.max_epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        predictor.train()
        action_encoder.train()
        proprio_encoder.train()
        epoch_loss = 0.0
        epoch_loss_future = 0.0
        epoch_loss_masked = 0.0
        epoch_loss_proprio = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process(rank))
        for batch in pbar:
            optimizer.zero_grad()

            # Compute loss
            losses = compute_loss(
                predictor.module if is_ddp else predictor,
                action_encoder.module if is_ddp else action_encoder,
                proprio_encoder.module if is_ddp else proprio_encoder,
                batch, cfg, device
            )

            # Backward pass
            losses["loss"].backward()
            optimizer.step()
            action_optimizer.step()
            proprio_optimizer.step()

            # Accumulate metrics
            epoch_loss += losses["loss"].item()
            epoch_loss_future += losses["loss_future"].item()
            epoch_loss_masked += losses["loss_masked_history"].item()
            epoch_loss_proprio += losses["loss_proprio"].item()
            num_batches += 1
            global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['loss'].item():.4f}",
                "future": f"{losses['loss_future'].item():.4f}",
            })

            # Log to wandb (every N steps)
            if wandb_logger is not None and global_step % cfg.get("log_every_n_steps", 10) == 0:
                wandb_logger.log({
                    "train/loss": losses["loss"].item(),
                    "train/loss_future": losses["loss_future"].item(),
                    "train/loss_masked_history": losses["loss_masked_history"].item(),
                    "train/loss_proprio": losses["loss_proprio"].item(),
                    "train/step": global_step,
                    "train/epoch": epoch,
                })

        # Epoch-level metrics
        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_train_loss_future = epoch_loss_future / max(num_batches, 1)
        avg_train_loss_masked = epoch_loss_masked / max(num_batches, 1)
        avg_train_loss_proprio = epoch_loss_proprio / max(num_batches, 1)

        if is_main_process(rank):
            logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Future: {avg_train_loss_future:.4f}, Masked: {avg_train_loss_masked:.4f}, Proprio: {avg_train_loss_proprio:.4f}")

        # Validation
        if (epoch + 1) % log_every_n_epochs == 0:
            val_metrics = validate(
                predictor.module if is_ddp else predictor,
                action_encoder.module if is_ddp else action_encoder,
                proprio_encoder.module if is_ddp else proprio_encoder,
                val_loader, cfg, device, world_size
            )

            if is_main_process(rank):
                logging.info(f"Epoch {epoch+1}: Val Loss: {val_metrics['val/loss']:.4f}")

                if wandb_logger is not None:
                    wandb_logger.log({
                        **val_metrics,
                        "train/epoch_loss": avg_train_loss,
                        "train/epoch_loss_future": avg_train_loss_future,
                        "train/epoch_loss_masked_history": avg_train_loss_masked,
                        "train/epoch_loss_proprio": avg_train_loss_proprio,
                        "epoch": epoch + 1,
                    })

        # Save checkpoint (only on main process)
        if is_main_process(rank):
            # Build CausalWM in the same format as train_causalwm.py
            predictor_state = predictor.module.state_dict() if is_ddp else predictor.state_dict()
            action_encoder_state = action_encoder.module.state_dict() if is_ddp else action_encoder.state_dict()
            proprio_encoder_state = proprio_encoder.module.state_dict() if is_ddp else proprio_encoder.state_dict()

            # Create a fresh CausalWM and load trained weights
            world_model_to_save = CausalWM(
                encoder=spt.backbone.EvalOnly(encoder),
                slot_attention=spt.backbone.EvalOnly(slot_attention),
                initializer=spt.backbone.EvalOnly(initializer),
                predictor=MaskedSlotPredictor(
                    num_slots=cfg.videosaur.NUM_SLOTS,
                    slot_dim=cfg.videosaur.SLOT_DIM + cfg.dinowm.action_embed_dim + cfg.dinowm.proprio_embed_dim,
                    history_frames=cfg.dinowm.history_size,
                    pred_frames=cfg.dinowm.num_preds,
                    num_masked_slots=cfg.get("num_masked_slots", 2),
                    seed=cfg.seed,
                    depth=cfg.predictor.get("depth", 6),
                    heads=cfg.predictor.get("heads", 16),
                    dim_head=cfg.predictor.get("dim_head", 64),
                    mlp_dim=cfg.predictor.get("mlp_dim", 2048),
                    dropout=cfg.predictor.get("dropout", 0.1),
                ),
                action_encoder=swm.wm.dinowm.Embedder(
                    in_chans=cfg.frameskip * cfg.dinowm.action_dim,
                    emb_dim=cfg.dinowm.action_embed_dim
                ),
                proprio_encoder=swm.wm.dinowm.Embedder(
                    in_chans=cfg.dinowm.proprio_dim,
                    emb_dim=cfg.dinowm.proprio_embed_dim
                ),
                history_size=cfg.dinowm.history_size,
                num_pred=cfg.dinowm.num_preds,
            )

            # Load trained weights into CausalWM
            world_model_to_save.predictor.load_state_dict(predictor_state)
            world_model_to_save.action_encoder.load_state_dict(action_encoder_state)
            world_model_to_save.proprio_encoder.load_state_dict(proprio_encoder_state)
            # Wrap in spt.Module (same format as train_causalwm.py)
            def add_opt(module_name, lr):
                return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}

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
                embedding = batch["embed"][:, : cfg.dinowm.history_size, :, :]  # (B, history_size, S, 64)
                

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

                            # Add proprioception loss if available
                    if proprio_key is not None:
                        proprio_dim = batch["proprio_embed"].shape[-1]
                        proprio_loss = F.mse_loss(
                            pred_future[..., pixels_dim : pixels_dim + proprio_dim],
                            target_embedding[..., pixels_dim : pixels_dim + proprio_dim].detach(),
                        )
                        batch["proprio_loss"] = proprio_loss
                        batch["loss"] = loss_masked_history + loss_future + proprio_loss
                    else:
                        batch["loss"] = loss_masked_history + loss_future
                    batch["loss_masked_history"] = loss_masked_history
                    batch["loss_future"] = loss_future
                else :
                    pred_embedding = pred_output[0]
                    pred_future = pred_embedding[:, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]       # (B, num_pred, S, 64)
                    target_embedding = batch["embed"][:, cfg.dinowm.history_size : cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]  # (B, num_pred, S, 64)
                    loss_future = F.mse_loss(pred_future[..., :pixels_dim], target_embedding[..., :pixels_dim].detach())
                    if proprio_key is not None:
                        proprio_dim = batch["proprio_embed"].shape[-1]
                        proprio_loss = F.mse_loss(
                            pred_future[..., pixels_dim : pixels_dim + proprio_dim],
                            target_embedding[..., pixels_dim : pixels_dim + proprio_dim].detach(),
                        )
                        batch["proprio_loss"] = proprio_loss
                        batch["loss"] = loss_future + proprio_loss
                    else:
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
            world_model_wrapped = spt.Module(
                model=world_model_to_save,
                forward=forward,
                optim={
                    "predictor_opt": add_opt("model.predictor", cfg.predictor_lr),
                    "proprio_opt": add_opt("model.proprio_encoder", cfg.proprio_encoder_lr),
                    "action_opt": add_opt("model.action_encoder", cfg.action_encoder_lr),
                },
            )

            checkpoint_path = os.path.join(cache_dir, f"{cfg.output_model_name}_epoch_{epoch+1}_object.ckpt")
            torch.save(world_model_wrapped, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

   
   
    # Cleanup
    if wandb_logger is not None:
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    run()
