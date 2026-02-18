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

from loguru import logger as logging
from omegaconf import OmegaConf
from torch.nn import functional as F
from tqdm import tqdm
import wandb
from src.cjepa_predictor import MaskedSlotPredictor

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


class ClevrerSlotDataset(torch.utils.data.Dataset):
    """
    Dataset for pre-extracted slot embeddings from CLEVRER videos.

    Args:
        data: Dict mapping video names to slot tensors, e.g., {'0_pixels.mp4': slots, ...}
              where each slots tensor has shape [num_frames, num_slots, slot_dim] (e.g., [128, 7, 128])
        split: 'train' or 'val'
    """
    def __init__(self, data, split, history_size, num_preds, frameskip=1):
        super().__init__()
        self.data = data  # {'0_pixels.mp4': [128, 7, 128], ...}
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.frameskip = frameskip

        # Total number of frames needed per clip (before frameskip)
        self.num_steps = history_size + num_preds
        self.clip_len = self.frameskip * self.num_steps

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_key, start_frame = self.samples[idx]
        slots = self.data[video_key]  # [num_frames, num_slots, slot_dim]

        # Extract clip with frameskip: frames at start, start+frameskip, start+2*frameskip, ...
        end_frame = start_frame + self.clip_len
        clip_slots = slots[start_frame:end_frame:self.frameskip]  # [num_steps, num_slots, slot_dim]

        # Ensure tensor type
        if not isinstance(clip_slots, torch.Tensor):
            clip_slots = torch.tensor(clip_slots, dtype=torch.float32)

        return {"embed": clip_slots}  # [history_size + num_preds, num_slots, slot_dim]


# ============================================================================
# Data Setup
# ============================================================================
def get_data(cfg, is_ddp, world_size, rank):

    # open pickle file to get train and val splits
    with open(cfg.embedding_dir, "rb") as f:
        data = pkl.load(f)  # data is slot embedding, shape of frame x num_slots x slot_dim per video

    train_dataset = ClevrerSlotDataset(
        data=data["train"],
        split="train",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        frameskip=cfg.frameskip
    )

    val_dataset = ClevrerSlotDataset(
        data=data["val"],
        split="val",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
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

    return MaskedSlotPredictor(
        num_slots=cfg.videosaur.NUM_SLOTS,  # S: number of slots
        slot_dim=cfg.videosaur.SLOT_DIM,  # 64 or higher if action/proprio included
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


def compute_loss(predictor, batch, cfg, device, inference=False):
    """Compute loss for a batch."""
    embed = batch["embed"].to(device)  # (B, T, S, D)

    # Split into history and target
    history = embed[:, :cfg.dinowm.history_size, :, :]  # (B, history_size, S, D)
    target = embed[:, cfg.dinowm.history_size:cfg.dinowm.history_size + cfg.dinowm.num_preds, :, :]  # (B, num_preds, S, D)

    # Forward pass
    if inference:
        pred_output = predictor.inference(history) # only future prediction
        # Loss on future prediction
        loss_future = F.mse_loss(pred_output, target.detach())

        losses = {}
        losses["loss_future"] = loss_future

        loss_masked_history = torch.tensor(0.0, device=device)
        losses["loss_masked_history"] = loss_masked_history

        # Total loss
        total_loss = loss_masked_history + loss_future
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
                pred_history[:, :, mask_indices, :],
                history[:, :, mask_indices, :].detach()
            )
            losses["loss_masked_history"] = loss_masked_history
        else:
            loss_masked_history = torch.tensor(0.0, device=device)
            losses["loss_masked_history"] = loss_masked_history

        # Loss on future prediction
        loss_future = F.mse_loss(pred_future, target.detach())
        losses["loss_future"] = loss_future

        # Total loss
        total_loss = loss_masked_history + loss_future
        losses["loss"] = total_loss

    return losses


@torch.no_grad()
def validate(predictor, val_loader, cfg, device, world_size):
    """Run validation and return average loss."""
    predictor.eval()
    total_loss = 0.0
    total_loss_future = 0.0
    total_loss_masked = 0.0
    num_batches = 0

    for batch in val_loader:
        losses = compute_loss(predictor, batch, cfg, device, inference=True)
        total_loss += losses["loss"].item()
        total_loss_future += losses["loss_future"].item()
        total_loss_masked += losses["loss_masked_history"].item()
        num_batches += 1

    # Average across batches
    avg_loss = total_loss / max(num_batches, 1)
    avg_loss_future = total_loss_future / max(num_batches, 1)
    avg_loss_masked = total_loss_masked / max(num_batches, 1)

    # Reduce across processes if DDP
    if world_size > 1:
        loss_tensor = torch.tensor([avg_loss, avg_loss_future, avg_loss_masked, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor[0].item() / world_size
        avg_loss_future = loss_tensor[1].item() / world_size
        avg_loss_masked = loss_tensor[2].item() / world_size

    return {
        "val/loss": avg_loss,
        "val/loss_future": avg_loss_future,
        "val/loss_masked_history": avg_loss_masked,
    }


@torch.no_grad()
def rollout_video_slots(predictor, pre_slots, cfg, device, batch_size=None):
    """
    Rollout slots from OBS_FRAMES (128) to TARGET_LEN (160) using autoregressive prediction.

    Args:
        predictor: MaskedSlotPredictor model
        pre_slots: Dict of {video_key: slots} where slots is [128, num_slots, slot_dim]
        cfg: Config with history_size, num_preds, frameskip
        device: torch device
        batch_size: Batch size for rollout. If None, uses number of GPUs.

    Returns:
        Dict of {video_key: extended_slots} where extended_slots is [160, num_slots, slot_dim]
    """
    predictor.eval()
    history_len = cfg.dinowm.history_size
    pred_len = cfg.dinowm.num_preds
    frameskip = cfg.frameskip

    # Default batch size: number of available GPUs
    if batch_size is None:
        batch_size = max(1, torch.cuda.device_count())

    all_fn = list(pre_slots.keys())
    all_slots = {}

    # Process videos in batches
    for batch_start in tqdm(range(0, len(all_fn), batch_size), desc="Rolling out slots"):
        batch_end = min(batch_start + batch_size, len(all_fn))
        batch_fns = all_fn[batch_start:batch_end]
        current_bs = len(batch_fns)

        # Load and stack slots for this batch
        batch_slots_list = []
        for fn in batch_fns:
            slots = pre_slots[fn]  # [128, num_slots, slot_dim]
            if isinstance(slots, np.ndarray):
                slots = torch.from_numpy(slots)
            batch_slots_list.append(slots.float())

        # Stack to [B, 128, N, C]
        batch_slots = torch.stack(batch_slots_list, dim=0).to(device)
        num_slots, slot_dim = batch_slots.shape[2], batch_slots.shape[3]

        # Pad to target length: [B, 160, N, C]
        extended_slots = torch.zeros(current_bs, TARGET_LEN, num_slots, slot_dim, device=device)
        extended_slots[:, :OBS_FRAMES, :, :] = batch_slots

        # For models trained with frameskip, we need to handle multiple offset sequences
        frames_to_predict = TARGET_LEN - OBS_FRAMES  # 32 frames to predict

        # For each offset in [0, frameskip), we predict a sequence
        all_pred_slots = []

        for off_idx in range(frameskip):
            # Start position for history (subsampled)
            start = OBS_FRAMES - history_len * frameskip + off_idx

            # Extract history with frameskip: [B, history_len, N, C]
            history_indices = list(range(start, OBS_FRAMES, frameskip))
            history = extended_slots[:, history_indices, :, :]  # [B, history_len, N, C]

            # Calculate how many autoregressive steps needed
            num_pred_frames_needed = (frames_to_predict + frameskip - 1) // frameskip

            pred_frames_list = []
            current_history = history

            while len(pred_frames_list) < num_pred_frames_needed:
                # Predict next pred_len frames: [B, pred_len, N, C]
                pred = predictor.inference(current_history)
                pred_frames_list.append(pred)

                # Update history: shift by pred_len and append predictions
                if current_history.shape[1] > pred_len:
                    current_history = torch.cat([
                        current_history[:, pred_len:, :, :],
                        pred
                    ], dim=1)
                else:
                    current_history = pred[:, -history_len:, :, :]

            # Concatenate all predictions for this offset: [B, >=num_pred_frames_needed, N, C]
            all_preds_for_offset = torch.cat(pred_frames_list, dim=1)
            all_pred_slots.append(all_preds_for_offset[:, :num_pred_frames_needed, :, :])

        # Interleave predictions from different offsets
        # Frame OBS_FRAMES + i should come from offset (i % frameskip), at position (i // frameskip)
        for i in range(frames_to_predict):
            offset = i % frameskip
            pos = i // frameskip
            if pos < all_pred_slots[offset].shape[1]:
                extended_slots[:, OBS_FRAMES + i, :, :] = all_pred_slots[offset][:, pos, :, :]

        # Store results for each video in batch
        for batch_idx, fn in enumerate(batch_fns):
            all_slots[fn] = extended_slots[batch_idx].cpu().numpy()

        # Clear cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_slots


# ============================================================================
# Main Entry Point
# ============================================================================
@hydra.main(version_base=None, config_path="../../configs", config_name="config_train_causal_clevrer_slot")
def run(cfg):
    """Run training of predictor"""

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
    predictor = get_world_model(cfg).to(device)

    # Wrap with DDP if needed
    if is_ddp:
        predictor = torch.nn.parallel.DistributedDataParallel(
            predictor, device_ids=[local_rank], output_device=local_rank
        )

    # Setup optimizer
    model_params = predictor.module.parameters() if is_ddp else predictor.parameters()
    optimizer = torch.optim.AdamW(model_params, lr=cfg.predictor_lr)

    if cfg.rollout.get("rollout_only", False):
        # Load checkpoint for rollout only
        checkpoint_path = cfg.rollout.get("rollout_checkpoint", None)
        if checkpoint_path is None:
            raise ValueError("rollout_checkpoint must be specified for rollout_only mode.")

        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        predictor.module.load_state_dict(state_dict) if is_ddp else predictor.load_state_dict(state_dict)

        if is_main_process(rank):
            logging.info(f"Loaded checkpoint from {checkpoint_path} for rollout only.")

        # Rollout slots
        logging.info("Starting slot rollout (128 -> 160 frames)...")

        # Use the model without DDP wrapper for inference
        predictor_for_rollout = predictor.module if is_ddp else predictor
        predictor_for_rollout.eval()

        rollout_data = {}

        for split in ["train", "val", "test"]:
            if split not in data:
                logging.warning(f"Split '{split}' not found in data, skipping...")
                continue

            logging.info(f"Processing {split} split...")
            rollout_data[split] = rollout_video_slots(
                predictor_for_rollout, data[split], cfg, device, batch_size=cfg.rollout.get("rollout_batch_size", None)
            )
            logging.info(f"Finished {split}: {len(rollout_data[split])} videos")

        # Save rollout data
        embedding_path = Path(cfg.embedding_dir)
        rollout_filename = f"rollout_{str(embedding_path.name)[:-4]}_lr{cfg.predictor_lr}_mask{cfg.num_masked_slots}.pkl"
        rollout_path = embedding_path.parent / rollout_filename

        with open(rollout_path, "wb") as f:
            pkl.dump(rollout_data, f)

        logging.info(f"Saved rollout slots to {rollout_path}")

        if wandb_logger is not None:
            wandb_logger.log({"rollout/path": str(rollout_path)})

        # Cleanup and exit
        if wandb_logger is not None:
            wandb.finish()
        cleanup_distributed()
        return
    # Training loop
    log_every_n_epochs = cfg.get("log_every_n_epochs", 1)
    global_step = 0

    for epoch in range(cfg.trainer.max_epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        predictor.train()
        epoch_loss = 0.0
        epoch_loss_future = 0.0
        epoch_loss_masked = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process(rank))
        for batch in pbar:
            optimizer.zero_grad()

            # Compute loss
            losses = compute_loss(
                predictor.module if is_ddp else predictor,
                batch, cfg, device
            )

            # Backward pass
            losses["loss"].backward()
            optimizer.step()

            # Accumulate metrics
            epoch_loss += losses["loss"].item()
            epoch_loss_future += losses["loss_future"].item()
            epoch_loss_masked += losses["loss_masked_history"].item()
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
                    "train/step": global_step,
                    "train/epoch": epoch,
                })

        # Epoch-level metrics
        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_train_loss_future = epoch_loss_future / max(num_batches, 1)
        avg_train_loss_masked = epoch_loss_masked / max(num_batches, 1)

        if is_main_process(rank):
            logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Future: {avg_train_loss_future:.4f}, Masked: {avg_train_loss_masked:.4f}")

        # Validation
        if (epoch + 1) % log_every_n_epochs == 0:
            val_metrics = validate(
                predictor.module if is_ddp else predictor,
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
                        "epoch": epoch + 1,
                    })

        # Save checkpoint (only on main process)
        if is_main_process(rank):
            checkpoint_path = os.path.join(cache_dir, f"{cfg.output_model_name}_epoch_{epoch+1}_predictor.ckpt")
            state_dict = predictor.module.state_dict() if is_ddp else predictor.state_dict()
            torch.save(state_dict, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    if is_main_process(rank):
        final_path = os.path.join(cache_dir, f"{cfg.output_model_name}_final_predictor.ckpt")
        state_dict = predictor.module.state_dict() if is_ddp else predictor.state_dict()
        torch.save(state_dict, final_path)
        logging.info(f"Saved final model to {final_path}")

    # Rollout slots (only on main process)
    if cfg.rollout.get("save_rollout", False) and is_main_process(rank):
        logging.info("Starting slot rollout (128 -> 160 frames)...")

        # Use the model without DDP wrapper for inference
        predictor_for_rollout = predictor.module if is_ddp else predictor
        predictor_for_rollout.eval()

        rollout_data = {}

        for split in ["train", "val", "test"]:
            if split not in data:
                logging.warning(f"Split '{split}' not found in data, skipping...")
                continue

            logging.info(f"Processing {split} split...")
            rollout_data[split] = rollout_video_slots(
                predictor_for_rollout, data[split], cfg, device, batch_size=cfg.rollout.get("rollout_batch_size", None)
            )
            logging.info(f"Finished {split}: {len(rollout_data[split])} videos")

        # Save rollout data
        embedding_path = Path(cfg.embedding_dir)
        rollout_filename = f"rollout_{str(embedding_path.name)[:-4]}_lr{cfg.predictor_lr}_mask{cfg.num_masked_slots}.pkl"
        rollout_path = embedding_path.parent / rollout_filename

        with open(rollout_path, "wb") as f:
            pkl.dump(rollout_data, f)

        logging.info(f"Saved rollout slots to {rollout_path}")

        if wandb_logger is not None:
            wandb_logger.log({"rollout/path": str(rollout_path)})

    # Cleanup
    if wandb_logger is not None:
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    run()
