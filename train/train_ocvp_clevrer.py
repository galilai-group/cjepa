"""
OCVP (Object-Centric Video Prediction) Training Script for CLEVRER.

This script trains an OCVP predictor on top of a frozen StoSAVi backbone.
Based on: "Object-Centric Video Prediction via Decoupling of Object Dynamics and Interactions"

Usage:
    python train/train_ocvp_clevrer.py
    
    # With custom config overrides:
    python train/train_ocvp_clevrer.py ocvp.predictor_type=ocvp_par loss.pred_img_weight=0.0
"""

import os
import sys
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf, DictConfig
from loguru import logger as logging
from tqdm import tqdm
import wandb

# Local imports
from custom_models.ocvp_predictor import build_ocvp_predictor, OCVPWrapper
from custom_models.custom_codes.custom_dataset import ClevrerVideoDataset
from slotformer.base_slots.models import build_model

import stable_pretraining as spt


# ============================================================================
# Distributed Training Setup
# ============================================================================
def setup_distributed() -> Tuple[bool, int, int, int]:
    """Setup distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if current process is the main process."""
    return rank == 0


# ============================================================================
# Data Loading
# ============================================================================
def get_img_pipeline(key: str, target: str, img_size: int = 64):
    """Image transform pipeline for CLEVRER."""
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(
            **spt.data.dataset_stats.ImageNet,
            source=key,
            target=target,
        ),
        spt.data.transforms.Resize((img_size, img_size), source=key, target=target),
    )


def get_data(cfg: DictConfig, is_ddp: bool, world_size: int, rank: int):
    """Setup CLEVRER dataset with video frames."""
    
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
        idx_offset=10000,  # CLEVRER val videos start at 10000
    )
    
    # Apply transforms
    transform = spt.data.transforms.Compose(
        *[get_img_pipeline(f"pixels.{i}", f"pixels.{i}", cfg.image_size) 
          for i in range(cfg.n_steps)],
    )
    train_set.transform = transform
    val_set.transform = transform
    
    # Setup samplers for DDP
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True
    ) if is_ddp else None
    
    val_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False
    ) if is_ddp else None
    
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )
    
    return train_loader, val_loader, train_sampler


# ============================================================================
# Model Building
# ============================================================================
def load_savi_backbone(cfg: DictConfig, device: torch.device) -> nn.Module:
    """Load pretrained StoSAVi backbone."""
    # Load params from config file
    params_path = cfg.savi.params
    if params_path.endswith('.py'):
        params_path = params_path[:-3]
    sys.path.append(os.path.dirname(params_path))
    params = importlib.import_module(os.path.basename(params_path))
    params = params.SlotFormerParams()
    
    # Build model
    model = build_model(params)
    
    # Load weights
    checkpoint = torch.load(cfg.savi.weight, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    # Set to eval mode and freeze
    model.eval()
    model.testing = True  # Only return slots
    for param in model.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    logging.info(f"Loaded StoSAVi from {cfg.savi.weight}")
    logging.info(f"  Slots: {cfg.savi.num_slots}, Dim: {cfg.savi.slot_dim}")
    
    return model


def build_ocvp_model(cfg: DictConfig) -> OCVPWrapper:
    """Build OCVP predictor."""
    predictor = build_ocvp_predictor(
        predictor_type=cfg.ocvp.predictor_type,
        num_slots=cfg.savi.num_slots,
        slot_dim=cfg.savi.slot_dim,
        token_dim=cfg.ocvp.token_dim,
        hidden_dim=cfg.ocvp.hidden_dim,
        num_layers=cfg.ocvp.num_layers,
        n_heads=cfg.ocvp.n_heads,
        residual=cfg.ocvp.residual,
        input_buffer_size=cfg.ocvp.input_buffer_size,
        num_context=cfg.num_context,
        num_preds=cfg.num_preds,
    )
    
    logging.info(f"Built OCVP predictor: {cfg.ocvp.predictor_type}")
    logging.info(f"  Layers: {cfg.ocvp.num_layers}, Heads: {cfg.ocvp.n_heads}")
    logging.info(f"  Token dim: {cfg.ocvp.token_dim}, Hidden dim: {cfg.ocvp.hidden_dim}")
    logging.info(f"  Context: {cfg.num_context}, Preds: {cfg.num_preds}")
    
    return predictor


# ============================================================================
# Training Logic
# ============================================================================
def extract_slots(
    savi: nn.Module, 
    batch: Dict, 
    cfg: DictConfig, 
    device: torch.device
) -> torch.Tensor:
    """
    Extract slot representations from video frames using frozen SAVi.
    
    Args:
        savi: Frozen StoSAVi model
        batch: Batch containing 'pixels.0', 'pixels.1', ... keys
        cfg: Config
        device: Device
        
    Returns:
        slots: Shape (B, T, num_slots, slot_dim)
    """
    video = batch["pixels"].to(device)  # (B, T, C, H, W)
    in_dict = {'img': video}
    B, T, C, H, W = video.shape
    
    # Process through SAVi to get slots
    with torch.no_grad():
        out_dict = savi(in_dict)
    
    return out_dict['post_slots']


def decode_slots(
    savi: nn.Module,
    slots: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode slots back to images using SAVi decoder.
    
    Args:
        savi: StoSAVi model with decoder
        slots: Shape (B, T, num_slots, slot_dim) or (B * T, num_slots, slot_dim)
        
    Returns:
        recon_combined: Shape (B, T, C, H, W) or (B*T, C, H, W)
        recons: Individual object reconstructions
        masks: Object masks
    """
    original_shape = slots.shape
    
    # Flatten to (B*T, num_slots, slot_dim) if needed
    if len(slots.shape) == 4:
        B, T, num_slots, slot_dim = slots.shape
        slots = slots.reshape(B * T, num_slots, slot_dim)
        reshape_output = True
    else:
        reshape_output = False
        B, T = None, None
    
    with torch.no_grad():
        recon_combined, recons, masks, _ = savi.decode(slots)
    
    if reshape_output:
        # Reshape back to (B, T, ...)
        recon_combined = recon_combined.reshape(B, T, *recon_combined.shape[1:])
        recons = recons.reshape(B, T, *recons.shape[1:])
        masks = masks.reshape(B, T, *masks.shape[1:])
    
    return recon_combined, recons, masks


def compute_loss(
    predictor: OCVPWrapper,
    savi: nn.Module,
    batch: Dict,
    cfg: DictConfig,
    device: torch.device,
    training: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute OCVP losses.
    
    Args:
        predictor: OCVP predictor
        savi: Frozen StoSAVi backbone
        batch: Input batch
        cfg: Config
        device: Device
        training: Whether in training mode
        
    Returns:
        Dictionary of losses
    """
    # Extract slots from all frames
    slots = extract_slots(savi, batch, cfg, device)  # (B, T, num_slots, slot_dim)
    
    B, T, num_slots, slot_dim = slots.shape
    
    # Split into context and target
    context_slots = slots[:, :cfg.num_context]  # (B, num_context, num_slots, slot_dim)
    target_slots = slots[:, cfg.num_context:cfg.num_context + cfg.num_preds]  # (B, num_preds, ...)
    
    # Predict future slots (autoregressive)
    pred_slots = predictor(slots)  # (B, num_preds, num_slots, slot_dim)
    
    losses = {}
    
    # Slot prediction loss
    if cfg.loss.pred_slot_weight > 0:
        slot_loss = F.mse_loss(pred_slots, target_slots.detach())
        losses["loss_slot"] = slot_loss * cfg.loss.pred_slot_weight
    else:
        losses["loss_slot"] = torch.tensor(0.0, device=device)
    
    # Image reconstruction loss (optional)
    if cfg.loss.pred_img_weight > 0:
        # Decode predicted slots to images
        pred_imgs, _, _ = decode_slots(savi, pred_slots)  # (B, num_preds, C, H, W)
        target_imgs = batch["pixels"][:, cfg.num_context:cfg.num_context + cfg.num_preds].to(device)  # (B, num_preds, C, H, W)
        
        img_loss = F.mse_loss(pred_imgs, target_imgs.detach())
        losses["loss_img"] = img_loss * cfg.loss.pred_img_weight
    else:
        losses["loss_img"] = torch.tensor(0.0, device=device)
    
    # Total loss
    losses["loss"] = losses["loss_slot"] + losses["loss_img"]
    
    return losses


@torch.no_grad()
def validate(
    predictor: OCVPWrapper,
    savi: nn.Module,
    val_loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    world_size: int,
    rank: int,
) -> Dict[str, float]:
    """Run validation and return average losses."""
    predictor.eval()
    
    total_losses = {"loss": 0.0, "loss_slot": 0.0, "loss_img": 0.0}
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validating", disable=not is_main_process(rank)):
        losses = compute_loss(predictor, savi, batch, cfg, device, training=False)
        
        for key in total_losses:
            total_losses[key] += losses[key].item()
        num_batches += 1
    
    # Average across batches
    avg_losses = {f"val/{k}": v / max(num_batches, 1) for k, v in total_losses.items()}
    
    # Reduce across processes if DDP
    if world_size > 1:
        for key in avg_losses:
            tensor = torch.tensor(avg_losses[key], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            avg_losses[key] = tensor.item()
    
    return avg_losses


def save_checkpoint(
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    cfg: DictConfig,
    is_best: bool = False,
):
    """Save training checkpoint."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get state dict from DDP wrapper if needed
    predictor_state = predictor.module.state_dict() if hasattr(predictor, 'module') else predictor.state_dict()
    
    checkpoint = {
        "epoch": epoch,
        "predictor_state_dict": predictor_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": OmegaConf.to_container(cfg),
    }
    
    # Save latest
    torch.save(checkpoint, output_dir / f"{cfg.output_model_name}_latest.pth")
    
    # Save periodic
    if (epoch + 1) % cfg.save_every_n_epochs == 0:
        torch.save(checkpoint, output_dir / f"{cfg.output_model_name}_epoch_{epoch+1}.pth")
    
    # Save best
    if is_best:
        torch.save(checkpoint, output_dir / f"{cfg.output_model_name}_best.pth")
    
    logging.info(f"Saved checkpoint at epoch {epoch + 1}")


def load_checkpoint(
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: DictConfig,
    device: torch.device,
) -> int:
    """Load checkpoint if resume_from is specified."""
    if not cfg.resume_from:
        return 0
    
    checkpoint = torch.load(cfg.resume_from, map_location=device)
    
    # Handle DDP wrapper
    if hasattr(predictor, 'module'):
        predictor.module.load_state_dict(checkpoint["predictor_state_dict"])
    else:
        predictor.load_state_dict(checkpoint["predictor_state_dict"])
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    start_epoch = checkpoint["epoch"] + 1
    logging.info(f"Resumed from {cfg.resume_from}, starting at epoch {start_epoch}")
    
    return start_epoch


# ============================================================================
# Main Training Loop
# ============================================================================
@hydra.main(version_base=None, config_path="../configs", config_name="config_train_ocvp")
def run(cfg: DictConfig):
    """Main training function."""
    
    # Setup distributed training
    is_ddp, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if is_main_process(rank):
        logging.info("=" * 60)
        logging.info("OCVP Training for CLEVRER")
        logging.info("=" * 60)
        logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Setup wandb
    if cfg.wandb.enable and is_main_process(rank):
        wandb.init(
            name=cfg.wandb.get("name", "ocvp_clevrer"),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    
    # Load data
    train_loader, val_loader, train_sampler = get_data(cfg, is_ddp, world_size, rank)
    if is_main_process(rank):
        logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Load SAVi backbone
    savi = load_savi_backbone(cfg, device)
    
    # Build OCVP predictor
    predictor = build_ocvp_model(cfg).to(device)
    
    if is_ddp:
        predictor = nn.parallel.DistributedDataParallel(
            predictor, device_ids=[local_rank], find_unused_parameters=False
        )
    
    # Count parameters
    num_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    if is_main_process(rank):
        logging.info(f"Trainable parameters: {num_params:,}")
    
    # Setup optimizer
    model_params = predictor.module.parameters() if is_ddp else predictor.parameters()
    optimizer = torch.optim.AdamW(
        model_params,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * cfg.trainer.max_epochs
    warmup_steps = len(train_loader) * cfg.scheduler.warmup_epochs
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=cfg.scheduler.min_lr,
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if cfg.resume_from:
        start_epoch = load_checkpoint(predictor, optimizer, scheduler, cfg, device)
    
    # Training loop
    best_val_loss = float("inf")
    global_step = 0
    
    for epoch in range(start_epoch, cfg.trainer.max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        predictor.train()
        epoch_losses = {"loss": 0.0, "loss_slot": 0.0, "loss_img": 0.0}
        num_batches = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.trainer.max_epochs}",
            disable=not is_main_process(rank),
        )
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # Compute loss
            losses = compute_loss(predictor, savi, batch, cfg, device, training=True)
            
            # Backward
            losses["loss"].backward()
            optimizer.step()
            
            # Warmup
            if global_step < warmup_steps:
                lr_scale = min(1.0, float(global_step + 1) / warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.optimizer.lr * lr_scale
            else:
                scheduler.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['loss'].item():.4f}",
                "slot": f"{losses['loss_slot'].item():.4f}",
                "img": f"{losses['loss_img'].item():.4f}",
            })
            
            # Log to wandb
            if cfg.wandb.enable and is_main_process(rank) and global_step % cfg.trainer.log_every_n_steps == 0:
                wandb.log({
                    "train/loss": losses["loss"].item(),
                    "train/loss_slot": losses["loss_slot"].item(),
                    "train/loss_img": losses["loss_img"].item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                    "train/step": global_step,
                }, step=global_step)
        
        # Average epoch losses
        avg_epoch_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        
        if is_main_process(rank):
            logging.info(
                f"Epoch {epoch + 1} - "
                f"Loss: {avg_epoch_losses['loss']:.4f}, "
                f"Slot: {avg_epoch_losses['loss_slot']:.4f}, "
                f"Img: {avg_epoch_losses['loss_img']:.4f}"
            )
        
        # Validation
        val_losses = validate(predictor, savi, val_loader, cfg, device, world_size, rank)
        
        if is_main_process(rank):
            logging.info(
                f"Validation - "
                f"Loss: {val_losses['val/loss']:.4f}, "
                f"Slot: {val_losses['val/loss_slot']:.4f}, "
                f"Img: {val_losses['val/loss_img']:.4f}"
            )
            
            if cfg.wandb.enable:
                wandb.log(val_losses, step=global_step)
            
            # Save checkpoint
            is_best = val_losses["val/loss"] < best_val_loss
            if is_best:
                best_val_loss = val_losses["val/loss"]
            
            save_checkpoint(predictor, optimizer, scheduler, epoch, cfg, is_best=is_best)
    
    # Final save
    if is_main_process(rank):
        save_checkpoint(predictor, optimizer, scheduler, cfg.trainer.max_epochs - 1, cfg)
        logging.info("Training complete!")
        
        if cfg.wandb.enable:
            wandb.finish()
    
    cleanup_distributed()


if __name__ == "__main__":
    run()
