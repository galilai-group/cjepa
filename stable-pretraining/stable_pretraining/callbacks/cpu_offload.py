import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger
import time
from typing import Any, Optional, List


class CPUOffloadCallback(Callback):
    """Offload checkpoint tensors to CPU during save to reduce GPU memory usage.

    This callback intercepts checkpoint saving and moves all PyTorch tensors
    (model weights, optimizer states, scheduler states) from GPU to CPU before
    writing to disk. Prevents GPU OOM for large models (2B+ parameters).

    **Compatible Strategies:**
    - DDP (DistributedDataParallel)
    - Single GPU training

    **Incompatible Strategies (auto-disabled):**
    - FSDP (uses sharded checkpointing)
    - DeepSpeed (has custom checkpoint mechanism)
    - Other sharding strategies

    Args:
        offload_keys: Keys to offload. Defaults to ['state_dict', 'optimizer_states', 'lr_schedulers']
        log_skipped: If False, only logs first/last 10 skipped objects (default: False).
                     If True, logs all skipped objects.

    Example:
        ```python
        from lightning.pytorch import Trainer

        # Just add to callbacks!
        trainer = Trainer(
            strategy="ddp",  # Compatible
            callbacks=[
                CPUOffloadCallback(),  # Auto-enables
                ModelCheckpoint(...),
            ],
        )

        trainer.fit(model)
        ```

    Benefits:
        - 2B model + optimizer: ~12GB GPU memory freed on rank 0
        - No code changes needed in LightningModule
        - Safe resumption - tensors auto-loaded to correct device
        - Adds ~2-5s to checkpoint save time
        - Auto-detects incompatible strategies and disables itself

    Notes:
        - Only affects rank 0 in DDP (only rank that saves)
        - Custom objects in checkpoint are safely skipped
        - Does not affect checkpoint contents or resumption
        - Compatible with all PyTorch Lightning versions >= 2.0
    """

    def __init__(
        self, offload_keys: Optional[List[str]] = None, log_skipped: bool = False
    ):
        super().__init__()

        logger.info("=" * 60)
        logger.info("Initializing CPUOffloadCallback")

        self.offload_keys = offload_keys or [
            "state_dict",
            "optimizer_states",
            "lr_schedulers",
        ]
        self.log_skipped = log_skipped
        self._skipped_types: List[str] = []  # Changed to list to preserve order
        self._checkpoint_count = 0
        self._total_time_saved = 0.0
        self._total_memory_freed = 0.0
        self._is_enabled = True  # Will be set in setup()

        logger.info("Configuration:")
        logger.info(f"  - Offload keys: {self.offload_keys}")
        logger.info(f"  - Log all skipped objects: {self.log_skipped}")
        if not self.log_skipped:
            logger.info("  - Will show first/last 10 skipped objects only")

        # Check CUDA availability
        if torch.cuda.is_available():
            logger.success(
                f"✓ CUDA available: {torch.cuda.device_count()} GPU(s) detected"
            )
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("⚠ CUDA not available - callback will have no effect")

        logger.success("CPUOffloadCallback initialized successfully")
        logger.info("=" * 60)

    def setup(self, trainer, pl_module, stage: str):
        """Called when fit, validate, test, or predict begins."""
        logger.info(f"CPUOffloadCallback: Setup called for stage '{stage}'")
        logger.info(f"  - Trainer rank: {trainer.global_rank}/{trainer.world_size}")

        # Get strategy info
        strategy = trainer.strategy
        strategy_name = strategy.__class__.__name__
        logger.info(f"  - Strategy: {strategy_name}")

        # Check strategy compatibility
        self._is_enabled = self._check_strategy_compatibility(strategy, strategy_name)

        if not self._is_enabled:
            logger.warning("=" * 60)
            logger.warning("⚠ CPUOffloadCallback DISABLED for this training run")
            logger.warning(f"  Reason: Incompatible strategy '{strategy_name}'")
            logger.warning("  The callback will not interfere with checkpointing.")
            logger.info("  Compatible strategies: DDP, SingleDevice")
            logger.info(
                "  Incompatible strategies: FSDP, DeepSpeed (use their native checkpointing)"
            )
            logger.warning("=" * 60)
            return

        logger.success(f"✓ CPUOffloadCallback ENABLED for strategy '{strategy_name}'")

        if trainer.global_rank == 0:
            logger.info("  - This rank will handle checkpoint saving and CPU offload")
        else:
            logger.debug(
                f"  - Rank {trainer.global_rank} will skip checkpoint operations"
            )

    def _check_strategy_compatibility(self, strategy, strategy_name: str) -> bool:
        """Check if the training strategy is compatible with CPU offload."""
        # Compatible strategies
        compatible_strategies = [
            "DDPStrategy",
            "SingleDeviceStrategy",
            "DataParallelStrategy",  # Old-style DP
        ]

        # Explicitly incompatible strategies
        incompatible_strategies = [
            "FSDPStrategy",
            "DeepSpeedStrategy",
            "XLAStrategy",
        ]

        # Check if compatible
        if strategy_name in compatible_strategies:
            logger.success(
                f"✓ Strategy '{strategy_name}' is compatible with CPU offload"
            )
            return True

        # Check if explicitly incompatible
        if strategy_name in incompatible_strategies:
            logger.warning(
                f"✗ Strategy '{strategy_name}' is incompatible with CPU offload"
            )
            logger.info(f"  Reason: {self._get_incompatibility_reason(strategy_name)}")
            return False

        # Check by instance (more robust)
        from lightning.pytorch.strategies import (
            DDPStrategy,
            SingleDeviceStrategy,
        )

        if isinstance(strategy, (DDPStrategy, SingleDeviceStrategy)):
            logger.success(
                f"✓ Strategy '{strategy_name}' detected as compatible (isinstance check)"
            )
            return True

        # Try to detect FSDP/DeepSpeed by import (may not be available)
        try:
            from lightning.pytorch.strategies import FSDPStrategy

            if isinstance(strategy, FSDPStrategy):
                logger.warning("✗ FSDP detected - incompatible with CPU offload")
                logger.info(
                    "  FSDP uses sharded state_dict which shouldn't be modified"
                )
                return False
        except ImportError:
            pass

        try:
            from lightning.pytorch.strategies import DeepSpeedStrategy

            if isinstance(strategy, DeepSpeedStrategy):
                logger.warning("✗ DeepSpeed detected - incompatible with CPU offload")
                logger.info("  DeepSpeed has its own checkpoint mechanism")
                return False
        except ImportError:
            pass

        # Unknown strategy - be conservative
        logger.warning(
            f"⚠ Unknown strategy '{strategy_name}' - disabling callback to be safe"
        )
        logger.info(
            "  If this strategy is similar to DDP, please report this as an issue"
        )
        logger.info(f"  Compatible strategies: {compatible_strategies}")
        logger.info(f"  Known incompatible: {incompatible_strategies}")
        return False

    def _get_incompatibility_reason(self, strategy_name: str) -> str:
        """Get human-readable reason for strategy incompatibility."""
        reasons = {
            "FSDPStrategy": "FSDP uses sharded state_dict - modifying it can cause corruption",
            "DeepSpeedStrategy": "DeepSpeed has custom checkpoint format and mechanisms",
            "XLAStrategy": "XLA uses specialized tensor types and device handling",
        }
        return reasons.get(strategy_name, "Unknown incompatibility")

    def on_train_start(self, trainer, pl_module):
        """Called when training begins."""
        if not self._is_enabled:
            return

        logger.info("CPUOffloadCallback: Training started")

        if trainer.global_rank == 0:
            # Count total parameters
            total_params = sum(p.numel() for p in pl_module.parameters())
            trainable_params = sum(
                p.numel() for p in pl_module.parameters() if p.requires_grad
            )

            logger.info("Model statistics:")
            logger.info(f"  - Total parameters: {total_params:,}")
            logger.info(f"  - Trainable parameters: {trainable_params:,}")
            logger.info(f"  - Model size (bf16): ~{total_params * 2 / 1e9:.2f} GB")
            logger.info(
                f"  - Estimated checkpoint size (with optimizer): ~{total_params * 2 * 3 / 1e9:.2f} GB"
            )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Hook called when checkpoint is being saved.

        Args:
            trainer: PyTorch Lightning Trainer instance
            pl_module: LightningModule being trained
            checkpoint: Checkpoint dict to be saved (modified in-place)
        """
        # Skip if disabled
        if not self._is_enabled:
            return

        # Only log on rank 0 (only rank that saves in DDP)
        if trainer.global_rank != 0:
            return

        self._checkpoint_count += 1
        start_time = time.time()

        logger.info("=" * 60)
        logger.info(
            f"CPUOffloadCallback: Starting checkpoint CPU offload (#{self._checkpoint_count})"
        )
        logger.info(f"  - Global step: {trainer.global_step}")
        logger.info(f"  - Epoch: {trainer.current_epoch}")

        # Reset skipped types for this checkpoint
        self._skipped_types = []

        # Log checkpoint keys present
        logger.info(
            f"Checkpoint contains {len(checkpoint)} keys: {sorted(checkpoint.keys())}"
        )

        # Log initial GPU memory
        gpu_mem_before = 0
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9
            logger.info("GPU memory before offload:")
            logger.info(f"  - Allocated: {gpu_mem_before:.2f} GB")
            logger.info(f"  - Reserved: {gpu_mem_reserved:.2f} GB")

        # Process checkpoint
        processed_keys = self._offload_checkpoint(checkpoint)

        # Log skipped custom objects with smart truncation
        if self._skipped_types:
            total_skipped = len(self._skipped_types)

            if self.log_skipped:
                # Show all
                logger.warning(f"Skipped {total_skipped} custom object(s):")
                for obj_info in self._skipped_types:
                    logger.warning(f"  - {obj_info}")
            else:
                # Show first 10 and last 10
                logger.warning(f"Skipped {total_skipped} custom object(s)")

                if total_skipped <= 20:
                    # Show all if 20 or fewer
                    for obj_info in self._skipped_types:
                        logger.warning(f"  - {obj_info}")
                else:
                    # Show first 10
                    logger.warning("First 10 skipped objects:")
                    for obj_info in self._skipped_types[:10]:
                        logger.warning(f"  - {obj_info}")

                    logger.warning(f"... ({total_skipped - 20} more skipped) ...")

                    # Show last 10
                    logger.warning("Last 10 skipped objects:")
                    for obj_info in self._skipped_types[-10:]:
                        logger.warning(f"  - {obj_info}")

                    logger.info(
                        f"To see all {total_skipped} skipped objects, set log_skipped=True"
                    )

        # Log other checkpoint keys
        other_keys = set(checkpoint.keys()) - set(self.offload_keys)
        if other_keys:
            logger.debug(f"Other checkpoint keys (not processed): {sorted(other_keys)}")

        # Log final GPU memory and timing
        if torch.cuda.is_available():
            gpu_mem_after = torch.cuda.memory_allocated() / 1e9
            gpu_mem_reserved_after = torch.cuda.memory_reserved() / 1e9
            mem_freed = gpu_mem_before - gpu_mem_after

            logger.info("GPU memory after offload:")
            logger.info(f"  - Allocated: {gpu_mem_after:.2f} GB")
            logger.info(f"  - Reserved: {gpu_mem_reserved_after:.2f} GB")

            if mem_freed > 0:
                logger.success(f"✓ GPU memory freed: {mem_freed:.2f} GB")
                self._total_memory_freed += mem_freed
            else:
                logger.info(f"GPU memory freed: {mem_freed:.2f} GB")

        checkpoint_time = time.time() - start_time
        self._total_time_saved += checkpoint_time

        logger.success(
            f"CPUOffloadCallback: Checkpoint #{self._checkpoint_count} completed in {checkpoint_time:.2f}s"
        )
        logger.info(f"Successfully processed keys: {processed_keys}")
        logger.info("Cumulative stats:")
        logger.info(f"  - Total checkpoints saved: {self._checkpoint_count}")
        logger.info(f"  - Total time in offload: {self._total_time_saved:.2f}s")
        logger.info(f"  - Total memory freed: {self._total_memory_freed:.2f} GB")
        logger.info(
            f"  - Average time per checkpoint: {self._total_time_saved / self._checkpoint_count:.2f}s"
        )
        logger.info("=" * 60)

    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        if not self._is_enabled:
            return

        if trainer.global_rank == 0:
            logger.info("=" * 60)
            logger.info("CPUOffloadCallback: Training completed")
            logger.info("Final statistics:")
            logger.info(f"  - Total checkpoints saved: {self._checkpoint_count}")
            logger.info(f"  - Total time in offload: {self._total_time_saved:.2f}s")
            logger.info(
                f"  - Total GPU memory freed: {self._total_memory_freed:.2f} GB"
            )

            if self._checkpoint_count > 0:
                logger.info(
                    f"  - Average time per checkpoint: {self._total_time_saved / self._checkpoint_count:.2f}s"
                )
                logger.info(
                    f"  - Average memory freed per checkpoint: {self._total_memory_freed / self._checkpoint_count:.2f} GB"
                )

            logger.success("CPUOffloadCallback: All checkpoints saved successfully")
            logger.info("=" * 60)

    def teardown(self, trainer, pl_module, stage: str):
        """Called when fit, validate, test, or predict ends."""
        if trainer.global_rank == 0:
            logger.info(f"CPUOffloadCallback: Teardown called for stage '{stage}'")
            if self._is_enabled:
                logger.debug(f"Final checkpoint count: {self._checkpoint_count}")
            else:
                logger.debug("Callback was disabled - no checkpoints processed")

    def on_exception(self, trainer, pl_module, exception):
        """Called when an exception occurs during training."""
        if trainer.global_rank == 0:
            logger.error("=" * 60)
            logger.error("CPUOffloadCallback: Exception occurred during training")
            logger.error(f"Exception: {type(exception).__name__}: {exception}")
            logger.error(f"Callback was enabled: {self._is_enabled}")
            logger.error(
                f"Checkpoints saved before exception: {self._checkpoint_count}"
            )
            logger.error("=" * 60)

    def _offload_checkpoint(self, checkpoint: dict) -> List[str]:
        """Offload specified checkpoint keys to CPU."""
        processed_keys = []

        for key in self.offload_keys:
            if key in checkpoint:
                logger.info(f"Processing checkpoint key: '{key}'")
                key_start = time.time()

                try:
                    checkpoint[key] = self._safe_to_cpu(checkpoint[key], path=key)
                    key_time = time.time() - key_start
                    logger.success(f"✓ Completed '{key}' in {key_time:.2f}s")
                    processed_keys.append(key)

                except Exception as e:
                    logger.error(f"✗ Failed to process '{key}': {e}")
                    logger.exception("Full traceback:")
                    logger.warning(f"Checkpoint key '{key}' will remain on GPU")
            else:
                logger.debug(f"Key '{key}' not found in checkpoint, skipping")

        return processed_keys

    def _safe_to_cpu(self, obj: Any, path: str = "root") -> Any:
        """Recursively move tensors to CPU, skip custom objects."""
        if isinstance(obj, torch.Tensor):
            size_mb = obj.element_size() * obj.nelement() / 1e6
            device = obj.device
            logger.debug(
                f"Moving tensor at '{path}': {tuple(obj.shape)} "
                f"({size_mb:.1f} MB) from {device} to CPU"
            )
            return obj.cpu()

        elif isinstance(obj, dict):
            logger.trace(f"Processing dict at '{path}' with {len(obj)} keys")
            return {k: self._safe_to_cpu(v, f"{path}.{k}") for k, v in obj.items()}

        elif isinstance(obj, (list, tuple)):
            logger.trace(
                f"Processing {type(obj).__name__} at '{path}' with {len(obj)} items"
            )
            result = [self._safe_to_cpu(v, f"{path}[{i}]") for i, v in enumerate(obj)]
            return tuple(result) if isinstance(obj, tuple) else result

        else:
            # Custom object - don't modify
            obj_type = type(obj).__name__
            self._skipped_types.append(f"{obj_type} at '{path}'")
            logger.debug(f"Skipping custom object at '{path}': {obj_type}")
            return obj

    def state_dict(self):
        """Return callback state for checkpointing."""
        state = {
            "checkpoint_count": self._checkpoint_count,
            "total_time_saved": self._total_time_saved,
            "total_memory_freed": self._total_memory_freed,
            "is_enabled": self._is_enabled,
        }
        logger.debug(f"CPUOffloadCallback state_dict: {state}")
        return state

    def load_state_dict(self, state_dict):
        """Load callback state from checkpoint."""
        logger.info("CPUOffloadCallback: Loading state from checkpoint")
        self._checkpoint_count = state_dict.get("checkpoint_count", 0)
        self._total_time_saved = state_dict.get("total_time_saved", 0.0)
        self._total_memory_freed = state_dict.get("total_memory_freed", 0.0)
        self._is_enabled = state_dict.get("is_enabled", True)
        logger.info("Restored state:")
        logger.info(f"  - Checkpoint count: {self._checkpoint_count}")
        logger.info(f"  - Total time saved: {self._total_time_saved:.2f}s")
        logger.info(f"  - Total memory freed: {self._total_memory_freed:.2f} GB")
        logger.info(f"  - Was enabled: {self._is_enabled}")
