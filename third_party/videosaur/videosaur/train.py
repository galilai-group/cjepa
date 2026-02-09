import argparse
import logging
import os
import pathlib
import random
import warnings
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_info as log_info

from third_party.videosaur.videosaur import configuration, data, metrics, models, utils

RESULT_FINISHED = 0
RESULT_TIMEOUT = 1

CHECKPOINT_SUBDIR = "checkpoints"
TENSORBOARD_SUBDIR = "tb"
METRICS_SUBDIR = "metrics"

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true", help="Be verbose")
group.add_argument("-q", "--quiet", action="store_true", help="Suppress outputs")
parser.add_argument("-n", "--dry", action="store_true", help="Dry run (no logfiles)")
parser.add_argument(
    "--no-interactive", action="store_true", help="If running in non-interactive environment"
)
parser.add_argument("--no-tensorboard", action="store_true", help="Do not write tensorboard logs")
parser.add_argument(
    "--check-validation", action="store_true", help="Run correctness checks on data used during eval"
)
parser.add_argument(
    "--run-eval-after-training", action="store_true", help="Evaluate after training has stopped"
)
parser.add_argument(
    "--use-optimizations", action="store_true", help="Enable Pytorch performance optimizations"
)
parser.add_argument("--timeout", help="Stop training after this time (format: DD:HH:MM:SS)")
parser.add_argument("--data-dir", help="Path to data directory")
parser.add_argument("--log-dir", default="./logs", help="Path to log directory")
parser.add_argument(
    "--no-sub-logdirs", action="store_true", help="Directly use log dir to store logs"
)
parser.add_argument(
    "--continue",
    dest="continue_from",
    type=pathlib.Path,
    help="Continue training from this log folder or checkpoint path",
)
parser.add_argument("--config_overrides_file", help="Configuration to override")
parser.add_argument("config", help="Configuration to run")
parser.add_argument("config_overrides", nargs="*", help="Additional arguments")


def _setup_callbacks(args, config, log_path: pathlib.Path, dataset=None) -> Dict[str, pl.Callback]:
    callbacks = {}

    if not args.dry:
        # Explicitly construct model checkpoint to have control over checkpoints directory
        checkpointer = pl.callbacks.ModelCheckpoint(
            log_path / CHECKPOINT_SUBDIR,
            filename="{step}",
            every_n_train_steps=config.checkpoint_every_n_steps,
            verbose=args.verbose,
        )
        callbacks["checkpointer"] = checkpointer

        # Monitor learning rates
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        callbacks["lr_monitor"] = lr_monitor

    if args.check_validation:
        assert dataset is not None
        val_size = dataset.val_size if hasattr(dataset, "val_size") else None
        callbacks["check_validation"] = utils.CheckValidationCallback(val_size)

    if args.timeout:
        timer = pl.callbacks.Timer(duration=args.timeout, interval="step", verbose=args.verbose)
        callbacks["timer"] = timer

    return callbacks


def _setup_loggers(
    args, log_path: pathlib.Path, config
) -> Dict[str, pl.loggers.logger.Logger]:
    if args.dry:
        return {}

    loggers = {}
    if not args.no_tensorboard:
        # Tensorboard logs go to <log_dir>/<tensorboard_subdir>/
        loggers["tensorboard"] = pl.loggers.TensorBoardLogger(
            save_dir=log_path, name=TENSORBOARD_SUBDIR, version=""
        )

    # CSV logs go to <log_dir>/<metrics_subdir>/version_N/metrics.csv, where N is the number of
    # restarts of the job
    loggers["csv"] = pl.loggers.CSVLogger(save_dir=log_path, name=METRICS_SUBDIR)

    wandb_config = config.get("wandb")
    wandb_enabled = wandb_config is not None and wandb_config.get("enable", False)
    if wandb_enabled:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError:
            log_info("wandb not installed, skipping WandbLogger.")
        else:
            if wandb_config.get("mode"):
                os.environ.setdefault("WANDB_MODE", wandb_config.get("mode"))
            wandb_logger = WandbLogger(
                project=wandb_config.get("project", config.experiment_group or "videosaur"),
                entity=wandb_config.get("entity", None),
                name=wandb_config.get("name", config.experiment_name),
                group=wandb_config.get("group", config.experiment_group),
                tags=list(wandb_config.get("tags", [])) or None,
                save_dir=str(log_path),
                config=OmegaConf.to_container(config, resolve=True),
                mode=wandb_config.get("mode", None),
            )
            loggers["wandb"] = wandb_logger

    return loggers


def _setup_trainer_config(
    trainer_config: Dict[str, Any], *, force_log_every_n_steps: Optional[int] = None
) -> Dict[str, Any]:
    # Configure number of training steps
    if "max_steps" not in trainer_config:
        trainer_config["max_steps"] = 100000
        log_info(f"Setting number of training steps to {trainer_config['max_steps']}")

    if "max_epochs" in trainer_config:
        del trainer_config["max_epochs"]
        log_info("Removing `max_epochs` from config because we do not use it")

    # Configure validation frequency
    if "val_check_interval" not in trainer_config:
        trainer_config["val_check_interval"] = 5000
        log_info(f"Setting `val_check_interval` to {trainer_config['val_check_interval']}")

    if "check_val_every_n_epoch" in trainer_config:
        del trainer_config["check_val_every_n_epoch"]
        log_info("Removing `check_val_every_n_epoch` from config because we do not use it")

    # Configure logging frequency
    if force_log_every_n_steps is not None:
        trainer_config["log_every_n_steps"] = force_log_every_n_steps
    elif "log_every_n_steps" not in trainer_config:
        trainer_config["log_every_n_steps"] = 100
        log_info(f"Setting logging frequency to every {trainer_config['log_every_n_steps']} steps")

    # Let Pytorch Lightning select the device if not specified otherwise.
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "auto"

    # Automatically select DDP as strategy if possible and not specified otherwise.
    if (
        "strategy" not in trainer_config
        and trainer_config.get("accelerator") != "cpu"
        and trainer_config.get("devices") != 1
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        strategy = "ddp_find_unused_parameters_false"
        if "find_unused_parameters" in trainer_config:
            if trainer_config["find_unused_parameters"]:
                strategy = "ddp"
            del trainer_config["find_unused_parameters"]
        log_info(f"Setting distributed strategy to {strategy}.")
        trainer_config["strategy"] = strategy

    return trainer_config


def main(args, config_overrides=None):
    rank_zero = utils.get_rank() == 0
    if config_overrides is None:
        config_overrides = args.config_overrides
    config = configuration.load_config(args.config, config_overrides)
    if args.config_overrides_file is not None:
        config = configuration.override_config(
            config,
            override_config_path=args.config_overrides_file,
            additional_overrides=config_overrides,
        )

    if not args.verbose or args.quiet:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        from pytorch_lightning.utilities.warnings import PossibleUserWarning

        warnings.filterwarnings("ignore", category=PossibleUserWarning)
    if args.quiet:
        warnings.filterwarnings("ignore", category=UserWarning)

    # Setup log path
    log_path: Optional[pathlib.Path] = None
    if not args.dry:
        if "VDINOSAUR_LOG_PATH" in os.environ:
            # Retrieve log path from the main process in the DDP setting
            log_path = pathlib.Path(os.environ["VDINOSAUR_LOG_PATH"])
        elif args.continue_from and args.continue_from.is_dir():
            # If `continue` points to a directory, use it as the output logging directory
            log_path = args.continue_from
        else:
            if args.no_sub_logdirs:
                log_path = pathlib.Path(args.log_dir)
                log_path.mkdir(parents=True, exist_ok=True)
            else:
                log_path = utils.make_log_dir(
                    args.log_dir, config.experiment_name, config.experiment_group
                )
        log_info(f"Using {log_path} as output directory")

    # Find checkpoint for automatic resuming of training
    ckpt_path: Optional[str] = None
    if args.continue_from:
        if args.continue_from.is_file() and args.continue_from.suffix == ".ckpt":
            ckpt_path = args.continue_from
        elif args.continue_from.is_dir():
            ckpt_path = utils.find_last_checkpoint(args.continue_from)
        else:
            raise ValueError("Unclear --continue argument: should be .ckpt file or directory")
    elif not args.dry:
        ckpt_path = utils.find_last_checkpoint(log_path)
        if ckpt_path is not None:
            log_info(f"Auto-resuming training from checkpoint {ckpt_path}")

    # Setup random seed
    if "PL_GLOBAL_SEED" in os.environ:
        # Retrieve random seed from the main process in the DDP setting
        seed = int(os.environ["PL_GLOBAL_SEED"])
    elif config.seed is not None:
        seed = config.seed
    else:
        seed = random.randint(0, 2**32 - 1)
    log_info(f"Using random seed {seed}.")
    config.seed = pl.seed_everything(seed, workers=True)

    if args.use_optimizations:
        torch.backends.cudnn.benchmark = True
        # Allow use of TensorFloat-32 on Ampere devices
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset = data.build(config.dataset, data_dir=args.data_dir)
    if args.verbose:
        log_info(str(dataset))

    if config.train_metrics is not None:
        train_metrics = {
            name: metrics.build(config) for name, config in config.train_metrics.items()
        }
    else:
        train_metrics = None

    if config.val_metrics is not None:
        val_metrics = {name: metrics.build(config) for name, config in config.val_metrics.items()}
    else:
        val_metrics = None

    model = models.build(config.model, config.optimizer, train_metrics, val_metrics)

    callbacks = _setup_callbacks(args, config, log_path, dataset)
    loggers = _setup_loggers(args, log_path, config)

    wandb_config = config.get("wandb")
    wandb_enabled = wandb_config is not None and wandb_config.get("enable", False)
    trainer_config = _setup_trainer_config(
        config.setdefault("trainer", {}), force_log_every_n_steps=1 if wandb_enabled else None
    )

    # Save the final configuration
    if rank_zero and log_path and not (log_path / "settings.yaml").exists():
        configuration.save_config(log_path / "settings.yaml", config)

    if "tensorboard" in loggers:
        loggers["tensorboard"].log_hyperparams(config)

    log_info(f"Configuration:\n{OmegaConf.to_yaml(config, resolve=True)}")

    # When running DDP, expose log path to other processes through environment variable
    if "strategy" in trainer_config and trainer_config["strategy"].startswith("ddp"):
        os.environ["VIDEOSAUR_LOG_PATH"] = str(log_path)

    trainer = pl.Trainer(
        max_epochs=-1,  # We control training duration using `max_steps`
        check_val_every_n_epoch=None,  # We do not use epochs for training
        default_root_dir=log_path,
        callbacks=[callback for callback in callbacks.values()],
        logger=[logger for logger in loggers.values()] if loggers else False,
        enable_progress_bar=(not args.quiet and not args.no_interactive),
        enable_model_summary=not args.quiet,
        enable_checkpointing="checkpointer" in callbacks,
        **trainer_config,
    )

    if ckpt_path is not None:
        log_info(f"Resuming training from checkpoint {ckpt_path}")
    else:
        log_info("Starting training from scratch")

    trainer.fit(model=model, datamodule=dataset, ckpt_path=ckpt_path)

    if "checkpointer" in callbacks:
        # Explicitly save a final checkpoint after training. This is needed because the end of
        # training might not align with the checkpointing frequency. Unfortunately, we need to rely
        # on private methods of ModelCheckpoint here, because we need to bypass the frequency checks
        # and ModelCheckpoint does not expose any method for explictly saving checkpoints.
        monitor_candidates = callbacks["checkpointer"]._monitor_candidates(trainer)
        callbacks["checkpointer"]._save_topk_checkpoint(trainer, monitor_candidates)

    if args.run_eval_after_training:
        # Run one more evaluation. This is useful because some more training might have happened
        # after the last epoch, but before training was stopped.
        trainer.validate(model=model, datamodule=dataset)

    if callbacks.get("timer") and callbacks["timer"].time_remaining() <= 0:
        return RESULT_TIMEOUT  # Signal that training was interrupted because of timeout

    return RESULT_FINISHED


if __name__ == "__main__":
    main(parser.parse_args())
