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

from src.third_party.videosaur.videosaur import configuration, data, metrics, models, utils

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
parser.add_argument("--data-dir", help="Path to data directory")
parser.add_argument("--config", help="Configuration to run")
parser.add_argument("--log-dir", help="Path to experiment log directory")
parser.add_argument(
    "--config-file", default="settings.yaml", help="Path to experiment log directory"
)
parser.add_argument("config_overrides", nargs="*", help="Additional arguments")


def _setup_trainer_config(trainer_config: Dict[str, Any]) -> Dict[str, Any]:
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


def _setup_loggers(args, log_path: pathlib.Path) -> Dict[str, pl.loggers.logger.Logger]:
    if args.dry:
        return {}

    # Tensorboard logs go to <log_dir>/<tensorboard_subdir>/
    logger_tensorboard = pl.loggers.TensorBoardLogger(
        save_dir=log_path, name=TENSORBOARD_SUBDIR, version=""
    )
    # CSV logs go to <log_dir>/<metrics_subdir>/version_N/metrics.csv, where N is the number of
    # restarts of the job
    logger_csv = pl.loggers.CSVLogger(save_dir=log_path, name=METRICS_SUBDIR)

    return {"tensorboard": logger_tensorboard, "csv": logger_csv}


def main(args, config_overrides=None):
    if config_overrides is None:
        config_overrides = args.config_overrides
    config_file = os.path.join(args.log_dir, args.config_file)
    config = configuration.load_config(config_file)
    if args.config:
        config = configuration.override_config(config, args.config, config_overrides)
    else:
        config = configuration.override_config(config, additional_overrides=config_overrides)

    if not args.verbose or args.quiet:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        from pytorch_lightning.utilities.warnings import PossibleUserWarning

        warnings.filterwarnings("ignore", category=PossibleUserWarning)
    if args.quiet:
        warnings.filterwarnings("ignore", category=UserWarning)

    # Setup log path
    log_path: Optional[pathlib.Path] = None
    if not args.dry:
        evaluation_dir = config.get("experiment_name", "eval")
        log_path = pathlib.Path(args.log_dir, evaluation_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        log_info(f"Using {log_path} as evaluation output directory")

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

    dataset = data.build(config.dataset, data_dir=args.data_dir)
    if args.verbose:
        log_info(str(dataset))

    train_metrics = None

    if config.val_metrics is not None:
        val_metrics = {name: metrics.build(config) for name, config in config.val_metrics.items()}
    else:
        raise ValueError("Eval metric should be validation metrics")

    model = models.build(config.model, config.optimizer, train_metrics, val_metrics)
    if config.model.load_weights is not None:
        weights_path = config.model.load_weights
    else:
        weights_path = utils.find_last_checkpoint(pathlib.Path(args.log_dir))
    assert weights_path is not None, "No checkpoint found"
    if os.path.exists(weights_path):
        model.load_weights_from_checkpoint(weights_path, module_mapping=config.model.modules_to_load)
    else:
        raise ValueError(f"Checkpoint file {weights_path} doesn't exist.")
    loggers = _setup_loggers(args, log_path)
    trainer_config = _setup_trainer_config(config.setdefault("trainer", {}))

    # Save the final configuration

    log_info(f"Configuration:\n{OmegaConf.to_yaml(config, resolve=True)}")

    # When running DDP, expose log path to other processes through environment variable
    if "strategy" in trainer_config and trainer_config["strategy"].startswith("ddp"):
        os.environ["VIDEOSAUR_LOG_PATH"] = str(log_path)

    trainer = pl.Trainer(
        max_epochs=-1,  # We control training duration using `max_steps`
        check_val_every_n_epoch=None,  # We do not use epochs for training
        default_root_dir=log_path,
        callbacks=[],
        logger=[logger for logger in loggers.values()] if loggers else False,
        enable_progress_bar=(not args.quiet and not args.no_interactive),
        enable_model_summary=not args.quiet,
        enable_checkpointing=False,
        **trainer_config,
    )
    trainer.validate(model=model, datamodule=dataset)


if __name__ == "__main__":
    main(parser.parse_args())
