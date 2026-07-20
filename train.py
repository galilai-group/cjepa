"""Train C-JEPA from raw pixels or pre-extracted slots in an HDF5 file."""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2

from cjepa import CJEPA
from encoder import build_object_encoder


class ZScore:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std.clamp_min(1e-6)

    def __call__(self, values):
        return ((values - self.mean) / self.std).float()


def column_normalizer(dataset, column):
    values = torch.from_numpy(np.asarray(dataset.get_col_data(column))).float()
    finite = torch.isfinite(values).all(dim=-1)
    values = values[finite]
    return spt.data.transforms.WrapTorchTransform(
        ZScore(values.mean(0), values.std(0)),
        source=column,
        target=column,
    )


def image_transform(size):
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source="pixels", target="pixels"),
        spt.data.transforms.WrapTorchTransform(
            v2.Resize((size, size)), source="pixels", target="pixels"
        ),
    )


def open_dataset(name, cfg, transform=None):
    return swm.data.load_dataset(
        name,
        cache_dir=cfg.data.cache_dir,
        num_steps=cfg.model.history_frames + cfg.model.pred_frames,
        frameskip=cfg.data.frameskip,
        transform=transform,
    )


def make_data(cfg):
    train_set = open_dataset(cfg.data.train, cfg)
    columns = set(train_set.column_names)
    if "slots" not in columns and "pixels" not in columns:
        raise ValueError("Dataset must contain a 'slots' or 'pixels' column")

    transforms = []
    if "slots" not in columns:
        transforms.append(image_transform(cfg.data.image_size))
    for column in ("action", "proprio"):
        if column in columns:
            transforms.append(column_normalizer(train_set, column))
    transform = spt.data.transforms.Compose(*transforms)
    train_set.transform = transform
    val_set = open_dataset(cfg.data.val, cfg, transform=transform)

    generator = torch.Generator().manual_seed(cfg.seed)
    loader = OmegaConf.to_container(cfg.loader, resolve=True)
    if loader["num_workers"] == 0:
        loader.pop("persistent_workers", None)
        loader.pop("prefetch_factor", None)
    train = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        **loader,
    )
    return spt.data.DataModule(train=train, val=val), columns


def make_model(cfg, columns):
    object_encoder = None
    if "slots" not in columns:
        encoder_cfg = cfg.data.encoder
        object_encoder = build_object_encoder(
            dataset=encoder_cfg.dataset,
            config=to_absolute_path(encoder_cfg.config),
            checkpoint=encoder_cfg.checkpoint,
            repo_id=encoder_cfg.repo_id,
        )

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    return CJEPA(object_encoder=object_encoder, **model_cfg)


def cjepa_forward(self, batch, stage):
    batch["action"] = torch.nan_to_num(batch["action"]) if "action" in batch else None
    if batch.get("action") is None:
        batch.pop("action", None)
    batch["proprio"] = (
        torch.nan_to_num(batch["proprio"]) if "proprio" in batch else None
    )
    if batch.get("proprio") is None:
        batch.pop("proprio", None)

    embedding = self.model.encode(batch)
    history_frames = self.model.history_frames
    history = embedding[:, :history_frames]
    target = embedding[:, history_frames:]

    if self.training:
        prediction, masked = self.model.predict(history)
        pred_history = prediction[:, :history_frames]
        pred_future = prediction[:, history_frames:]
        masked_loss = (
            F.mse_loss(
                pred_history[:, :, masked],
                history[:, :, masked].detach(),
            )
            if len(masked)
            else embedding.new_zeros(())
        )
    else:
        pred_future = self.model.predict(history, inference=True)
        masked_loss = embedding.new_zeros(())

    # An action node conditions prediction; it is not itself a prediction target.
    target_end = self.model.action_node or embedding.shape[2]
    future_loss = F.mse_loss(
        pred_future[:, :, :target_end],
        target[:, :, :target_end].detach(),
    )
    loss = masked_loss + future_loss
    prefix = "train" if self.training else "val"
    self.log_dict(
        {
            f"{prefix}/loss": loss,
            f"{prefix}/masked_loss": masked_loss,
            f"{prefix}/future_loss": future_loss,
        },
        on_step=self.training,
        on_epoch=True,
        sync_dist=True,
    )
    return {
        "loss": loss,
        "masked_loss": masked_loss.detach(),
        "future_loss": future_loss.detach(),
    }


class SaveModel(Callback):
    def __init__(self, run_dir: Path, filename: str, every: int = 1):
        self.run_dir = run_dir
        self.filename = filename
        self.every = every

    def on_train_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero or (
            epoch % self.every and epoch != trainer.max_epochs
        ):
            return
        torch.save(
            module.model.state_dict(),
            self.run_dir / f"{self.filename}_epoch_{epoch}_weights.pt",
        )
        torch.save(
            module.model,
            self.run_dir / f"{self.filename}_object.ckpt",
        )


@hydra.main(version_base=None, config_path="config", config_name="train")
def run(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    data, columns = make_data(cfg)
    model = make_model(cfg, columns)
    module = spt.Module(
        model=model,
        forward=cjepa_forward,
        optim=OmegaConf.to_container(cfg.optimizer, resolve=True),
        hparams=OmegaConf.to_container(cfg, resolve=True),
    )

    run_dir = Path(
        swm.data.utils.get_cache_dir(cfg.data.cache_dir, sub_folder="checkpoints"),
        cfg.run_name,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        cfg,
        run_dir / f"{cfg.output_model_name}_config.yaml",
        resolve=True,
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**OmegaConf.to_container(cfg.wandb.config))
    callbacks = [SaveModel(run_dir, cfg.output_model_name, cfg.save_every_n_epochs)]
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
    )
    lightning_ckpt = run_dir / f"{cfg.output_model_name}_lightning.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        ckpt_path=lightning_ckpt if lightning_ckpt.exists() else None,
    )
    manager()


if __name__ == "__main__":
    run()
