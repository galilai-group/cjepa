"""Train ALOE on C-JEPA-rolled CLEVRER slots."""

from __future__ import annotations

import math
from pathlib import Path

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from .data import load_vocab, make_dataset, make_loader
from .model import ALOE


def build_model(cfg) -> ALOE:
    vocab = load_vocab()
    model = OmegaConf.to_container(cfg.model, resolve=True)
    return ALOE(
        **model,
        sample_frames=cfg.data.sample_frames,
        max_question_len=cfg.data.max_question_len,
        max_choice_len=cfg.data.max_choice_len,
        question_vocab_size=len(vocab["q_vocab"]),
        answer_vocab_size=len(vocab["a_vocab"]),
    )


def batch_accuracy(batch, output):
    result = {
        "descriptive": [0, 0],
        "multiple-choice": [0, 0],
        "explanatory": [0, 0],
        "predictive": [0, 0],
        "counterfactual": [0, 0],
    }
    cls_logits = output["cls_answer_logits"]
    if cls_logits is not None:
        labels = batch["cls_label"].long()
        result["descriptive"] = [
            int((cls_logits.argmax(-1) == labels).sum()),
            labels.numel(),
        ]

    mc_logits = output["mc_answer_logits"]
    if mc_logits is not None:
        correct_choices = (mc_logits > 0) == batch["mc_label"].bool()
        flags = batch["mc_flag"].long()
        subtype_names = {1: "explanatory", 2: "predictive", 3: "counterfactual"}
        for index, subtype in enumerate(batch["mc_subtype"].tolist()):
            correct = int(correct_choices[flags == index].all())
            result["multiple-choice"][0] += correct
            result["multiple-choice"][1] += 1
            result[subtype_names[subtype]][0] += correct
            result[subtype_names[subtype]][1] += 1
    return result


class ALOEModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters({"config": cfg})
        self.cfg = OmegaConf.create(cfg)
        self.model = build_model(self.cfg)
        self._validation_counts = {}

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, _batch_index):
        output = self(batch)
        losses = self.model.loss(batch, output)
        self.log("train/loss", losses["loss"], on_step=True, on_epoch=True)
        if output["cls_answer_logits"] is not None:
            self.log("train/cls_loss", losses["cls_loss"], on_step=True, on_epoch=True)
        if output["mc_answer_logits"] is not None:
            self.log("train/mc_loss", losses["mc_loss"], on_step=True, on_epoch=True)
        return losses["loss"]

    def on_validation_epoch_start(self):
        self._validation_counts = {
            name: [0, 0]
            for name in (
                "descriptive",
                "multiple-choice",
                "explanatory",
                "predictive",
                "counterfactual",
            )
        }

    def validation_step(self, batch, _batch_index):
        output = self(batch)
        losses = self.model.loss(batch, output)
        self.log("val/loss", losses["loss"], on_epoch=True, sync_dist=True)
        for name, (correct, total) in batch_accuracy(batch, output).items():
            self._validation_counts[name][0] += correct
            self._validation_counts[name][1] += total

    def on_validation_epoch_end(self):
        overall_correct = overall_total = 0
        for name, (correct, total) in self._validation_counts.items():
            accuracy = correct / total if total else 0.0
            self.log(f"val/{name}_acc", accuracy, sync_dist=True)
            if name in {"descriptive", "multiple-choice"}:
                overall_correct += correct
                overall_total += total
        self.log(
            "val/overall_acc",
            overall_correct / overall_total if overall_total else 0.0,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )
        total = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = int(total * self.cfg.train.warmup_fraction)
        minimum = self.cfg.train.min_learning_rate / self.cfg.train.learning_rate

        def schedule(step):
            if warmup and step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            return minimum + (1 - minimum) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    train_data = make_dataset(cfg, "train")
    val_data = make_dataset(cfg, "val")
    train_loader = make_loader(train_data, cfg, train=True)
    val_loader = make_loader(val_data, cfg, train=False)

    output = Path(cfg.train.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=output,
        filename="best",
        monitor="val/overall_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    module = ALOEModule(OmegaConf.to_container(cfg, resolve=True))
    trainer = pl.Trainer(
        default_root_dir=output,
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        log_every_n_steps=cfg.train.log_every_n_steps,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        num_sanity_val_steps=0,
    )
    last = output / "last.ckpt"
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=last if cfg.train.resume and last.exists() else None,
    )
    print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    run()
