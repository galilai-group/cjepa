"""Inference-only object encoder adapter.

VideoSAUR itself is installed under ``.venv/src`` by ``setup.sh``. Its
training code and dependencies are intentionally not vendored in this repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch import nn

CHECKPOINTS = {
    "clevrer": "clevrer_videosaur_model.ckpt",
    "pusht": "pusht_videosaur_model.ckpt",
}


def checkpoint_dir() -> Path:
    root = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    path = root / "artifacts" / "object-encoders"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_checkpoint(
    dataset: str,
    checkpoint: str | Path | None = None,
    repo_id: str = "HazelNam/CJEPA",
) -> Path:
    """Return an encoder checkpoint, downloading it on first use."""
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if path.exists():
            return path
        filename = path.name
        local_dir = path.parent
    else:
        try:
            filename = CHECKPOINTS[dataset]
        except KeyError as error:
            raise ValueError(
                f"No default object encoder is registered for {dataset!r}"
            ) from error
        local_dir = checkpoint_dir()
        path = local_dir / filename

    if path.exists():
        return path
    print(f"Object encoder not found; downloading {repo_id}/{filename} ...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )
    return Path(downloaded)


class VideoSAUREncoder(nn.Module):
    """Frozen VideoSAUR encoder that maps videos to object slots."""

    def __init__(self, config: str | Path, checkpoint: str | Path):
        super().__init__()
        try:
            # VideoSAUR registers these names unconditionally at import time;
            # stable-pretraining already owns some of them in this process.
            for resolver in (
                "eval",
                "add",
                "sub",
                "mul",
                "div",
                "min",
                "max",
                "config_prop",
            ):
                if OmegaConf.has_resolver(resolver):
                    OmegaConf.clear_resolver(resolver)
            from timm.layers import set_fused_attn
            from videosaur import models

            # TorchVision's FX feature extractor cannot trace timm's fused
            # attention ``is_causal`` argument. The unfused path is traceable
            # and produces the same frozen DINO features.
            set_fused_attn(False)
        except ImportError as error:
            raise RuntimeError(
                "VideoSAUR is missing from .venv. Run ./setup.sh once."
            ) from error

        config = OmegaConf.load(Path(config))
        model = models.build(config.model, config.optimizer, None, None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        incompatible = model.load_state_dict(state, strict=False)
        loaded = len(state) - len(incompatible.unexpected_keys)
        if loaded == 0:
            raise RuntimeError(f"No VideoSAUR weights were loaded from {checkpoint}")

        self.encoder = model.encoder
        self.processor = model.processor
        self.initializer = model.initializer
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # The checkpoint is always inference-only, even while C-JEPA trains.
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixels):
        squeeze_time = pixels.ndim == 4
        if squeeze_time:
            pixels = pixels.unsqueeze(1)
        if pixels.ndim != 5:
            raise ValueError(f"Expected pixels shaped (B,T,C,H,W), got {pixels.shape}")
        features = self.encoder(pixels)["features"]
        initial = self.initializer(batch_size=pixels.shape[0])
        slots = self.processor(initial, features)["state"]
        return slots[:, 0] if squeeze_time else slots


def build_object_encoder(
    dataset: str,
    config: str | Path,
    checkpoint: str | Path | None = None,
    repo_id: str = "HazelNam/CJEPA",
) -> VideoSAUREncoder:
    path = ensure_checkpoint(dataset, checkpoint, repo_id)
    return VideoSAUREncoder(config=config, checkpoint=path)


__all__ = ["VideoSAUREncoder", "build_object_encoder", "ensure_checkpoint"]
