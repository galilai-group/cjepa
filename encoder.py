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
from transformers import Dinov2Config, Dinov2Model

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


class DINOv2FrameEncoder(nn.Module):
    """The HuggingFace DINOv2 FrameEncoder used by the released checkpoints.

    VideoSAUR's current generic ``FrameEncoder`` cannot consume a Transformers
    ``BaseModelOutput`` directly.  This forward method intentionally mirrors
    the FrameEncoder bundled with the original C-JEPA repository.
    """

    def __init__(self, backbone: nn.Module, output_transform: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone.config.output_hidden_states = True
        self.pos_embed = None
        self.output_transform = output_transform
        self.spatial_flatten = False
        self.main_features_key = "vit_block12"

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        layer12 = self.backbone.encoder.layer[11]
        attention = layer12.attention.attention
        with torch.no_grad():
            output = self.backbone(images, output_hidden_states=True)

        backbone_features = output.last_hidden_state.detach()[:, 1:]
        hidden_before_layer12 = output.hidden_states[-2]
        normalized = layer12.norm1(hidden_before_layer12)
        keys = attention.key(normalized)[:, 1:]
        features = self.output_transform(backbone_features.clone())
        return {
            "features": features,
            "backbone_features": backbone_features,
            "vit_block12": backbone_features,
            "vit_block_keys12": keys,
        }


def _dinov2_small() -> Dinov2Model:
    # Instantiate without network access: every parameter is subsequently
    # loaded strictly from the VideoSAUR checkpoint.
    config = Dinov2Config(
        image_size=518,
        patch_size=14,
        num_channels=3,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        use_mask_token=True,
        apply_layernorm=True,
        use_swiglu_ffn=False,
        layerscale_value=1.0,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-6,
    )
    return Dinov2Model(config)


def _videosaur_modules(num_slots: int, slot_dim: int):
    try:
        from videosaur.modules import groupers, initializers, networks, video
    except ImportError as error:
        raise RuntimeError(
            "VideoSAUR is missing from .venv. Run ./setup.sh once, then use "
            "`source /grid/klindt/home/nam/cjepa/.venv/bin/activate`."
        ) from error

    output_transform = networks.MLP(
        384, slot_dim, [768], initial_layer_norm=True
    )
    encoder = video.MapOverTime(
        DINOv2FrameEncoder(_dinov2_small(), output_transform)
    )
    initializer = initializers.RandomInit(n_slots=num_slots, dim=slot_dim)
    corrector = groupers.SlotAttention(
        inp_dim=slot_dim,
        slot_dim=slot_dim,
        n_iters=2,
        use_mlp=False,
    )
    predictor = networks.TransformerEncoder(dim=slot_dim, n_blocks=1, n_heads=4)
    processor = video.ScanOverTime(
        video.LatentProcessor(
            corrector=corrector,
            predictor=predictor,
            first_step_corrector_args={"n_iters": 3},
        )
    )
    return encoder, processor, initializer


class VideoSAUREncoder(nn.Module):
    """Frozen legacy-compatible VideoSAUR encoder mapping video to slots."""

    architecture = "videosaur-hf-dinov2-frame-encoder-v1"

    def __init__(
        self,
        config: str | Path,
        checkpoint: str | Path | None = None,
        component_states: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__()
        # Retain this argument for the public API and Hydra configuration.  The
        # architecture is fixed because released C-JEPA weights require it.
        self.config = str(config)
        parsed_config = OmegaConf.load(Path(config))
        num_slots = int(parsed_config.model.initializer.n_slots)
        slot_dim = int(parsed_config.model.initializer.dim)
        self.encoder, self.processor, self.initializer = _videosaur_modules(
            num_slots, slot_dim
        )

        if component_states is None:
            if checkpoint is None:
                raise ValueError("A checkpoint or component_states must be provided")
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = state.get("state_dict", state)
            component_states = {
                name: {
                    key.removeprefix(f"{name}."): value
                    for key, value in state.items()
                    if key.startswith(f"{name}.")
                }
                for name in ("encoder", "processor", "initializer")
            }

        for name in ("encoder", "processor", "initializer"):
            state = component_states.get(name, {})
            if not state:
                raise RuntimeError(f"VideoSAUR checkpoint is missing {name} weights")
            try:
                getattr(self, name).load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise RuntimeError(
                    f"VideoSAUR {name} weights do not match the legacy "
                    "HuggingFace FrameEncoder architecture"
                ) from error

        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # The checkpoint is always inference-only, even while C-JEPA trains.
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixels, initial_slots=None, return_last_slot=False):
        squeeze_time = pixels.ndim == 4
        if squeeze_time:
            pixels = pixels.unsqueeze(1)
        if pixels.ndim != 5:
            raise ValueError(f"Expected pixels shaped (B,T,C,H,W), got {pixels.shape}")
        features = self.encoder(pixels)["features"]
        initial = (
            self.initializer(batch_size=pixels.shape[0])
            if initial_slots is None
            else initial_slots
        )
        slots = self.processor(initial, features)["state"]
        output = slots[:, 0] if squeeze_time else slots
        if return_last_slot:
            return output, slots[:, -1].detach().clone()
        return output


def build_object_encoder(
    dataset: str,
    config: str | Path,
    checkpoint: str | Path | None = None,
    repo_id: str = "HazelNam/CJEPA",
    component_states: dict[str, dict[str, torch.Tensor]] | None = None,
) -> VideoSAUREncoder:
    path = (
        None
        if component_states is not None
        else ensure_checkpoint(dataset, checkpoint, repo_id)
    )
    return VideoSAUREncoder(
        config=config,
        checkpoint=path,
        component_states=component_states,
    )


__all__ = [
    "DINOv2FrameEncoder",
    "VideoSAUREncoder",
    "build_object_encoder",
    "ensure_checkpoint",
]
