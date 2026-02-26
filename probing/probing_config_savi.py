"""
Probing configuration for loading SAVi (StoSAVi) model.

This config mirrors the relevant pieces from the SAVi params so that
probing code doesn't require the full training config.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SaviConfig:
    params: str = "src/third_party/slotformer/base_slots/configs/stosavi_clevrer_params.py"
    weight: str = ""  # path to SAVi checkpoint, set via CLI
    NUM_SLOTS: int = 7
    SLOT_DIM: int = 128
    RESOLUTION: tuple = (64, 64)  # SAVi native resolution
    INPUT_FRAMES: int = 6


@dataclass
class PredictorConfig:
    depth: int = 6
    heads: int = 16
    mlp_dim: int = 2048
    dim_head: int = 64
    dropout: float = 0.1


@dataclass
class ProbingConfigSavi:
    # SAVi
    savi: SaviConfig = field(default_factory=SaviConfig)

    # C-JEPA predictor
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    # Model architecture (must match training config)
    slot_dim: int = 128  # pure slot dim from SAVi
    num_slots: int = 7

    # Frameskip (1 = no frameskip, slots index = video frame index)
    frameskip: int = 1

    # Output
    output_dir: str = "probing/outputs"

    # Device
    device: str = "cuda"


def get_default_config() -> ProbingConfigSavi:
    return ProbingConfigSavi()
