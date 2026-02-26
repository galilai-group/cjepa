"""
Probing configuration for loading videosaur model.

This config mirrors the relevant pieces from configs/config_train_causal_clevrer_slot.yaml
so that probing code doesn't require the full training config.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideosaurConfig:
    NUM_SLOTS: int = 7
    SLOT_DIM: int = 128
    DINO_MODEL: str = "vit_small_patch14_dinov2.lvd142m"
    FEAT_DIM: int = 384
    NUM_PATCHES: int = 196
    INPUT_SIZE: int = 196  # resolution for videosaur inference


@dataclass
class PredictorConfig:
    depth: int = 6
    heads: int = 16
    mlp_dim: int = 2048
    dim_head: int = 64
    dropout: float = 0.1


@dataclass
class ProbingConfig:
    # Videosaur
    videosaur: VideosaurConfig = field(default_factory=VideosaurConfig)
    
    # C-JEPA predictor
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    
    # Model architecture (must match training config)
    slot_dim: int = 128  # pure slot dim from videosaur
    num_slots: int = 7
    
    # Videosaur model config path (for building the model architecture)
    videosaur_model_config: str = "src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml"
    
    # Frameskip (1 = no frameskip, slots index = video frame index)
    frameskip: int = 1
    
    # Output
    output_dir: str = "probing/outputs"
    
    # Device
    device: str = "cuda"


def get_default_config() -> ProbingConfig:
    return ProbingConfig()
