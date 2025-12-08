import pathlib
from dataclasses import MISSING, dataclass, field
from functools import reduce
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

ModuleConfig = Dict[str, Any]


@dataclass
class ModelConfig:
    initializer: ModuleConfig
    encoder: ModuleConfig
    grouper: ModuleConfig
    decoder: ModuleConfig
    predictor: Optional[ModuleConfig] = None
    target_encoder: Optional[ModuleConfig] = None
    latent_processor: Optional[ModuleConfig] = None
    mask_resizers: Optional[Dict[str, ModuleConfig]] = None
    losses: Optional[Dict[str, ModuleConfig]] = None
    loss_weights: Optional[Dict[str, float]] = None
    input_type: str = "image"
    target_type: str = "features"
    target_encoder_input: Optional[str] = None
    visualize: bool = False
    eval_mode_config: Optional[Dict[str, Any]] = None
    visualize_every_n_steps: Optional[int] = 1000
    masks_to_visualize: Optional[List[str]] = None
    load_weights: Optional[str] = None
    modules_to_load: Optional[Dict[str, str]] = None


@dataclass
class Config:
    optimizer: ModuleConfig = MISSING
    model: ModelConfig = MISSING
    dataset: ModuleConfig = MISSING
    trainer: Optional[ModuleConfig] = field(default_factory=lambda: {})
    wandb: Optional[Dict[str, Any]] = None
    train_metrics: Optional[Dict[str, ModuleConfig]] = None
    val_metrics: Optional[Dict[str, ModuleConfig]] = None

    globals: Optional[Dict[str, Any]] = None
    experiment_name: Optional[str] = None
    experiment_group: Optional[str] = None
    seed: Optional[int] = None
    checkpoint_every_n_steps: int = 1000


def load_config(path: pathlib.Path, overrides: Optional[List[str]] = None) -> OmegaConf:
    schema = OmegaConf.structured(Config)
    config = OmegaConf.load(path)

    if overrides is not None:
        if isinstance(overrides, list):

            overrides = OmegaConf.from_dotlist(overrides)
        elif isinstance(overrides, dict):
            overrides = OmegaConf.create(overrides)
        else:
            ValueError("overrides should be dotlist or dict")
        config = OmegaConf.merge(schema, config, overrides)
    else:
        config = OmegaConf.merge(schema, config)

    return config


def override_config(
    config: Optional[pathlib.Path] = None,
    override_config_path: Optional[pathlib.Path] = None,
    additional_overrides: Optional[List[str]] = None,
) -> OmegaConf:
    schema = OmegaConf.structured(Config)
    config_objects = [schema, config]
    if override_config_path is not None:
        override_config = OmegaConf.load(override_config_path)
        config_objects.append(override_config)

    if additional_overrides is not None:
        if isinstance(additional_overrides, list):

            additional_overrides = OmegaConf.from_dotlist(additional_overrides)
        elif isinstance(additional_overrides, dict):
            additional_overrides = OmegaConf.create(additional_overrides)
        else:
            ValueError("overrides should be dotlist or dict")
        config_objects.append(additional_overrides)

    config = OmegaConf.merge(*config_objects)
    return config


def save_config(path: pathlib.Path, config: OmegaConf):
    OmegaConf.save(config, path, resolve=True)


def resolver_eval(fn: str, *args):
    params, _, body = fn.partition(":")
    if body == "":
        body = params
        params = ""

    if len(params) == 0:
        arg_names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        assert len(args) <= len(arg_names), f"Only up to {len(arg_names)} arguments are supported"
        params = ",".join(arg_names[: len(args)])

    if not params.startswith("lambda "):
        params = "lambda " + params

    return eval(f"{params}: {body}")(*args)


OmegaConf.register_new_resolver("eval_dummy", resolver_eval)
OmegaConf.register_new_resolver("add", lambda *args: sum(args))
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda *args: reduce(lambda prod, cur: prod * cur, args, 1))
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("min", lambda *args: min(*args))
OmegaConf.register_new_resolver("max", lambda *args: max(*args))
# should be useful in case of dependencies between config params
OmegaConf.register_new_resolver(
    "config_prop", lambda prop, *keys: get_predefined_property(prop, keys)
)

VIT_PARAMS = {
    "vit_small_patch16_224_dino": {"FEAT_DIM": 384, "NUM_PATCHES": 196},
    "vit_small_patch8_224_dino": {"FEAT_DIM": 384, "NUM_PATCHES": 784},
    "vit_base_patch16_224_dino": {"FEAT_DIM": 768, "NUM_PATCHES": 196},
    "vit_base_patch16_448_dino": {"FEAT_DIM": 768, "NUM_PATCHES": 784},
    "vit_base_patch8_224_dino": {"FEAT_DIM": 768, "NUM_PATCHES": 784},
    "vit_base_patch16_224_mae": {"FEAT_DIM": 768, "NUM_PATCHES": 196},
    "vit_base_patch16_224_mocov3": {"FEAT_DIM": 768, "NUM_PATCHES": 196},
    "vit_base_patch16_224_msn": {"FEAT_DIM": 768, "NUM_PATCHES": 196},
    "vit_base_patch14_dinov2": {"FEAT_DIM": 768, "NUM_PATCHES": 256},
    "vit_small_patch14_dinov2": {"FEAT_DIM": 384, "NUM_PATCHES": 256},
    "vit_large_patch14_dinov2": {"FEAT_DIM": 1024, "NUM_PATCHES": 256},
}


def get_predefined_property(prop, keys):
    value = globals()[prop]
    for key in keys:
        if callable(value):
            value = value(key)
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise ValueError(f"Can not handle type {type(value)}")
    return value
