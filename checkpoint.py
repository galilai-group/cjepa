"""Download and upgrade released C-JEPA planning checkpoints."""

from __future__ import annotations

import gc
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch import nn

from cjepa import CJEPA
from encoder import DINOv2FrameEncoder, build_object_encoder

POLICY_PATTERN = re.compile(r"^(?P<dataset>[a-z0-9_-]+)_m(?P<masks>\d+)$")


def policy_spec(policy: str) -> tuple[str, int]:
    """Extract the dataset and masked-slot count from a policy name."""
    name = Path(policy).name
    if name.endswith("_object.ckpt"):
        name = name.removesuffix("_object.ckpt")
    match = POLICY_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            "Downloaded policies must be named '<dataset>_m<masked-slots>', "
            f"for example 'cjepa/pusht_m2'; got {policy!r}"
        )
    return match.group("dataset"), int(match.group("masks"))


def policy_checkpoint_path(policy: str, checkpoint_root: str | Path) -> Path:
    """Resolve a stable-worldmodel policy name to its object checkpoint."""
    path = Path(policy).expanduser()
    if path.name.endswith("_object.ckpt"):
        checkpoint = path
    else:
        checkpoint = path.with_name(f"{path.name}_object.ckpt")
    if not checkpoint.is_absolute():
        checkpoint = Path(checkpoint_root).expanduser() / checkpoint
    return checkpoint


def released_checkpoint_name(
    dataset: str,
    num_masked_slots: int,
    *,
    backbone: str = "videosaur",
    epoch: int = 30,
) -> str:
    """Return the current legacy filename used in the Hugging Face repo."""
    return (
        f"cjepa-ckpts/{dataset}_{backbone}_{num_masked_slots}"
        f"_epoch_{epoch}_object.ckpt"
    )


class _LegacyModule(nn.Module):
    """Placeholder that lets PyTorch recover parameters from legacy modules."""


_LEGACY_CLASSES = {
    "custom_models.dinowm_causal_AP_node": ["CausalWM_AP"],
    "custom_models.cjepa_predictor": [
        "MaskedSlot_AP_Predictor",
        "NonCausalTransformer",
    ],
    "videosaur.videosaur.modules.video": [
        "MapOverTime",
        "ScanOverTime",
        "LatentProcessor",
    ],
    "videosaur.videosaur.modules.encoders": ["FrameEncoder"],
    "videosaur.videosaur.modules.networks": [
        "MLP",
        "TransformerEncoder",
        "TransformerEncoderLayer",
        "Attention",
    ],
    "videosaur.videosaur.modules.groupers": ["SlotAttention"],
    "videosaur.videosaur.modules.initializers": ["RandomInit"],
    "stable_worldmodel.wm.dinowm": ["Embedder"],
}
_LEGACY_PACKAGES = (
    "custom_models",
    "videosaur.videosaur",
    "videosaur.videosaur.modules",
    "stable_worldmodel.wm",
)


@contextmanager
def _legacy_imports():
    """Temporarily supply module names embedded in the released checkpoints."""
    names = (*_LEGACY_PACKAGES, *_LEGACY_CLASSES)
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name in _LEGACY_PACKAGES:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        for module_name, class_names in _LEGACY_CLASSES.items():
            module = types.ModuleType(module_name)
            for class_name in class_names:
                legacy_class = type(
                    class_name,
                    (_LegacyModule,),
                    {"__module__": module_name},
                )
                setattr(module, class_name, legacy_class)
            sys.modules[module_name] = module
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _find_current_model(module) -> CJEPA | None:
    if isinstance(module, CJEPA):
        return module
    if isinstance(module, nn.Module):
        for child in module.children():
            result = _find_current_model(child)
            if result is not None:
                return result
    return None


def _load_if_current(path: Path) -> CJEPA | None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None
    return _find_current_model(checkpoint)


def _legacy_trainable_state(path: Path):
    with _legacy_imports():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    world_model = getattr(checkpoint, "model", checkpoint)
    required = ("predictor", "action_encoder", "proprio_encoder")
    missing = [name for name in required if not hasattr(world_model, name)]
    if missing:
        raise RuntimeError(
            f"Legacy checkpoint is missing planning modules: {', '.join(missing)}"
        )

    predictor = world_model.predictor
    attributes = {
        name: getattr(predictor, name, None)
        for name in (
            "num_slots",
            "slot_dim",
            "history_frames",
            "pred_frames",
            "num_masked_slots",
        )
    }
    states = {
        name: {
            key: value.detach().cpu().clone()
            for key, value in getattr(world_model, name).state_dict().items()
        }
        for name in required
    }
    object_modules = {
        "encoder": world_model.encoder.backbone,
        "processor": world_model.slot_attention.backbone,
        "initializer": world_model.initializer.backbone,
    }
    states["object_encoder"] = {
        name: {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }
        for name, module in object_modules.items()
    }
    del checkpoint, world_model, predictor
    gc.collect()
    return attributes, states


def _current_model_config(dataset: str, num_masked_slots: int):
    project_dir = Path(__file__).resolve().parent
    data = OmegaConf.load(project_dir / "config" / "data" / f"{dataset}.yaml")
    train = OmegaConf.load(project_dir / "config" / "train.yaml")
    combined = OmegaConf.create(
        {"data": data, "seed": train.seed, "model": train.model}
    )
    model = OmegaConf.to_container(combined.model, resolve=True)
    model["num_masked_slots"] = num_masked_slots
    return project_dir, data, model


def _load_legacy_embedder(embedder: nn.Module, state: dict[str, torch.Tensor]):
    renamed = {
        key.replace("patch_embed.", "projection.", 1): value
        for key, value in state.items()
    }
    embedder.load_state_dict(renamed, strict=True)


def convert_legacy_checkpoint(
    path: str | Path,
    *,
    dataset: str,
    num_masked_slots: int,
    repo_id: str = "HazelNam/CJEPA",
) -> CJEPA:
    """Replace a released legacy model object with the current CJEPA model."""
    path = Path(path)
    legacy, states = _legacy_trainable_state(path)
    project_dir, data, model_config = _current_model_config(
        dataset, num_masked_slots
    )

    expected = {
        "num_slots": model_config["num_object_slots"] + 2,
        "slot_dim": model_config["slot_dim"],
        "history_frames": model_config["history_frames"],
        "pred_frames": model_config["pred_frames"],
        "num_masked_slots": num_masked_slots,
    }
    differences = {
        name: (legacy[name], value)
        for name, value in expected.items()
        if legacy[name] != value
    }
    if differences:
        details = ", ".join(
            f"{name}: released={old}, current={new}"
            for name, (old, new) in differences.items()
        )
        raise RuntimeError(f"Incompatible C-JEPA architecture ({details})")

    encoder = data.encoder
    object_encoder = build_object_encoder(
        dataset=encoder.dataset,
        config=project_dir / encoder.config,
        repo_id=repo_id,
        component_states=states["object_encoder"],
    )
    model = CJEPA(object_encoder=object_encoder, **model_config)
    try:
        model.predictor.load_state_dict(states["predictor"], strict=True)
        _load_legacy_embedder(model.action_encoder, states["action_encoder"])
        _load_legacy_embedder(model.proprio_encoder, states["proprio_encoder"])
    except RuntimeError as error:
        raise RuntimeError(
            "Released predictor or action/proprio tensors do not fit the "
            "current CJEPA architecture"
        ) from error
    model.eval()

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(model, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return model


def ensure_planning_checkpoint(
    policy: str,
    checkpoint_root: str | Path,
    *,
    repo_id: str = "HazelNam/CJEPA",
    revision: str = "main",
    backbone: str = "videosaur",
    epoch: int = 30,
    filename: str | None = None,
) -> Path:
    """Download, rename, and if necessary upgrade a planning checkpoint."""
    dataset, num_masked_slots = policy_spec(policy)
    if dataset != "pusht":
        raise ValueError("eval.py planning currently supports the 'pusht' policy only")

    target = policy_checkpoint_path(policy, checkpoint_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    current = _load_if_current(target) if target.exists() else None
    encoder_module = (
        getattr(getattr(current, "object_encoder", None), "encoder", None)
        if current is not None
        else None
    )
    frame_encoder = getattr(encoder_module, "module", None)
    needs_architecture_upgrade = current is not None and not isinstance(
        frame_encoder, DINOv2FrameEncoder
    )

    if not target.exists() or needs_architecture_upgrade:
        remote_name = filename or released_checkpoint_name(
            dataset,
            num_masked_slots,
            backbone=backbone,
            epoch=epoch,
        )
        print(f"Downloading {repo_id}/{remote_name} ...")
        with TemporaryDirectory(prefix=".cjepa-download-", dir=target.parent) as stage:
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_name,
                    revision=revision,
                    local_dir=stage,
                )
            )
            downloaded.replace(target)
        if needs_architecture_upgrade:
            print(f"Replaced the incompatible converted checkpoint at {target}")
        else:
            print(f"Renamed checkpoint to {target}")

    current = _load_if_current(target)
    if current is None:
        print("Released checkpoint uses the legacy architecture; upgrading it ...")
        convert_legacy_checkpoint(
            target,
            dataset=dataset,
            num_masked_slots=num_masked_slots,
            repo_id=repo_id,
        )
        print(
            "Upgraded predictor weights and action/proprio encoders to the "
            f"current CJEPA architecture: {target}"
        )
    elif current.predictor.num_masked_slots != num_masked_slots:
        raise RuntimeError(
            f"{target} contains m{current.predictor.num_masked_slots}, "
            f"but policy requests m{num_masked_slots}"
        )
    else:
        print(f"Using existing checkpoint: {target}")
    return target


__all__ = [
    "convert_legacy_checkpoint",
    "ensure_planning_checkpoint",
    "policy_checkpoint_path",
    "policy_spec",
    "released_checkpoint_name",
]
