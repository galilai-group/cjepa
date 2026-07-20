"""Roll CLEVRER slots from 128 to 160 frames with a trained C-JEPA model."""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(name)


def load_cjepa(path: str | Path, device: torch.device):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"C-JEPA object checkpoint not found: {path}")
    model = torch.load(path, map_location=device, weights_only=False)
    if not hasattr(model, "predict") or not hasattr(model, "history_frames"):
        raise TypeError(
            f"{path} is not a serialized C-JEPA object checkpoint. "
            "Use the *_object.ckpt file written by train.py."
        )
    return model.to(device).eval()


@torch.inference_mode()
def extend_slots(
    model,
    slots: torch.Tensor,
    *,
    target_frames: int,
    frameskip: int,
) -> torch.Tensor:
    """Autoregressively extend a batch while preserving frame offsets."""
    observed = slots.shape[1]
    if target_frames <= observed:
        return slots[:, :target_frames]
    history_frames = int(model.history_frames)
    if observed < history_frames * frameskip:
        raise ValueError(
            f"Need at least {history_frames * frameskip} observed frames, "
            f"got {observed}"
        )

    future_count = target_frames - observed
    streams = []
    for offset in range(frameskip):
        start = observed - history_frames * frameskip + offset
        history = slots[:, start:observed:frameskip]
        if history.shape[1] != history_frames:
            raise RuntimeError("Could not construct the C-JEPA history window")
        predictions = []
        required = (future_count + frameskip - 1 - offset) // frameskip
        while sum(chunk.shape[1] for chunk in predictions) < required:
            prediction = model.predict(history, inference=True)
            predictions.append(prediction)
            history = torch.cat([history, prediction], dim=1)[:, -history_frames:]
        streams.append(torch.cat(predictions, dim=1)[:, :required])

    future = torch.empty(
        slots.shape[0],
        future_count,
        *slots.shape[2:],
        device=slots.device,
        dtype=slots.dtype,
    )
    for index in range(future_count):
        future[:, index] = streams[index % frameskip][:, index // frameskip]
    return torch.cat([slots, future], dim=1)


def rollout_file(model, source_path: Path, output_path: Path, cfg, device):
    if output_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"{output_path} exists; set rollout.overwrite=true")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.unlink(missing_ok=True)

    try:
        with h5py.File(source_path, "r") as source:
            if "slots" not in source:
                raise KeyError(f"{source_path} does not contain a 'slots' column")
            lengths = np.asarray(source["ep_len"][:], dtype=np.int64)
            offsets = np.asarray(source["ep_offset"][:], dtype=np.int64)
            if np.any(lengths < 128):
                raise ValueError(
                    "Every input episode must contain at least 128 frames"
                )
            observed = min(128, int(lengths.min()))
            if "episode_idx" in source:
                episode_ids = np.asarray(source["episode_idx"][offsets], np.int64)
            else:
                episode_ids = np.arange(len(lengths), dtype=np.int64)
            slot_shape = tuple(source["slots"].shape[1:])
            total = len(lengths) * int(cfg.target_frames)

            with h5py.File(temporary, "w", libver="latest") as output:
                compression = hdf5plugin.Blosc(
                    cname="zstd",
                    clevel=5,
                    shuffle=hdf5plugin.Blosc.SHUFFLE,
                )
                out_slots = output.create_dataset(
                    "slots",
                    shape=(total, *slot_shape),
                    chunks=(min(128, max(total, 1)), *slot_shape),
                    dtype=np.float32,
                    **compression,
                )
                out_episode = output.create_dataset(
                    "episode_idx", shape=(total,), dtype=np.int64
                )
                out_step = output.create_dataset(
                    "step_idx", shape=(total,), dtype=np.int64
                )
                output.create_dataset(
                    "ep_len", data=np.full(len(lengths), cfg.target_frames, np.int32)
                )
                output.create_dataset(
                    "ep_offset",
                    data=np.arange(len(lengths), dtype=np.int64) * cfg.target_frames,
                )
                for key, value in source.attrs.items():
                    output.attrs[key] = value
                output.attrs["rollout_model"] = str(Path(cfg.checkpoint).expanduser())
                output.attrs["observed_frames"] = observed

                for begin in tqdm(
                    range(0, len(lengths), cfg.batch_size),
                    desc=source_path.stem,
                ):
                    end = min(begin + cfg.batch_size, len(lengths))
                    batch = []
                    for episode in range(begin, end):
                        offset = int(offsets[episode])
                        batch.append(source["slots"][offset : offset + observed])
                    values = torch.from_numpy(
                        np.stack(batch).astype(np.float32)
                    ).to(device)
                    values = extend_slots(
                        model,
                        values,
                        target_frames=cfg.target_frames,
                        frameskip=cfg.frameskip,
                    ).cpu().numpy()
                    row_begin = begin * cfg.target_frames
                    row_end = end * cfg.target_frames
                    out_slots[row_begin:row_end] = values.reshape(-1, *slot_shape)
                    out_episode[row_begin:row_end] = np.repeat(
                        episode_ids[begin:end], cfg.target_frames
                    )
                    out_step[row_begin:row_end] = np.tile(
                        np.arange(cfg.target_frames), end - begin
                    )
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Wrote {output_path}")


@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig):
    device = resolve_device(cfg.rollout.device)
    model = load_cjepa(cfg.rollout.checkpoint, device)
    for split in cfg.rollout.splits:
        source = Path(cfg.rollout.inputs[split]).expanduser().resolve()
        output = Path(cfg.rollout.outputs[split]).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(
                f"Missing {split} slots: {source}. Run scripts/extract_slots.py first."
            )
        rollout_file(model, source, output, cfg.rollout, device)


if __name__ == "__main__":
    run()
