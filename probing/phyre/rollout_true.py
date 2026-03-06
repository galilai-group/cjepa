"""
Ground-truth slot decoding for PHYRE (comparison baseline for rollout.py).

Decodes pre-extracted SAVi slots (frames 0-10) directly through the SAVi
decoder — no C-JEPA prediction involved.  Produces MP4s with the same
frame count as rollout.py so they can be compared side-by-side.

All frames show a blue indicator (top-left) since every frame is real.

Usage
-----
python probing/phyre/rollout_true.py \
    --video_names 0_pixels.mp4 1_pixels.mp4 \
    --slot_pkl /path/to/slots.pkl \
    --savi_ckpt /path/to/savi_phyre.pth
"""

import argparse
import importlib
import os
import pickle
import re
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Load SAVi model
# ---------------------------------------------------------------------------

def load_savi_model(ckpt_path: str, params_path: str, device: str = "cuda"):
    from src.third_party.slotformer.base_slots.models import build_model

    if params_path.endswith(".py"):
        params_module_path = params_path[:-3]
    else:
        params_module_path = params_path
    sys.path.append(os.path.dirname(params_module_path))
    params_module = importlib.import_module(os.path.basename(params_module_path))
    params = params_module.SlotFormerParams()

    model = build_model(params)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model, params


# ---------------------------------------------------------------------------
# Load slot data
# ---------------------------------------------------------------------------

def load_slot_data(slot_pkl_path: str) -> dict[str, np.ndarray]:
    with open(slot_pkl_path, "rb") as f:
        data = pickle.load(f)

    merged: dict[str, np.ndarray] = {}
    for split in ("train", "val"):
        if split in data:
            for k, v in data[split].items():
                merged[k] = np.array(v)
    if not merged:
        for k, v in data.items():
            if isinstance(v, (np.ndarray, list)):
                merged[k] = np.array(v)
    return merged


def resolve_video_names(
    slot_data: dict[str, np.ndarray],
    video_names: list[str] | None,
    video_range: tuple[int, int] | None = None,
) -> list[str]:
    if video_names:
        missing = [v for v in video_names if v not in slot_data]
        if missing:
            print(f"[WARN] {len(missing)} video(s) not found (skipped): {missing[:5]}...")
            video_names = [v for v in video_names if v in slot_data]
        return video_names

    if video_range is not None:
        lo, hi = video_range
        matched = []
        for k in sorted(slot_data.keys()):
            m = re.match(r"^(\d+)", k)
            if m:
                num = int(m.group(1))
                if lo <= num <= hi:
                    matched.append(k)
        if not matched:
            raise ValueError(
                f"video_range {lo}-{hi} matched 0 videos. "
                f"Example keys: {list(slot_data.keys())[:5]}"
            )
        return matched

    return sorted(slot_data.keys())


# ---------------------------------------------------------------------------
# Decode slots → frames
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_slots_to_frames(
    savi_model,
    slots_np: np.ndarray,
    device: str = "cuda",
    cell_size: int = 128,
) -> np.ndarray:
    slots_tensor = torch.from_numpy(slots_np).float().to(device)
    recon_combined, _, _, _ = savi_model.decode(slots_tensor)
    rc = (recon_combined * 0.5 + 0.5).clamp(0, 1)
    rc_resized = F.interpolate(
        rc, size=(cell_size, cell_size), mode="bilinear", align_corners=False
    )
    recon_np = (
        rc_resized.permute(0, 2, 3, 1).cpu().numpy() * 255
    ).astype(np.uint8)
    return recon_np


# ---------------------------------------------------------------------------
# Add blue indicator (all real)
# ---------------------------------------------------------------------------

def add_indicator(frames: np.ndarray) -> np.ndarray:
    T, H, W, _ = frames.shape
    sq_h = max(H // 8, 2)
    sq_w = max(W // 8, 2)
    for t in range(T):
        frames[t, :sq_h, :sq_w, 0] = 0
        frames[t, :sq_h, :sq_w, 1] = 0
        frames[t, :sq_h, :sq_w, 2] = 255
    return frames


# ---------------------------------------------------------------------------
# Video label helper
# ---------------------------------------------------------------------------

def _video_label(name: str) -> str:
    m = re.match(r"^(\d+)_pixels", name)
    if m:
        return m.group(1)
    m = re.search(r"(\d+)", name)
    if m:
        return m.group(1)
    return Path(name).stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ground-truth slot decoding for PHYRE (no prediction)."
    )
    parser.add_argument(
        "--video_names", type=str, nargs="*", default=None,
    )
    parser.add_argument(
        "--video_range", type=int, nargs=2, metavar=("START", "END"), default=None,
    )
    parser.add_argument(
        "--slot_pkl", type=str,
        default="/cs/data/people/hnam16/phyre_savi_slots.pkl",
    )
    parser.add_argument(
        "--savi_ckpt", type=str,
        default="/cs/data/people/hnam16/savi_phyre.pth",
    )
    parser.add_argument(
        "--savi_params", type=str,
        default="src/third_party/slotformer/base_slots/configs/savi_phyre_params-fold0.py",
    )
    parser.add_argument(
        "--output_dir", type=str, default="probing/outputs_savi_phyre/rollout_true",
    )
    parser.add_argument("--cell_size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument(
        "--num_frames", type=int, default=11,
        help="Total number of frames to decode (should match rollout length, default 5+6=11).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # --- 1. Load slot data ---
    print(f"[1] Loading slot pickle: {args.slot_pkl} ...")
    slot_data = load_slot_data(args.slot_pkl)
    print(f"    Total videos in pickle: {len(slot_data)}")

    # --- 2. Resolve video list ---
    video_range = tuple(args.video_range) if args.video_range else None
    video_names = resolve_video_names(slot_data, args.video_names, video_range)
    print(f"[2] Videos to process: {len(video_names)}")

    # --- 3. Load SAVi model ---
    print(f"[3] Loading SAVi model ...")
    savi_model, _ = load_savi_model(args.savi_ckpt, args.savi_params, device=device)
    print(f"    SAVi loaded on {device}.")

    # --- 4. Decode + save ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[4] Decoding {args.num_frames} frames per video ...")

    for i, vname in enumerate(video_names):
        slots_np = slot_data[vname]  # (T, S, D)

        if slots_np.shape[0] < args.num_frames:
            print(f"  [SKIP] {vname}: only {slots_np.shape[0]} frames "
                  f"(need {args.num_frames})")
            continue

        # Take first num_frames frames
        slots_np = slots_np[: args.num_frames]

        frames = decode_slots_to_frames(
            savi_model, slots_np, device=device, cell_size=args.cell_size,
        )

        frames = add_indicator(frames)

        label = _video_label(vname)
        fname = f"rollout_true_{label}.mp4"
        out_path = os.path.join(args.output_dir, fname)
        imageio.mimsave(out_path, frames, fps=args.fps)

        if (i + 1) % 50 == 0 or (i + 1) == len(video_names):
            print(f"    Saved {i + 1}/{len(video_names)} videos")

    print(f"Done! Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
