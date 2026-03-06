"""
Generate 10×10 grid thumbnail videos (MP4) from pre-extracted SAVi slots.

Usage examples
--------------
# From an explicit list of video names:
python probing/phyre/get_thumbnails.py \
    --video_names 0_pixels.mp4 1_pixels.mp4 2_pixels.mp4 \
    --slot_pkl /path/to/slots.pkl \
    --savi_ckpt /path/to/savi_phyre.pth

# From a glob pattern (matches keys inside the slot pickle):
python probing/phyre/get_thumbnails.py \
    --video_glob "*_pixels.mp4" \
    --slot_pkl /path/to/slots.pkl \
    --savi_ckpt /path/to/savi_phyre.pth
"""

import argparse
import glob
import importlib
import math
import os
import pickle
import re
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so ``src.*`` imports work.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Load SAVi model
# ---------------------------------------------------------------------------

def load_savi_model(ckpt_path: str, params_path: str, device: str = "cuda"):
    """Build and load a StoSAVi model from checkpoint + params file."""
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
# Load all slots from pickle
# ---------------------------------------------------------------------------

def load_slot_data(slot_pkl_path: str) -> dict[str, np.ndarray]:
    """
    Load the slot pickle and return a flat dict {video_name: np.ndarray}.
    Merges 'train' and 'val' splits.
    """
    with open(slot_pkl_path, "rb") as f:
        data = pickle.load(f)

    merged: dict[str, np.ndarray] = {}
    for split in ("train", "val"):
        if split in data:
            for k, v in data[split].items():
                merged[k] = np.array(v)
    # fallback: flat dict without splits
    if not merged:
        for k, v in data.items():
            if isinstance(v, (np.ndarray, list)):
                merged[k] = np.array(v)
    return merged


def resolve_video_names(
    slot_data: dict[str, np.ndarray],
    video_names: list[str] | None,
    video_glob: str | None,
    video_range: tuple[int, int] | None = None,
) -> list[str]:
    """
    Return an ordered list of video names to process.

    Priority: video_names > video_range > video_glob > all keys.

    * *video_names*  – explicit list of pickle keys.
    * *video_range*  – (start, end) inclusive numeric id range,
                       matched against the label extracted by _video_label().
    * *video_glob*   – fnmatch pattern matched against pickle keys.
    * (none)         – all keys sorted.
    """
    import fnmatch

    if video_names:
        missing = [v for v in video_names if v not in slot_data]
        if missing:
            print(f"[WARN] {len(missing)} video(s) not found in slot pickle (skipped): {missing[:5]}...")
            video_names = [v for v in video_names if v in slot_data]
        return video_names

    if video_range is not None:
        lo, hi = video_range
        matched = []
        for k in sorted(slot_data.keys()):
            label = _video_label(k)
            try:
                num = int(label)
                if lo <= num <= hi:
                    matched.append(k)
            except ValueError:
                pass
        if not matched:
            raise ValueError(
                f"video_range {lo}-{hi} matched 0 videos in slot pickle. "
                f"Example keys: {list(slot_data.keys())[:5]}"
            )
        return matched

    if video_glob:
        matched = sorted(
            k for k in slot_data if fnmatch.fnmatch(k, video_glob)
        )
        if not matched:
            raise ValueError(
                f"Glob pattern '{video_glob}' matched 0 videos in slot pickle. "
                f"Example keys: {list(slot_data.keys())[:5]}"
            )
        return matched

    return sorted(slot_data.keys())


# ---------------------------------------------------------------------------
# Decode slots → combined reconstruction frames
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_slots_to_frames(
    savi_model,
    slots_np: np.ndarray,
    device: str = "cuda",
    cell_size: int = 64,
) -> np.ndarray:
    """
    Decode slots → reconstruction frames.

    Parameters
    ----------
    slots_np : (T, num_slots, slot_dim)

    Returns
    -------
    recon : (T, cell_size, cell_size, 3) uint8
    """
    slots_tensor = torch.from_numpy(slots_np).float().to(device)
    recon_combined, _, _, _ = savi_model.decode(slots_tensor)
    # recon_combined: (T, 3, H, W) in [-1, 1]
    rc = (recon_combined * 0.5 + 0.5).clamp(0, 1)
    rc_resized = F.interpolate(
        rc, size=(cell_size, cell_size), mode="bilinear", align_corners=False
    )
    recon_np = (
        rc_resized.permute(0, 2, 3, 1).cpu().numpy() * 255
    ).astype(np.uint8)
    return recon_np


# ---------------------------------------------------------------------------
# Extract a human-readable video number from the filename
# ---------------------------------------------------------------------------

def _video_label(name: str) -> str:
    """
    Try to extract a compact identifier from the video filename.
    E.g. "0_pixels.mp4" → "0", "video_007039.mp4" → "007039".
    Falls back to the full name (without extension).
    """
    # pattern: digits before _pixels
    m = re.match(r"^(\d+)_pixels", name)
    if m:
        return m.group(1)
    m = re.search(r"(\d+)", name)
    if m:
        return m.group(1)
    return Path(name).stem


# ---------------------------------------------------------------------------
# Build a single 10×10 grid thumbnail video
# ---------------------------------------------------------------------------

def build_grid_video(
    video_frames: list[np.ndarray],
    video_labels: list[str],
    cell_size: int = 64,
    grid_rows: int = 10,
    grid_cols: int = 10,
    header_h: int = 16,
) -> list[np.ndarray]:
    """
    Assemble a list of per-frame grid images from decoded video frames.

    Parameters
    ----------
    video_frames : list of (T, cell_size, cell_size, 3) uint8 arrays.
                   len <= grid_rows * grid_cols. All must share the same T.
    video_labels : human-readable label per video (same length as video_frames).

    Returns
    -------
    grid_frames : list of (H, W, 3) uint8 arrays, length T.
    """
    max_slots = grid_rows * grid_cols
    n_videos = len(video_frames)
    T = max(v.shape[0] for v in video_frames)

    # Pad shorter videos to T by repeating their last frame
    padded_frames: list[np.ndarray] = []
    for v in video_frames:
        if v.shape[0] < T:
            pad_count = T - v.shape[0]
            padding = np.repeat(v[-1:], pad_count, axis=0)
            v = np.concatenate([v, padding], axis=0)
        padded_frames.append(v)
    video_frames = padded_frames

    tile_h = header_h + cell_size
    tile_w = cell_size
    canvas_h = grid_rows * tile_h
    canvas_w = grid_cols * tile_w

    # Pre-render header strips with labels
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10
        )
    except (IOError, OSError):
        font = ImageFont.load_default()

    header_strips: list[np.ndarray] = []
    for idx in range(max_slots):
        img = Image.new("RGB", (tile_w, header_h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        if idx < n_videos:
            label = video_labels[idx]
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            x = max((tile_w - tw) // 2, 0)
            draw.text((x, 1), label, fill=(255, 255, 255), font=font)
        header_strips.append(np.array(img))

    # Build blank cell for empty positions
    blank_cell = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)

    grid_frames: list[np.ndarray] = []
    for t in range(T):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for idx in range(max_slots):
            row = idx // grid_cols
            col = idx % grid_cols
            y0 = row * tile_h
            x0 = col * tile_w

            # Header
            canvas[y0 : y0 + header_h, x0 : x0 + tile_w] = header_strips[idx]

            # Video frame or blank
            if idx < n_videos:
                canvas[y0 + header_h : y0 + tile_h, x0 : x0 + tile_w] = (
                    video_frames[idx][t]
                )
            else:
                canvas[y0 + header_h : y0 + tile_h, x0 : x0 + tile_w] = blank_cell

        grid_frames.append(canvas)

    return grid_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate 10×10 grid thumbnail videos from SAVi slots."
    )
    parser.add_argument(
        "--video_names", type=str, nargs="*", default=None,
        help="Explicit list of video filenames (keys in the slot pickle).",
    )
    parser.add_argument(
        "--video_glob", type=str, default=None,
        help="Glob/fnmatch pattern matched against slot pickle keys. "
             "Ignored if --video_names is given.",
    )
    parser.add_argument(
        "--video_range", type=int, nargs=2, metavar=("START", "END"), default=None,
        help="Inclusive numeric id range, e.g. --video_range 0 1000. "
             "Matched against the numeric part of each video key. "
             "Ignored if --video_names is given.",
    )
    parser.add_argument(
        "--slot_pkl", type=str,
        default="/cs/data/people/hnam16/phyre_savi_slots.pkl",
        help="Path to the slot pickle file.",
    )
    parser.add_argument(
        "--savi_ckpt", type=str,
        default="/cs/data/people/hnam16/savi_phyre.pth",
        help="Path to the SAVi checkpoint (.pth).",
    )
    parser.add_argument(
        "--savi_params", type=str,
        default="src/third_party/slotformer/base_slots/configs/savi_phyre_params-fold0.py",
        help="Path to the SlotFormerParams .py config file.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="probing/outputs_savi_phyre/thumbnails",
        help="Directory to save thumbnail MP4s.",
    )
    parser.add_argument(
        "--cell_size", type=int, default=64,
        help="Pixel size of each thumbnail cell (square).",
    )
    parser.add_argument(
        "--fps", type=int, default=8,
        help="Frames per second for the output MP4.",
    )
    parser.add_argument(
        "--grid_rows", type=int, default=10,
        help="Number of rows in each grid.",
    )
    parser.add_argument(
        "--grid_cols", type=int, default=10,
        help="Number of columns in each grid.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu).",
    )
    parser.add_argument(
        "--batch_decode", type=int, default=32,
        help="Number of videos to decode at once (GPU memory trade-off).",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    grid_size = args.grid_rows * args.grid_cols  # videos per thumbnail

    # --- 1. Load slot data ---
    print(f"[1] Loading slot pickle: {args.slot_pkl} ...")
    slot_data = load_slot_data(args.slot_pkl)
    print(f"    Total videos in pickle: {len(slot_data)}")

    # --- 2. Resolve video list ---
    video_range = tuple(args.video_range) if args.video_range else None
    video_names = resolve_video_names(slot_data, args.video_names, args.video_glob, video_range)
    print(f"[2] Videos to process: {len(video_names)}")

    # --- 3. Load SAVi model ---
    print(f"[3] Loading SAVi model ...")
    savi_model, _ = load_savi_model(args.savi_ckpt, args.savi_params, device=device)
    print(f"    SAVi loaded on {device}.")

    # --- 4. Decode all videos ---
    print(f"[4] Decoding slots → frames ...")
    all_frames: list[np.ndarray] = []          # each (T, cs, cs, 3)
    all_labels: list[str] = []
    for i, vname in enumerate(video_names):
        slots_np = slot_data[vname]            # (T, S, D)
        frames = decode_slots_to_frames(
            savi_model, slots_np, device=device, cell_size=args.cell_size,
        )
        all_frames.append(frames)
        all_labels.append(_video_label(vname))
        if (i + 1) % 50 == 0 or (i + 1) == len(video_names):
            print(f"    Decoded {i + 1}/{len(video_names)} videos")

    # --- 5. Build & save grid thumbnails ---
    os.makedirs(args.output_dir, exist_ok=True)
    n_pages = math.ceil(len(all_frames) / grid_size)
    print(f"[5] Building {n_pages} thumbnail page(s) ...")

    for page_idx in range(n_pages):
        start = page_idx * grid_size
        end = min(start + grid_size, len(all_frames))
        page_frames = all_frames[start:end]
        page_labels = all_labels[start:end]

        first_label = page_labels[0]
        last_label = page_labels[-1]

        grid = build_grid_video(
            page_frames, page_labels,
            cell_size=args.cell_size,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )

        fname = f"thumbnails_{first_label}-{last_label}.mp4"
        out_path = os.path.join(args.output_dir, fname)
        imageio.mimsave(out_path, grid, fps=args.fps)
        print(f"    Saved: {out_path}  ({len(page_frames)} videos, {len(grid)} frames)")

    print("Done!")


if __name__ == "__main__":
    main()
