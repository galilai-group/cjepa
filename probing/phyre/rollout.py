"""
Autoregressive rollout for C-JEPA on PHYRE.

Takes pre-extracted SAVi slots (frames 0-4), autoregressively predicts
frames 5-10 using the C-JEPA predictor, decodes all 11 frames through
the SAVi decoder, and saves each video as an MP4.

Blue square (top-left) = real input slot frame.
Red  square (top-left) = predicted slot frame.

Usage
-----
python probing/phyre/rollout.py \
    --video_names 0_pixels.mp4 1_pixels.mp4 \
    --slot_pkl /path/to/slots.pkl \
    --savi_ckpt /path/to/savi_phyre.pth \
    --cjepa_ckpt /path/to/predictor.ckpt
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
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so ``src.*`` imports work.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.phyre.probing_config_savi import get_default_config


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
# Load C-JEPA predictor
# ---------------------------------------------------------------------------

def load_cjepa_predictor(ckpt_path: str, num_mask_slots: int, configs, device: str = "cuda"):
    from src.cjepa_predictor import MaskedSlotPredictor

    spt_module = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    predictor = MaskedSlotPredictor(
        num_slots=configs.savi.NUM_SLOTS,
        slot_dim=configs.savi.SLOT_DIM,
        history_frames=configs.savi.INPUT_FRAMES,
        pred_frames=configs.savi.OUTPUT_FRAMES,
        num_masked_slots=num_mask_slots,
        seed=0,
        depth=configs.predictor.depth,
        heads=configs.predictor.heads,
        dim_head=configs.predictor.dim_head,
        mlp_dim=configs.predictor.mlp_dim,
        dropout=configs.predictor.dropout,
    )
    missing, unexpected = predictor.load_state_dict(spt_module, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    predictor.eval()
    predictor.to(device)
    return predictor


# ---------------------------------------------------------------------------
# Load slot data
# ---------------------------------------------------------------------------

def load_slot_data(slot_pkl_path: str) -> dict[str, np.ndarray]:
    """Load the slot pickle and return a flat dict {video_name: np.ndarray}."""
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
    """Resolve which videos to process."""
    import fnmatch

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
# Autoregressive rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def autoregressive_rollout(
    predictor,
    slots_np: np.ndarray,
    num_input_frames: int = 5,
    num_pred_frames: int = 6,
    device: str = "cuda",
) -> np.ndarray:
    """
    Autoregressively predict future slots.

    Parameters
    ----------
    slots_np : (T_total, S, D)  — all pre-extracted slots
    num_input_frames : number of real input frames (default 5)
    num_pred_frames  : number of frames to predict (default 6)

    Returns
    -------
    all_slots : (num_input_frames + num_pred_frames, S, D)  float32 numpy
        Frames 0..num_input_frames-1 are real, the rest are predicted.
    """
    S, D = slots_np.shape[1], slots_np.shape[2]

    # Start with the real input frames
    real_slots = torch.from_numpy(slots_np[:num_input_frames]).float().to(device)
    # Will collect: [real_0, ..., real_4, pred_5, ..., pred_10]
    collected = [real_slots]  # list of (K, S, D) tensors

    # Current sliding window
    window = real_slots.clone()  # (5, S, D)

    for step in range(num_pred_frames):
        # inference expects (B, T_hist, S, D)
        x = window.unsqueeze(0)  # (1, 5, S, D)
        pred = predictor.inference(x)  # (1, 1, S, D)
        pred_frame = pred[0]  # (1, S, D)
        collected.append(pred_frame)

        # Slide window: drop oldest, append predicted
        window = torch.cat([window[1:], pred_frame], dim=0)  # (5, S, D)

    all_slots = torch.cat(collected, dim=0)  # (11, S, D)
    return all_slots.cpu().numpy()


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
    """
    Decode slots → reconstruction frames.

    Returns
    -------
    recon : (T, cell_size, cell_size, 3) uint8
    """
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
# Add indicator square to frames
# ---------------------------------------------------------------------------

def add_indicator(
    frames: np.ndarray,
    num_input_frames: int,
) -> np.ndarray:
    """
    Add a coloured square to the top-left corner of each frame.
    Blue = real input slot, Red = predicted slot.

    Parameters
    ----------
    frames : (T, H, W, 3) uint8

    Returns
    -------
    frames : (T, H, W, 3) uint8  (modified in-place)
    """
    T, H, W, _ = frames.shape
    sq_h = max(H // 8, 2)
    sq_w = max(W // 8, 2)

    for t in range(T):
        if t < num_input_frames:
            # Blue (real)
            frames[t, :sq_h, :sq_w, 0] = 0
            frames[t, :sq_h, :sq_w, 1] = 0
            frames[t, :sq_h, :sq_w, 2] = 255
        else:
            # Red (predicted)
            frames[t, :sq_h, :sq_w, 0] = 255
            frames[t, :sq_h, :sq_w, 1] = 0
            frames[t, :sq_h, :sq_w, 2] = 0

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
        description="Autoregressive rollout for C-JEPA (PHYRE)."
    )
    parser.add_argument(
        "--video_names", type=str, nargs="*", default=None,
        help="Explicit list of video filenames (keys in the slot pickle).",
    )
    parser.add_argument(
        "--video_range", type=int, nargs=2, metavar=("START", "END"), default=None,
        help="Inclusive numeric id range, e.g. --video_range 0 100.",
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
        "--cjepa_ckpt", type=str,
        default="/cs/data/people/hnam16/phyre_cjepa_mask1_final_predictor.ckpt",
        help="Path to the C-JEPA predictor checkpoint.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="probing/outputs_savi_phyre/rollout",
        help="Directory to save rollout MP4s.",
    )
    parser.add_argument(
        "--cell_size", type=int, default=128,
        help="Pixel size of each decoded frame (square).",
    )
    parser.add_argument(
        "--fps", type=int, default=4,
        help="Frames per second for the output MP4.",
    )
    parser.add_argument(
        "--num_input_frames", type=int, default=5,
        help="Number of real input frames (context).",
    )
    parser.add_argument(
        "--num_pred_frames", type=int, default=6,
        help="Number of frames to predict autoregressively.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu).",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    configs = get_default_config()

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

    # --- 4. Load C-JEPA predictor ---
    print(f"[4] Loading C-JEPA predictor: {args.cjepa_ckpt} ...")
    # Parse num_mask_slots from checkpoint name
    num_mask = args.cjepa_ckpt.split("/")[-1].split("_")[2]
    num_mask_slots = int(num_mask[-1])
    predictor = load_cjepa_predictor(
        args.cjepa_ckpt, num_mask_slots, configs, device=device,
    )
    print(f"    Predictor loaded (history={predictor.history_frames}, "
          f"pred={predictor.pred_frames}).")

    # --- 5. Rollout + decode + save ---
    model_name = args.cjepa_ckpt.split("/")[-1].split(".")[0]
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[5] Running rollout ({args.num_input_frames} input → "
          f"{args.num_pred_frames} predicted) ...")

    for i, vname in enumerate(video_names):
        slots_np = slot_data[vname]  # (T, S, D)

        if slots_np.shape[0] < args.num_input_frames:
            print(f"  [SKIP] {vname}: only {slots_np.shape[0]} frames "
                  f"(need {args.num_input_frames})")
            continue

        # Autoregressive rollout
        all_slots = autoregressive_rollout(
            predictor, slots_np,
            num_input_frames=args.num_input_frames,
            num_pred_frames=args.num_pred_frames,
            device=device,
        )  # (11, S, D)

        # Decode through SAVi
        frames = decode_slots_to_frames(
            savi_model, all_slots, device=device, cell_size=args.cell_size,
        )  # (11, cs, cs, 3)

        # Add blue/red indicator
        frames = add_indicator(frames, args.num_input_frames)

        # Save MP4
        label = _video_label(vname)
        fname = f"rollout_{model_name}_{label}.mp4"
        out_path = os.path.join(output_dir, fname)
        imageio.mimsave(out_path, frames, fps=args.fps)

        if (i + 1) % 50 == 0 or (i + 1) == len(video_names):
            print(f"    Saved {i + 1}/{len(video_names)} videos")

    print(f"Done! Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
