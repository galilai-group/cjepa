#!/usr/bin/env python3
"""
Attention Probing for Causal JEPA — SAVi variant.

Given a video, pre-extracted slots, annotation file, and model checkpoints,
this script:
  1. Reads the video frames with torchcodec (fallback to torchvision).
  2. Loads slots from a pickle file for the target video.
  3. Reads collision frames from the CLEVRER annotation JSON.
  4. Loads the SAVi (StoSAVi) model and C-JEPA predictor.
  5. For each collision frame, constructs a 7×16 slot matrix centred on the
     collision, applies the requested binary mask, inserts learned mask tokens,
     adds time positional encoding + id_projector anchoring, and forwards
     through the C-JEPA transformer to obtain per-token attention maps.
  6. Outputs:
       (a) CSVs of raw & normalized attention per masked token
           (saved under raw/ and normalized/ sub-directories).
       (b) Table-style GIFs per masked token: columns are
           [Original | Recon | Slot0 | … | SlotN] with per-slot
           SAVi reconstructions and overlaid raw (R) / normalized (N)
           attention values. Positions excluded from normalization
           show ``N: -``.
       (c) A resized RGB video and a slot-colored overlay video.

Usage:
    python probing/main_savi.py \
        --video_path /path/to/video_08001.mp4 \
        --mask_name mask_slot0 \
        --slot_pkl /path/to/slots.pkl \
        --annotation_path /path/to/annotation_08001.json \
        --savi_ckpt /path/to/savi.ckpt \
        --cjepa_ckpt /path/to/cjepa_object.ckpt \
        --timestep 4
"""

import argparse
import importlib
import os
import pickle
import re
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so ``src.*`` imports work.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.mask_config import get_mask, list_masks
from probing.probing_config_savi import get_default_config
from probing.utils import (
    forward_with_attention,
    get_video_frames_for_indices,
    load_cjepa_predictor,
    load_collision_frames,
    prepare_masked_input,
    prepare_slot_window,
    read_video_frames,
    save_attention_csv,
    save_resized_rgb_video,
)


# ---------------------------------------------------------------------------
# Load slots from pickle
# ---------------------------------------------------------------------------

def load_slots_for_video(slot_pkl_path: str, video_filename: str) -> np.ndarray:
    """
    Return slot array of shape (T, num_slots, slot_dim) for *video_filename*.
    """
    with open(slot_pkl_path, "rb") as f:
        data = pickle.load(f)

    if video_filename in data['train']:
        return np.array(data['train'][video_filename])
    if video_filename in data['val']:
        return np.array(data['val'][video_filename])
    if video_filename in data['test']:
        return np.array(data['test'][video_filename])

    raise KeyError(
        f"Video '{video_filename}' not found in slot pickle. "
        f"Top-level keys: {list(data.keys())[:10]}"
    )


# ---------------------------------------------------------------------------
# Load SAVi model
# ---------------------------------------------------------------------------

def load_savi_model(ckpt_path: str, params_path: str, device: str = "cuda"):
    """
    Build and load a StoSAVi model from checkpoint + params file.

    Parameters
    ----------
    ckpt_path : path to .ckpt file (state_dict with 'state_dict' key)
    params_path : path to the SlotFormerParams .py config file

    Returns
    -------
    model : StoSAVi in eval mode on *device*
    params : SlotFormerParams instance
    """
    from src.third_party.slotformer.base_slots.models import build_model

    # Import the params module dynamically
    if params_path.endswith('.py'):
        params_module_path = params_path[:-3]
    else:
        params_module_path = params_path
    sys.path.append(os.path.dirname(params_module_path))
    params_module = importlib.import_module(os.path.basename(params_module_path))
    params = params_module.SlotFormerParams()

    model = build_model(params)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    unexpected, missing = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.testing = False  # we need decode outputs, not just slots
    model.to(device)
    return model, params


# ---------------------------------------------------------------------------
# SAVi mask helpers
# ---------------------------------------------------------------------------

def _preprocess_for_savi(
    frames: torch.Tensor,
    resolution: tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """
    Preprocess frames for SAVi input: resize to *resolution* and normalize
    from [0, 1] to [-1, 1].

    Parameters
    ----------
    frames : (T, C, H, W) float32 in [0, 1]

    Returns
    -------
    preprocessed : (T, C, H', W') float32 in [-1, 1]
    """
    resized = F.interpolate(
        frames, size=resolution, mode="bilinear", align_corners=False
    )  # (T, C, H', W')
    return resized * 2.0 - 1.0


@torch.no_grad()
def get_savi_masks_for_frames(
    savi_model,
    frames: torch.Tensor,
    resolution: tuple[int, int] = (64, 64),
    device: str = "cuda",
    vis_size: int = 224,
) -> torch.Tensor:
    """
    Run SAVi on multiple frames to get per-slot hard masks.

    Parameters
    ----------
    frames : (T, C, H, W) float32 in [0, 1]

    Returns
    -------
    hard_masks : (T, num_slots, vis_size, vis_size) float32 (0 or 1)
    """
    T = frames.shape[0]
    preprocessed = _preprocess_for_savi(frames, resolution)  # (T, C, H', W')

    # SAVi expects (B, T, C, H, W). Feed all frames as one clip.
    inp = preprocessed.unsqueeze(0).to(device)  # (1, T, C, H', W')
    data_dict = {"img": inp}
    out_dict = savi_model(data_dict)

    # post_masks: (1, T, num_slots, 1, H', W') — softmax masks
    post_masks = out_dict["post_masks"]  # (1, T, S, 1, H', W')
    masks = post_masks[0, :, :, 0, :, :]  # (T, S, H', W')

    # Resize to vis_size
    S = masks.shape[1]
    masks_flat = masks.reshape(T * S, 1, resolution[0], resolution[1])
    masks_resized = F.interpolate(
        masks_flat, size=(vis_size, vis_size), mode="bilinear", align_corners=False
    )  # (T*S, 1, vis_size, vis_size)
    masks_resized = masks_resized.reshape(T, S, vis_size, vis_size)

    # Hard masks: argmax over slots
    ind = torch.argmax(masks_resized, dim=1, keepdim=True)  # (T, 1, H, W)
    hard_masks = torch.zeros_like(masks_resized)
    hard_masks.scatter_(1, ind, 1)

    return hard_masks.cpu()  # (T, num_slots, vis_size, vis_size)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_savi_recon_for_frames(
    savi_model,
    frames: torch.Tensor,
    resolution: tuple[int, int] = (64, 64),
    device: str = "cuda",
    cell_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run SAVi on *frames* and return reconstruction outputs as uint8 images.

    Parameters
    ----------
    frames : (T, C, H, W) float32 in [0, 1]

    Returns
    -------
    recon_combined : (T, cell_size, cell_size, 3) uint8
    slot_recons    : (T, num_slots, cell_size, cell_size, 3) uint8
        Per-slot reconstruction masked by softmax masks (recons * masks).
    """
    T = frames.shape[0]
    preprocessed = _preprocess_for_savi(frames, resolution)  # (T, C, H', W')
    inp = preprocessed.unsqueeze(0).to(device)  # (1, T, C, H', W')
    out_dict = savi_model({"img": inp})

    # recon_combined: (1, T, 3, H', W') in [-1, 1]
    rc = out_dict["post_recon_combined"]  # (1, T, 3, H', W')
    rc = (rc * 0.5 + 0.5).clamp(0, 1)  # → [0, 1]
    rc = rc[0]  # (T, 3, H', W')
    rc_resized = F.interpolate(
        rc, size=(cell_size, cell_size), mode="bilinear", align_corners=False
    )  # (T, 3, cs, cs)
    recon_combined = (
        rc_resized.permute(0, 2, 3, 1).cpu().numpy() * 255
    ).astype(np.uint8)  # (T, cs, cs, 3)

    # per-slot: recons * masks
    recons = out_dict["post_recons"]  # (1, T, S, 3, H', W') in [-1, 1]
    masks = out_dict["post_masks"]    # (1, T, S, 1, H', W') in [0, 1]
    slot_vis = recons * masks  # (1, T, S, 3, H', W')
    slot_vis = (slot_vis * 0.5 + 0.5).clamp(0, 1)
    slot_vis = slot_vis[0]  # (T, S, 3, H', W')
    S = slot_vis.shape[1]
    # Resize: flatten T*S, interpolate, reshape
    flat = slot_vis.reshape(T * S, 3, resolution[0], resolution[1])
    flat_resized = F.interpolate(
        flat, size=(cell_size, cell_size), mode="bilinear", align_corners=False
    )  # (T*S, 3, cs, cs)
    flat_resized = flat_resized.reshape(T, S, 3, cell_size, cell_size)
    slot_recons = (
        flat_resized.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
    ).astype(np.uint8)  # (T, S, cs, cs, 3)

    return recon_combined, slot_recons


def save_attention_table_gifs(
    raw_dict: dict[str, np.ndarray],
    norm_dict: dict[str, np.ndarray],
    rgb_frames: list[np.ndarray],
    recon_combined: np.ndarray,
    slot_recons: np.ndarray,
    mask: torch.Tensor,
    mask_name: str,
    video_name: str,
    collision_frame: int,
    timestep: int,
    num_slots: int,
    num_timesteps: int,
    output_dir: str,
    cell_size: int = 128,
    fps: int = 4,
):
    """
    Save one GIF per masked token as a table:
      [Original | Recon | Slot0 | Slot1 | ... | SlotN]

    On each slot cell, raw (R) and normalized (N) attention values are
    overlaid as text.  Positions where normalization excluded the slot
    show ``N: -``.
    """
    gif_dir = os.path.join(
        output_dir, video_name, mask_name, str(collision_frame), "gif"
    )
    os.makedirs(gif_dir, exist_ok=True)

    # Determine which (t, s) positions should show "-" for normalized.
    visible_mask_ts = mask.T.numpy().astype(bool)  # (T, S)
    slot_match = re.match(r"^mask_slot([0-6])$", mask_name)
    exclude_slot: int | None = int(slot_match.group(1)) if slot_match else None
    # norm_valid_ts[t, s] = True  iff normalised value is meaningful
    norm_valid_ts = visible_mask_ts.copy()
    if exclude_slot is not None:
        norm_valid_ts[:, exclude_slot] = False

    # Header labels
    labels = ["Original", "Recon"] + [f"Slot {s}" for s in range(num_slots)]
    header_h = 20
    total_cols = 2 + num_slots
    table_w = cell_size * total_cols

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
        )
        font_bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12
        )
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_bold = font

    for key in raw_dict:
        raw_map = raw_dict[key]    # (T, S)
        norm_map = norm_dict[key]  # (T, S)
        gif_frames = []

        for t in range(num_timesteps):
            # --- build the row of images ---
            cells: list[np.ndarray] = []

            # Col 0: original RGB
            orig = cv2.resize(rgb_frames[t], (cell_size, cell_size))
            cells.append(orig)

            # Col 1: reconstruction combined
            cells.append(recon_combined[t])  # already (cs, cs, 3) uint8

            # Col 2..2+S: per-slot recon
            for s in range(num_slots):
                cells.append(slot_recons[t, s])  # (cs, cs, 3) uint8

            row = np.concatenate(cells, axis=1)  # (cs, table_w, 3)

            # --- compose with PIL for text ---
            frame_h = header_h + cell_size
            canvas = Image.new("RGB", (table_w, frame_h), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            # Header labels
            for col_idx, label in enumerate(labels):
                x_center = col_idx * cell_size + cell_size // 2
                bbox = draw.textbbox((0, 0), label, font=font_bold)
                tw = bbox[2] - bbox[0]
                draw.text(
                    (x_center - tw // 2, 2), label,
                    fill=(0, 0, 0), font=font_bold,
                )

            # Paste the image row below the header
            canvas.paste(Image.fromarray(row), (0, header_h))

            # --- overlay attention text on slot cells (numpy-accelerated) ---
            canvas_arr = np.array(canvas)
            for s in range(num_slots):
                raw_val = raw_map[t, s]
                x_left = (2 + s) * cell_size
                text_y = header_h + cell_size - 28

                # Semi-transparent dark bar behind text (vectorised)
                y0, y1 = text_y - 1, header_h + cell_size
                x0, x1 = x_left, x_left + cell_size
                canvas_arr[y0:y1, x0:x1] = (
                    canvas_arr[y0:y1, x0:x1].astype(np.uint16) * 2 // 5
                ).astype(np.uint8)

            canvas = Image.fromarray(canvas_arr)
            draw = ImageDraw.Draw(canvas)

            for s in range(num_slots):
                raw_val = raw_map[t, s]
                x_left = (2 + s) * cell_size
                text_y = header_h + cell_size - 28

                r_str = f"R:{raw_val:.2f}"
                if norm_valid_ts[t, s]:
                    n_str = f"N:{norm_map[t, s]:.2f}"
                else:
                    n_str = "N: -"

                draw.text(
                    (x_left + 3, text_y), r_str,
                    fill=(255, 255, 255), font=font,
                )
                draw.text(
                    (x_left + 3, text_y + 13), n_str,
                    fill=(255, 255, 255), font=font,
                )

            gif_frames.append(np.array(canvas))

        fname = (
            f"{video_name}_{mask_name}_f{collision_frame}"
            f"_at{timestep}_{key}.gif"
        )
        path = os.path.join(gif_dir, fname)
        imageio.mimsave(path, gif_frames, fps=fps, loop=0)
        print(f"  Saved GIF: {path}")


# ---------------------------------------------------------------------------
# Slot colour map (distinct per-slot colours, NOT green→red)
# ---------------------------------------------------------------------------

def _get_slot_color_map(num_slots: int) -> list[tuple[int, int, int]]:
    """
    Generate perceptually distinct colours for *num_slots* slots.
    """
    _TAB_COLORS = [
        (31, 119, 180),   # blue
        (255, 127, 14),   # orange
        (44, 160, 44),    # green
        (214, 39, 40),    # red
        (148, 103, 189),  # purple
        (140, 86, 75),    # brown
        (227, 119, 194),  # pink
        (127, 127, 127),  # gray
        (188, 189, 34),   # olive
        (23, 190, 207),   # cyan
    ]
    if num_slots <= len(_TAB_COLORS):
        return _TAB_COLORS[:num_slots]
    cmap = []
    for i in range(num_slots):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (((c >> 0) & 1) << (7 - j))
            g = g | (((c >> 1) & 1) << (7 - j))
            b = b | (((c >> 2) & 1) << (7 - j))
            c = c >> 3
        cmap.append((r, g, b))
    return cmap


# ---------------------------------------------------------------------------
# Slot-index reference image
# ---------------------------------------------------------------------------

def save_slot_colored_video(
    savi_model,
    all_video_frames: torch.Tensor,
    output_path: str,
    resolution: tuple[int, int] = (64, 64),
    num_slots: int = 7,
    vis_size: int = 224,
    fps: int = 25,
    alpha: float = 0.5,
    device: str = "cuda",
):
    """
    Save a video with distinct-colored slot overlays + black borders
    using SAVi masks.

    Parameters
    ----------
    all_video_frames : (T, C, H, W) float32 in [0, 1]
    """
    cmap = _get_slot_color_map(num_slots)

    T = all_video_frames.shape[0]
    chunk_size = 32
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with imageio.get_writer(output_path, fps=fps) as writer:
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk = all_video_frames[start:end]

            hard_masks = get_savi_masks_for_frames(
                savi_model, chunk, resolution, device, vis_size=vis_size,
            )  # (chunk_len, S, vis_size, vis_size)

            for i in range(chunk.shape[0]):
                rgb = chunk[i].permute(1, 2, 0).cpu().numpy()
                rgb = cv2.resize(rgb, (vis_size, vis_size))
                rgb = (rgb * 255).astype(np.float32)

                overlay = rgb.copy()
                masks_i = hard_masks[i].numpy()

                for s in range(num_slots):
                    region = masks_i[s].astype(bool)
                    if not region.any():
                        continue
                    color = np.array(cmap[s], dtype=np.float32)
                    overlay[region] = (
                        rgb[region] * (1 - alpha) + color * alpha
                    )

                overlay = np.ascontiguousarray(np.clip(overlay, 0, 255).astype(np.uint8))

                for s in range(num_slots):
                    region = masks_i[s].astype(np.uint8)
                    if not region.any():
                        continue
                    contours, _ = cv2.findContours(
                        region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(
                        overlay, contours, -1, (0, 0, 0), thickness=2
                    )

                writer.append_data(overlay)

    print(f"  Saved slot-colored video: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Attention Probing for Causal JEPA (SAVi)"
    )
    parser.add_argument(
        "--video_path", type=str,
        default="/cs/data/people/hnam16/data/clevrer_for_savi/videos/train/"
                "video_07000-08000/video_07039.mp4",
        help="Path to the video file",
    )
    parser.add_argument(
        "--mask_name", type=str, default="mask_slot5",
        help=f"Mask name from mask_config. Available: {list_masks()}",
    )
    parser.add_argument(
        "--slot_pkl", type=str,
        default="/cs/data/people/hnam16/data/modified_extraction/clevrer_savi_reproduced.pkl",
        help="Path to the slot pickle file",
    )
    parser.add_argument(
        "--annotation_path", type=str,
        default="/cs/data/people/hnam16/data/clevrer_for_savi/annotations/train/"
                "annotation_07000-08000/annotation_07039.json",
        help="Path to the CLEVRER annotation JSON file",
    )
    parser.add_argument(
        "--savi_ckpt", type=str,
        default="/cs/data/people/hnam16/savi_pretrained/clevrer_savi_reproduce_20260109_102641_LR0.0001/savi/epoch/model_8.pth",
        help="Path to the SAVi (StoSAVi) checkpoint (.pth)",
    )
    parser.add_argument(
        "--savi_params", type=str,
        default="src/third_party/slotformer/base_slots/configs/"
                "stosavi_clevrer_params.py",
        help="Path to the SlotFormerParams .py config file",
    )
    parser.add_argument(
        "--cjepa_ckpt", type=str,
        default="/cs/data/people/hnam16/clevrer_savi_4_epoch_30_object.ckpt",
        help="Path to the C-JEPA checkpoint (_object.ckpt)",
    )
    parser.add_argument(
        "--timestep", type=int, default=4,
        help="Which timestep position (0-15) the collision frame should occupy",
    )
    parser.add_argument(
        "--output_dir", type=str, default="probing/outputs_savi",
        help="Output directory for CSVs and images",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--layer_idx", type=int, default=-1,
        help="Transformer layer index for attention extraction (-1 = last)",
    )
    parser.add_argument(
        "--frameskip", type=int, default=1,
        help="Frame sub-sampling factor (default 1)",
    )
    parser.add_argument("--num_slots", type=int, default=7)
    parser.add_argument(
        "--window_size", type=int, default=16,
        help="Number of timesteps in the probing window",
    )
    args = parser.parse_args()
    configs = get_default_config()
    model_name = args.cjepa_ckpt.split("/")[-1].split(".")[0]
    args.output_dir = os.path.join(args.output_dir, model_name)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    assert 0 <= args.timestep < args.window_size, (
        f"timestep must be in [0, {args.window_size - 1}], got {args.timestep}"
    )

    video_basename = os.path.basename(args.video_path)
    video_name = os.path.splitext(video_basename)[0]

    # --- 1. Load binary mask ---
    print(f"\n[1] Loading mask '{args.mask_name}' ...")
    mask = get_mask(args.mask_name)
    print(f"    Mask shape: {mask.shape}, masked positions: {(~mask).sum().item()}")

    # --- 2. Read video ---
    print(f"\n[2] Reading video: {args.video_path} ...")
    all_video_frames = read_video_frames(args.video_path)  # (T, C, H, W)
    print(f"    Video shape: {all_video_frames.shape}")

    # --- 3. Load slots ---
    print(f"\n[3] Loading slots from: {args.slot_pkl} ...")
    all_slots = load_slots_for_video(args.slot_pkl, video_basename)
    print(f"    Slots shape: {all_slots.shape}")

    # --- 4. Load collision frames ---
    print(f"\n[4] Reading annotation: {args.annotation_path} ...")
    collision_frames = load_collision_frames(args.annotation_path)
    print(f"    Collision frames: {collision_frames}")
    if not collision_frames:
        print("    WARNING: No collisions found in annotation. Exiting.")
        return

    # --- 5. Load SAVi model ---
    print(f"\n[5] Loading SAVi model ...")
    savi_model, savi_params = load_savi_model(
        args.savi_ckpt, args.savi_params, device=device
    )
    savi_resolution = tuple(savi_params.resolution)
    print(
        f"    SAVi loaded. Resolution: {savi_resolution}, "
        f"num_slots: {savi_params.slot_dict['num_slots']}"
    )

    # --- 6. Load C-JEPA predictor ---
    print(f"\n[6] Loading C-JEPA predictor from: {args.cjepa_ckpt} ...")
    num_mask_slots = int(args.cjepa_ckpt.split("/")[-1].split("_")[2])
    assert num_mask_slots in [0, 1, 2, 3, 4, 5, 6, 7], (
        f"Unexpected num_mask_slots parsed from checkpoint name: {num_mask_slots}"
    )
    predictor = load_cjepa_predictor(
        args.cjepa_ckpt, num_mask_slots, configs, device=device
    )
    print(f"    Predictor loaded. mask_token shape: {predictor.mask_token.shape}")
    print(f"    time_pos_embed shape: {predictor.time_pos_embed.shape}")

    # --- 7. Save resized RGB video + slot-colored video ---
    ref_dir = os.path.join(args.output_dir, video_name)

    print(f"\n[7a] Saving resized RGB video ...")
    rgb_video_path = os.path.join(ref_dir, f"{video_name}_resized_rgb.mp4")
    save_resized_rgb_video(all_video_frames, rgb_video_path)

    print(f"\n[7b] Saving slot-colored overlay video ...")
    slot_video_path = os.path.join(ref_dir, f"{video_name}_slot_colored.mp4")
    save_slot_colored_video(
        savi_model, all_video_frames, slot_video_path,
        resolution=savi_resolution,
        num_slots=args.num_slots, device=device,
    )

    # --- 8. Process each collision frame ---
    for col_frame in collision_frames:
        print(f"\n{'='*60}")
        print(f"Processing collision at frame {col_frame}, timestep={args.timestep}")
        print(f"{'='*60}")

        # 8a. Prepare slot window
        slots_window, frame_indices = prepare_slot_window(
            all_slots, col_frame, args.timestep,
            num_slots=args.num_slots,
            window_size=args.window_size,
            frameskip=args.frameskip,
        )
        print(f"  Slot window shape: {slots_window.shape}")
        print(f"  Frame indices: {frame_indices}")

        # 8b. Prepare masked input
        x_flat = prepare_masked_input(
            slots_window, mask, predictor, device=device
        )
        print(f"  Input shape (flat): {x_flat.shape}")

        # 8c. Forward + attention
        raw_dict, norm_dict = forward_with_attention(
            x_flat, predictor, mask,
            num_slots=args.num_slots,
            num_timesteps=args.window_size,
            mask_name=args.mask_name,
            layer_idx=args.layer_idx,
        )
        print(f"  Extracted attention for {len(norm_dict)} masked tokens")

        # 8d. Save CSVs
        print(f"\n  Saving CSVs ...")
        save_attention_csv(
            raw_dict, norm_dict, video_name, args.mask_name,
            col_frame, args.timestep, args.output_dir,
        )

        # 8e. Get RGB frames and SAVi reconstructions for the window
        print(f"\n  Computing SAVi reconstructions for window frames ...")
        rgb_frames = get_video_frames_for_indices(
            all_video_frames, frame_indices
        )

        window_frames_tensor = []
        for idx in frame_indices:
            if 0 <= idx < all_video_frames.shape[0]:
                window_frames_tensor.append(all_video_frames[idx])
            else:
                window_frames_tensor.append(
                    torch.zeros_like(all_video_frames[0])
                )
        window_frames_tensor = torch.stack(window_frames_tensor)

        recon_combined, slot_recons = get_savi_recon_for_frames(
            savi_model, window_frames_tensor,
            savi_resolution, device=device,
        )

        # 8f. Save attention table GIFs
        print(f"\n  Saving attention table GIFs ...")
        save_attention_table_gifs(
            raw_dict, norm_dict,
            rgb_frames, recon_combined, slot_recons,
            mask, args.mask_name,
            video_name, col_frame, args.timestep,
            args.num_slots, args.window_size,
            args.output_dir,
        )

    print(f"\n{'='*60}")
    print(f"Done! Outputs saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
