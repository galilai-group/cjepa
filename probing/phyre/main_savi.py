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
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so ``src.*`` imports work.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.phyre.mask_config import get_mask, list_masks
from probing.phyre.probing_config_savi import get_default_config
from probing.utils import (
    forward_with_attention,
    load_cjepa_predictor,
    prepare_masked_input,
    prepare_slot_window,
    save_attention_csv,
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

    video_filename = str(int(video_filename.split('.')[0].split('_')[-1])) + '_pixels.mp4'

    if video_filename in data['train']:
        return np.array(data['train'][video_filename])
    if video_filename in data['val']:
        return np.array(data['val'][video_filename])

    raise KeyError(
        f"Video '{video_filename}' not found in slot pickle. "
        f"Top-level keys: {list(data.keys())[:10]}"
    )


# ---------------------------------------------------------------------------
# Load SAVi model (for decoder-only reconstruction from slots)
# ---------------------------------------------------------------------------

def load_savi_model(ckpt_path: str, params_path: str, device: str = "cuda"):
    """
    Build and load a StoSAVi model from checkpoint + params file.
    Only the decoder is needed (to reconstruct images from slots).
    """
    from src.third_party.slotformer.base_slots.models import build_model

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
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model, params


# ---------------------------------------------------------------------------
# Decode slots → reconstruction images
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_slots_to_recon(
    savi_model,
    slots_np: np.ndarray,
    device: str = "cuda",
    cell_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode pre-extracted slots through SAVi's decoder to get reconstructions.

    Parameters
    ----------
    slots_np : (T, num_slots, slot_dim) float array — pre-extracted slots

    Returns
    -------
    recon_combined : (T, cell_size, cell_size, 3) uint8
    slot_recons    : (T, num_slots, cell_size, cell_size, 3) uint8
    """
    T, S, D = slots_np.shape
    slots_tensor = torch.from_numpy(slots_np).float().to(device)  # (T, S, D)

    # model.decode expects (B, num_slots, slot_size) → treats T as batch
    recon_combined, recons, masks, _ = savi_model.decode(slots_tensor)
    # recon_combined: (T, 3, H, W) in [-1, 1]
    # recons:         (T, S, 3, H, W) in [-1, 1]
    # masks:          (T, S, 1, H, W) in [0, 1]

    # Combined reconstruction
    rc = (recon_combined * 0.5 + 0.5).clamp(0, 1)  # → [0, 1]
    rc_resized = F.interpolate(
        rc, size=(cell_size, cell_size), mode="bilinear", align_corners=False
    )
    recon_combined_np = (
        rc_resized.permute(0, 2, 3, 1).cpu().numpy() * 255
    ).astype(np.uint8)  # (T, cs, cs, 3)

    # Per-slot: recons * masks
    slot_vis = recons * masks  # (T, S, 3, H, W)
    slot_vis = (slot_vis * 0.5 + 0.5).clamp(0, 1)
    H, W = slot_vis.shape[-2:]
    flat = slot_vis.reshape(T * S, 3, H, W)
    flat_resized = F.interpolate(
        flat, size=(cell_size, cell_size), mode="bilinear", align_corners=False
    )
    flat_resized = flat_resized.reshape(T, S, 3, cell_size, cell_size)
    slot_recons_np = (
        flat_resized.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
    ).astype(np.uint8)  # (T, S, cs, cs, 3)

    return recon_combined_np, slot_recons_np


# ---------------------------------------------------------------------------
# Attention table GIF
# ---------------------------------------------------------------------------

def save_attention_table_gifs(
    raw_dict: dict[str, np.ndarray],
    norm_dict: dict[str, np.ndarray],
    recon_combined: np.ndarray,
    slot_recons: np.ndarray,
    mask: torch.Tensor,
    mask_name: str,
    video_name: str,
    start_frame: int,
    num_slots: int,
    num_timesteps: int,
    output_dir: str,
    cell_size: int = 128,
    fps: int = 4,
):
    """
    Save one GIF per masked token as a table:
      [Recon | Slot0 | Slot1 | ... | SlotN]

    On each slot cell, raw (R) and normalized (N) attention values are
    overlaid as text.  Positions where normalization excluded the slot
    show ``N: -``.
    """
    gif_dir = os.path.join(
        output_dir, video_name, mask_name, str(start_frame), "gif"
    )
    os.makedirs(gif_dir, exist_ok=True)

    # Determine which (t, s) positions should show "-" for normalized.
    visible_mask_ts = mask.T.numpy().astype(bool)  # (T, S)
    slot_match = re.match(r"^mask_slot([0-7])$", mask_name)
    exclude_slot: int | None = int(slot_match.group(1)) if slot_match else None
    norm_valid_ts = visible_mask_ts.copy()
    if exclude_slot is not None:
        norm_valid_ts[:, exclude_slot] = False

    # Header labels
    labels = ["Recon"] + [f"Slot {s}" for s in range(num_slots)]
    header_h = 20
    total_cols = 1 + num_slots
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
            cells: list[np.ndarray] = []

            # Col 0: reconstruction combined
            cells.append(recon_combined[t])

            # Col 1..S: per-slot recon
            for s in range(num_slots):
                cells.append(slot_recons[t, s])

            row = np.concatenate(cells, axis=1)

            frame_h = header_h + cell_size
            canvas = Image.new("RGB", (table_w, frame_h), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            for col_idx, label in enumerate(labels):
                x_center = col_idx * cell_size + cell_size // 2
                bbox = draw.textbbox((0, 0), label, font=font_bold)
                tw = bbox[2] - bbox[0]
                draw.text(
                    (x_center - tw // 2, 2), label,
                    fill=(0, 0, 0), font=font_bold,
                )

            canvas.paste(Image.fromarray(row), (0, header_h))

            # Overlay attention text on slot cells
            canvas_arr = np.array(canvas)
            for s in range(num_slots):
                x_left = (1 + s) * cell_size
                text_y = header_h + cell_size - 28
                y0, y1 = text_y - 1, header_h + cell_size
                x0, x1 = x_left, x_left + cell_size
                canvas_arr[y0:y1, x0:x1] = (
                    canvas_arr[y0:y1, x0:x1].astype(np.uint16) * 2 // 5
                ).astype(np.uint8)

            canvas = Image.fromarray(canvas_arr)
            draw = ImageDraw.Draw(canvas)

            for s in range(num_slots):
                raw_val = raw_map[t, s]
                x_left = (1 + s) * cell_size
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
            f"{video_name}_{mask_name}_f{start_frame}"
            f"_{key}.gif"
        )
        path = os.path.join(gif_dir, fname)
        imageio.mimsave(path, gif_frames, fps=fps, loop=0)
        print(f"  Saved GIF: {path}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Attention Probing for Causal JEPA (SAVi)"
    )
    parser.add_argument(
        "--video_name", type=str,
        default="video_000158.mp4",
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
        "--savi_ckpt", type=str,
        default="/cs/data/people/hnam16/savi_phyre.pth",
        help="Path to the SAVi (StoSAVi) checkpoint (.pth) — decoder used for slot reconstruction",
    )
    parser.add_argument(
        "--savi_params", type=str,
        default="src/third_party/slotformer/base_slots/configs/savi_phyre_params-fold0.py",
        help="Path to the SlotFormerParams .py config file",
    )
    parser.add_argument(
        "--cjepa_ckpt", type=str,
        default="/cs/data/people/hnam16/phyre_cjepa_mask1_final_predictor.ckpt",
        help="Path to the C-JEPA checkpoint (_object.ckpt)",
    )
    parser.add_argument(
        "--start_frames", type=int, nargs="+", default=[0],
        help="Start frame indices for each probing window",
    )
    parser.add_argument(
        "--output_dir", type=str, default="probing/outputs_savi_phyre",
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
    parser.add_argument("--num_slots", type=int, default=8)
    parser.add_argument(
        "--window_size", type=int, default=6,
        help="Number of timesteps in the probing window",
    )
    args = parser.parse_args()
    configs = get_default_config()
    model_name = args.cjepa_ckpt.split("/")[-1].split(".")[0]
    args.output_dir = os.path.join(args.output_dir, model_name)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    video_name = args.video_name

    # --- 1. Load binary mask ---
    print(f"\n[1] Loading mask '{args.mask_name}' ...")
    mask = get_mask(args.mask_name)
    print(f"    Mask shape: {mask.shape}, masked positions: {(~mask).sum().item()}")

    # --- 2. Load slots ---
    print(f"\n[2] Loading slots from: {args.slot_pkl} ...")
    all_slots = load_slots_for_video(args.slot_pkl, video_name)
    print(f"    Slots shape: {all_slots.shape}")

    # --- 3. Load SAVi model (decoder only, for slot→image reconstruction) ---
    print(f"\n[3] Loading SAVi model ...")
    savi_model, savi_params = load_savi_model(
        args.savi_ckpt, args.savi_params, device=device
    )
    print(f"    SAVi loaded (decoder will be used for slot reconstruction).")

    # --- 4. Load C-JEPA predictor ---
    print(f"\n[4] Loading C-JEPA predictor from: {args.cjepa_ckpt} ...")
    num_mask = args.cjepa_ckpt.split("/")[-1].split("_")[2]
    num_mask_slots = int(num_mask[-1])
    assert num_mask_slots in [0, 1, 2, 3, 4, 5, 6, 7], (
        f"Unexpected num_mask_slots parsed from checkpoint name: {num_mask_slots}"
    )
    predictor = load_cjepa_predictor(
        args.cjepa_ckpt, num_mask_slots, configs, device=device
    )
    print(f"    Predictor loaded. mask_token shape: {predictor.mask_token.shape}")
    print(f"    time_pos_embed shape: {predictor.time_pos_embed.shape}")

    # --- 5. Process each start frame ---
    print(f"\n{'='*60}")
    print(f"Processing start_frames={args.start_frames}")
    print(f"{'='*60}")

    for start_frame in args.start_frames:
        print(f"\n  --- Window starting at frame {start_frame} ---")

        # 5a. Prepare slot window
        # Pass timestep=0 so window starts exactly at start_frame.
        slots_window, frame_indices = prepare_slot_window(
            all_slots, start_frame, 0,
            num_slots=args.num_slots,
            window_size=args.window_size,
            frameskip=args.frameskip,
        )
        print(f"  Slot window shape: {slots_window.shape}")
        print(f"  Frame indices: {frame_indices}")

        # 5b. Prepare masked input
        x_flat = prepare_masked_input(
            slots_window, mask, predictor, device=device
        )
        print(f"  Input shape (flat): {x_flat.shape}")

        # 5c. Forward + attention
        raw_dict, norm_dict = forward_with_attention(
            x_flat, predictor, mask,
            num_slots=args.num_slots,
            num_timesteps=args.window_size,
            mask_name=args.mask_name,
            layer_idx=args.layer_idx,
        )
        print(f"  Extracted attention for {len(norm_dict)} masked tokens")

        # 5d. Save CSVs
        print(f"\n  Saving CSVs ...")
        save_attention_csv(
            raw_dict, norm_dict, video_name, args.mask_name,
            start_frame, 0, args.output_dir,
        )

        # 5e. Decode slots → reconstruction images
        print(f"\n  Decoding slots to reconstruction images ...")
        # slots_window is (1, W, S, D) — squeeze batch dim for decode
        window_slots_np = slots_window[0].numpy()  # (W, S, D)
        recon_combined, slot_recons = decode_slots_to_recon(
            savi_model, window_slots_np, device=device,
        )

        # 5f. Save attention table GIFs
        print(f"\n  Saving attention table GIFs ...")
        save_attention_table_gifs(
            raw_dict, norm_dict,
            recon_combined, slot_recons,
            mask, args.mask_name,
            video_name, start_frame,
            args.num_slots, args.window_size,
            args.output_dir,
        )

    print(f"\n{'='*60}")
    print(f"Done! Outputs saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
