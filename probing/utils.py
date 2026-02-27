"""
Shared utilities for probing scripts (main_videosaur.py, main_savi.py).

Functions extracted here are identical or near-identical across the two
probing entry-points; keeping them in one place makes bug-fixes and
feature additions easier.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from einops import rearrange

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so ``src.*`` imports work.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cjepa_predictor import MaskedSlotPredictor


# ---------------------------------------------------------------------------
# Video reading helpers
# ---------------------------------------------------------------------------

def read_video_frames(video_path: str) -> torch.Tensor:
    """
    Read all frames from *video_path*.

    Returns
    -------
    frames : torch.Tensor  (T, C, H, W) float32 in [0, 1]
    """
    try:
        from torchcodec.decoders import VideoDecoder
        decoder = VideoDecoder(video_path)
        # VideoDecoder returns (T, C, H, W) uint8
        frames = decoder.get_frames_in_range(start=0, stop=len(decoder)).data
        frames = frames.float() / 255.0
        return frames  # (T, C, H, W)
    except Exception:
        pass

    try:
        from torchvision.io import read_video as tv_read_video
        video, _, _ = tv_read_video(video_path)  # (T, H, W, C) uint8
        frames = video.float().permute(0, 3, 1, 2) / 255.0
        return frames
    except Exception:
        pass

    # Last resort: OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    cap.release()
    if not frame_list:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    return torch.stack(frame_list)  # (T, C, H, W)


# ---------------------------------------------------------------------------
# Load annotation (CLEVRER format)
# ---------------------------------------------------------------------------

def load_collision_frames(annotation_path: str) -> list[int]:
    """
    Read a CLEVRER annotation JSON and return sorted list of collision frame_ids.
    """
    with open(annotation_path) as f:
        ann = json.load(f)
    collisions = ann.get("collision", [])
    return sorted(set(c["frame_id"] for c in collisions))


# ---------------------------------------------------------------------------
# Load C-JEPA predictor from checkpoint
# ---------------------------------------------------------------------------

def load_cjepa_predictor(ckpt_path: str, num_mask_slots: int, configs, device: str = "cuda"):
    """
    Load a C-JEPA checkpoint saved with ``torch.save(pl_module, path)``.
    Returns the ``MaskedSlotPredictor`` sub-module.
    """
    spt_module = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    predictor = MaskedSlotPredictor(
        num_slots=7,  # S: number of slots
        slot_dim=128,
        history_frames=6,  # T: history length
        pred_frames=10,  # number of future frames to predict
        num_masked_slots=num_mask_slots,  # M: number of slots to mask
        seed=0,  # for reproducible masking
        depth=configs.predictor.depth,
        heads=configs.predictor.heads,
        dim_head=configs.predictor.dim_head,
        mlp_dim=configs.predictor.mlp_dim,
        dropout=configs.predictor.dropout,
    )
    missing, unexpected = predictor.load_state_dict(spt_module, strict=False)
    if missing:
        print(f"Missing keys in predictor: {missing}")
    if unexpected:
        print(f"Unexpected keys in predictor: {unexpected}")

    predictor.eval()
    predictor.to(device)
    return predictor


# ---------------------------------------------------------------------------
# Prepare slot input around a collision frame
# ---------------------------------------------------------------------------

def prepare_slot_window(
    all_slots: np.ndarray,
    collision_frame: int,
    timestep: int,
    num_slots: int = 7,
    window_size: int = 16,
    frameskip: int = 1,
) -> tuple[torch.Tensor, list[int]]:
    """
    Build a (1, window_size, num_slots, slot_dim) tensor centred around
    the requested collision frame.

    Parameters
    ----------
    all_slots : (T_total, num_slots, slot_dim)
    collision_frame : raw video frame index of the collision
    timestep : which time step position within the window the collision
               should land on (0-indexed, between 0 and window_size-1).
    frameskip : sub-sampling factor (default 1 = every frame).

    Returns
    -------
    slots_tensor : (1, window_size, num_slots, slot_dim) float32
    frame_indices : list[int] of length window_size — raw frame indices used
    """
    T_total = all_slots.shape[0]

    # Starting frame so that the collision lands at position `timestep`.
    # The window samples every `frameskip`-th frame:
    #   [start, start+fs, start+2*fs, ..., start+(W-1)*fs]
    # with collision_frame at position `timestep`.
    start_frame = collision_frame - timestep * frameskip

    frame_indices = []
    slot_frames = []
    for i in range(window_size):
        frame_idx = start_frame + i * frameskip
        if 0 <= frame_idx < T_total:
            slot_frames.append(all_slots[frame_idx])
            frame_indices.append(frame_idx)
        else:
            # Pad with zeros if out of bounds
            slot_frames.append(np.zeros_like(all_slots[0]))
            frame_indices.append(-1)  # sentinel for out-of-bounds

    slots_np = np.stack(slot_frames, axis=0)  # (window_size, num_slots, slot_dim)
    slots_tensor = torch.from_numpy(slots_np).float().unsqueeze(0)  # (1, W, S, D)
    return slots_tensor, frame_indices


# ---------------------------------------------------------------------------
# Apply mask + prepare input for the transformer
# ---------------------------------------------------------------------------

def prepare_masked_input(
    slots: torch.Tensor,
    mask: torch.Tensor,
    predictor,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Given raw slots (1, T, S, D) and a binary mask (S, T) where
    1=visible and 0=masked, construct the full input tensor for the
    C-JEPA transformer replicating MaskedSlotPredictor logic.

    Parameters
    ----------
    slots : (1, T, S, D) — raw slot features
    mask  : (S, T) bool — True = visible, False = masked
    predictor : MaskedSlotPredictor with learned parameters

    Returns
    -------
    x_flat : (1, T*S, D) — ready for transformer.forward()
    """
    B, T, S, D = slots.shape
    assert mask.shape == (S, T), f"Mask shape {mask.shape} != expected ({S}, {T})"

    slots = slots.to(device)
    mask = mask.to(device)  # (S, T) — True=visible

    # Transpose mask to (T, S) for easier indexing
    mask_ts = mask.T  # (T, S)

    # --- Build components ---
    # Time positional embedding: predictor.time_pos_embed is (1, T_train, 1, D).
    # We may need more timesteps than the training config had, so we interpolate.
    trained_T = predictor.time_pos_embed.shape[1]
    if T <= trained_T:
        time_pe = predictor.time_pos_embed[:, :T, :, :]  # (1, T, 1, D)
    else:
        raise ValueError(f"Requested window_size {T} exceeds predictor's trained time embedding length {trained_T}")
    time_pe = time_pe.expand(B, T, S, D)  # (1, T, S, D)

    # Anchor queries from t=0
    anchors = slots[:, 0, :, :]  # (1, S, D)
    anchor_queries = predictor.id_projector(anchors)  # (1, S, D)
    anchor_grid = anchor_queries.unsqueeze(1).expand(B, T, S, D)  # (1, T, S, D)

    # Mask token grid
    mask_token_grid = predictor.mask_token.expand(B, T, S, D)  # (1, T, S, D)

    # --- Compose ---
    # Default (masked) input: mask_token + time_pe + anchor_query
    query_input = mask_token_grid + time_pe + anchor_grid  # (1, T, S, D)

    # Visible input: real slot + time_pe
    visible_input = slots + time_pe  # (1, T, S, D)

    # Final: pick visible where mask=True, query where mask=False
    # mask_ts: (T, S) -> (1, T, S, 1) for broadcasting
    mask_4d = mask_ts.unsqueeze(0).unsqueeze(-1).expand_as(slots)  # (1, T, S, D)
    final_input = torch.where(mask_4d, visible_input, query_input)

    # Flatten to (B, T*S, D)
    x_flat = rearrange(final_input, "b t s d -> b (t s) d")
    return x_flat


# ---------------------------------------------------------------------------
# Forward + attention extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_with_attention(
    x_flat: torch.Tensor,
    predictor,
    mask: torch.Tensor,
    num_slots: int,
    num_timesteps: int,
    mask_name: str = "",
    layer_idx: int = -1,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Forward *x_flat* through the predictor's transformer and extract
    attention for every **masked** token.

    Returns
    -------
    raw_dict  : {f"token{s}_{t}": np.ndarray (T, S)} — raw attention.
    norm_dict : {f"token{s}_{t}": np.ndarray (T, S)} — normalized attention
                for the masked token at slot *s*, timestep *t*.
    """
    out_flat, attn_list = predictor.transformer(x_flat, return_attention=True)
    # attn_list[layer_idx] : (B, T*S, T*S)
    attn = attn_list[layer_idx]  # (1, T*S, T*S)

    # Reshape to (1, T, S, T, S)
    attn_5d = rearrange(
        attn,
        "b (tq sq) (tk sk) -> b tq sq tk sk",
        tq=num_timesteps, sq=num_slots,
        tk=num_timesteps, sk=num_slots,
    )

    # mask: (S, T) — True=visible, False=masked
    # visible_mask_ts: (T, S) — True where token is visible (unmasked)
    visible_mask_ts = mask.T.cpu().numpy()  # (T, S)

    # For mask_slot0 .. mask_slot6, also exclude ALL tokens of the masked
    # slot (including the visible anchor at t=0) during normalization.
    slot_mask_match = re.match(r"^mask_slot([0-6])$", mask_name)
    exclude_slot_idx: int | None = None
    if slot_mask_match:
        exclude_slot_idx = int(slot_mask_match.group(1))

    # Build normalization mask: tokens used for min-max statistics
    norm_mask_ts = visible_mask_ts.copy()  # (T, S)
    if exclude_slot_idx is not None:
        norm_mask_ts[:, exclude_slot_idx] = False

    # Iterate over masked positions
    raw_dict = {}
    norm_dict = {}
    for s in range(num_slots):
        for t in range(num_timesteps):
            if not mask[s, t]:  # position is masked
                # Extract what this token attends to: (T, S)
                raw = attn_5d[0, t, s, :, :]  # (T, S)
                raw_np = raw.cpu().numpy()
                raw_dict[f"token{s}_{t}"] = raw_np.copy()

                # Min-max normalize using tokens in norm_mask_ts
                norm_vals = raw_np[norm_mask_ts]
                has_nan = np.isnan(norm_vals).any()
                if has_nan or len(norm_vals) == 0 or norm_vals.max() == norm_vals.min():
                    norm = np.full_like(raw_np, -1.0)
                else:
                    rmin, rmax = norm_vals.min(), norm_vals.max()
                    norm = (raw_np - rmin) / (rmax - rmin)

                # Set masked token positions to 0.5 (will be gray anyway)
                norm[~visible_mask_ts] = 0.5

                norm_dict[f"token{s}_{t}"] = norm

    return raw_dict, norm_dict


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_attention_csv(
    raw_dict: dict[str, np.ndarray],
    norm_dict: dict[str, np.ndarray],
    video_name: str,
    mask_name: str,
    collision_frame: int,
    timestep: int,
    output_dir: str,
):
    """Save one CSV per masked token for both raw and normalized attention.

    Directory: {output_dir}/{video_name}/{mask_name}/{collision_frame}/csv/raw/
               {output_dir}/{video_name}/{mask_name}/{collision_frame}/csv/normalized/
    """
    base_csv_dir = os.path.join(
        output_dir, video_name, mask_name, str(collision_frame), "csv"
    )

    for subdir, attn_dict in [("raw", raw_dict), ("normalized", norm_dict)]:
        csv_dir = os.path.join(base_csv_dir, subdir)
        os.makedirs(csv_dir, exist_ok=True)
        for key, attn_map in attn_dict.items():
            fname = (
                f"{video_name}_{mask_name}_f{collision_frame}"
                f"_at{timestep}_{key}.csv"
            )
            path = os.path.join(csv_dir, fname)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                T, S = attn_map.shape
                writer.writerow([f"slot{s}" for s in range(S)])
                for t_row in range(T):
                    writer.writerow([f"{v:.6f}" for v in attn_map[t_row]])
            print(f"  Saved CSV: {path}")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def get_video_frames_for_indices(
    all_video_frames: torch.Tensor, frame_indices: list[int]
) -> list[np.ndarray]:
    """
    Extract specific frames from the full video tensor.
    Returns list of (H, W, 3) uint8 numpy arrays.
    Missing indices (sentinel -1) produce black frames.
    """
    C, H, W = (
        all_video_frames.shape[1],
        all_video_frames.shape[2],
        all_video_frames.shape[3],
    )
    frames = []
    for idx in frame_indices:
        if 0 <= idx < all_video_frames.shape[0]:
            frame = (
                all_video_frames[idx].permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)
        else:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def save_resized_rgb_video(
    all_video_frames: torch.Tensor,
    output_path: str,
    vis_size: int = 224,
    fps: int = 25,
):
    """
    Save the original video resized to (vis_size, vis_size) as an MP4.

    Parameters
    ----------
    all_video_frames : (T, C, H, W) float32 in [0, 1]
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for t in range(all_video_frames.shape[0]):
            frame = all_video_frames[t].permute(1, 2, 0).cpu().numpy()  # (H,W,3)
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.resize(frame, (vis_size, vis_size))
            writer.append_data(frame)
    print(f"  Saved resized RGB video: {output_path}")
