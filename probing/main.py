#!/usr/bin/env python3
"""
Attention Probing for Causal JEPA.

Given a video, pre-extracted slots, annotation file, and model checkpoints,
this script:
  1. Reads the video frames with torchcodec (fallback to torchvision).
  2. Loads slots from a pickle file for the target video.
  3. Reads collision frames from the CLEVRER annotation JSON.
  4. Loads the videosaur model (for mask visualization) and C-JEPA predictor.
  5. For each collision frame, constructs a 7×16 slot matrix centred on the
     collision, applies the requested binary mask, inserts learned mask tokens,
     adds time positional encoding + id_projector anchoring, and forwards
     through the C-JEPA transformer to obtain per-token attention maps.
  6. Outputs:
       (a) CSVs of normalized attention per masked token.
       (b) MP4 videos of attention overlaid on actual video frames using
           videosaur slot masks (gray=masked slot, green→red=attention).
       (c) A slot-index reference image.

Usage:
    python probing/main.py \
        --video_path /path/to/video_08001.mp4 \
        --mask_name mask_slot0 \
        --slot_pkl /path/to/slots.pkl \
        --annotation_path /path/to/annotation_08001.json \
        --videosaur_ckpt /path/to/videosaur.ckpt \
        --cjepa_ckpt /path/to/cjepa_object.ckpt \
        --timestep 4
"""

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so ``src.*`` imports work.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.mask_config import get_mask, list_masks
from probing.probing_config import get_default_config
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
# Load slots from pickle
# ---------------------------------------------------------------------------

def load_slots_for_video(slot_pkl_path: str, video_filename: str) -> np.ndarray:
    """
    Return slot array of shape (T, num_slots, slot_dim) for *video_filename*.
    The pickle may be ``{split: {filename: array}}`` or just ``{filename: array}``.
    """
    with open(slot_pkl_path, "rb") as f:
        data = pickle.load(f)

    # Direct lookup
    id = str(int(video_filename.split(".")[0].split('_')[-1]))
    id = id + '_pixels.mp4'
    if id in data['train']:
        return np.array(data['train'][id])
    if id in data['val']:
        return np.array(data['val'][id])
    if id in data['test']:
        return np.array(data['test'][id])

    raise KeyError(
        f"Video '{video_filename}' not found in slot pickle. "
        f"Top-level keys: {list(data.keys())[:10]}"
    )


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
# Load videosaur model
# ---------------------------------------------------------------------------

def load_videosaur_model(ckpt_path: str, config_path: str, n_slots: int = 7, device: str = "cuda"):
    """
    Build and load a videosaur model from checkpoint + config.
    Returns (model, config).
    """
    from src.third_party.videosaur.videosaur import configuration, models
    config = configuration.load_config(config_path)
    model = models.build(config.model, config.optimizer)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.initializer.n_slots = n_slots
    model.eval()
    model.to(device)
    return model, config


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
    if missing != []:
        print(f"Missing keys in predictor: {missing}")
    if unexpected != []:
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
    layer_idx: int = -1,
) -> dict[str, np.ndarray]:
    """
    Forward *x_flat* through the predictor's transformer and extract
    attention for every **masked** token.

    Returns
    -------
    attn_dict : {f"token{s}_{t}": np.ndarray (T, S)} — normalized attention
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
    # Iterate over masked positions
    attn_dict = {}
    for s in range(num_slots):
        for t in range(num_timesteps):
            if not mask[s, t]:  # position is masked
                # Extract what this token attends to: (1, T, S)
                raw = attn_5d[0, t, s, :, :]  # (T, S)
                raw_np = raw.cpu().numpy()

                # Min-max normalize: map to [0, 1], NaN -> -1
                has_nan = np.isnan(raw_np).any()
                if has_nan or raw_np.max() == raw_np.min():
                    norm = np.full_like(raw_np, -1.0)
                else:
                    rmin, rmax = raw_np.min(), raw_np.max()
                    norm = (raw_np - rmin) / (rmax - rmin)

                attn_dict[f"token{s}_{t}"] = norm

    return attn_dict


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_attention_csv(
    attn_dict: dict[str, np.ndarray],
    video_name: str,
    mask_name: str,
    collision_frame: int,
    timestep: int,
    output_dir: str,
):
    """Save one CSV per masked token.

    Directory: {output_dir}/{video_name}/{mask_name}/{collision_frame}/csv/
    """
    csv_dir = os.path.join(output_dir, video_name, mask_name, str(collision_frame), "csv")
    os.makedirs(csv_dir, exist_ok=True)
    for key, attn_map in attn_dict.items():
        fname = f"{video_name}_{mask_name}_f{collision_frame}_at{timestep}_{key}.csv"
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

def get_video_frames_for_indices(all_video_frames: torch.Tensor, frame_indices: list[int]) -> list[np.ndarray]:
    """
    Extract specific frames from the full video tensor.
    Returns list of (H, W, 3) uint8 numpy arrays.
    Missing indices (sentinel -1) produce black frames.
    """
    C, H, W = all_video_frames.shape[1], all_video_frames.shape[2], all_video_frames.shape[3]
    frames = []
    for idx in frame_indices:
        if 0 <= idx < all_video_frames.shape[0]:
            frame = (all_video_frames[idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        else:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def get_videosaur_masks_for_frames(
    videosaur_model,
    frames: torch.Tensor,
    config,
    device: str = "cuda",
    vis_size: int = 224,
) -> torch.Tensor:
    """
    Run videosaur on multiple frames to get per-slot hard masks.

    Parameters
    ----------
    frames : (T, C, H, W) float32 in [0, 1]

    Returns
    -------
    hard_masks : (T, num_slots, vis_size, vis_size) bool
    """
    from src.third_party.videosaur.videosaur.data.transforms import build_inference_transform, Resize
    from omegaconf import OmegaConf
    import torchvision.transforms as tvt

    input_size = 196

    tf_cfg = OmegaConf.create({
        "dataset_type": "video",
        "input_size": input_size,
        "use_movi_normalization": False,
    })
    tfs = build_inference_transform(tf_cfg)

    # Transform expects (C, F, H, W)
    video_cfhw = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
    video_cfhw = tfs(video_cfhw)              # (C, T, H', W') normalised
    video_fchw = video_cfhw.permute(1, 0, 2, 3)  # (T, C, H', W')

    inputs = {"video": video_fchw.unsqueeze(0).to(device)}  # (1, T, C, H', W')

    with torch.no_grad():
        outputs = videosaur_model(inputs)

    # decoder masks: (B, F, n_slots, H*W)
    raw_masks = outputs["decoder"]["masks"]  # (1, T, n_slots, H*W)
    T_out = raw_masks.shape[1]
    n_slots = raw_masks.shape[2]
    hw = raw_masks.shape[-1]
    h = int(np.sqrt(hw))
    masks = raw_masks[0].reshape(T_out, n_slots, h, h)  # (T, n_slots, h, h)

    # Resize to vis_size
    resizer = Resize(vis_size, mode="bilinear")
    masks = resizer(masks)  # (T, n_slots, vis_size, vis_size)

    # Hard masks per frame
    ind = torch.argmax(masks, dim=1, keepdim=True)  # (T, 1, H, W)
    hard_masks = torch.zeros_like(masks)
    hard_masks.scatter_(1, ind, 1)
    return hard_masks.cpu()  # (T, n_slots, vis_size, vis_size)


def create_attention_overlay_frame(
    rgb_frame: np.ndarray,
    slot_masks: np.ndarray,
    attn_per_slot: np.ndarray,
    mask_per_slot: np.ndarray,
    vis_size: int = 224,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create a single overlay frame with per-slot attention colouring.

    For each pixel the colour is determined by the slot it belongs to:
      - **Masked slot** (mask_per_slot[s] == False): gray overlay.
      - **Unmasked slot, attention == -1**: perfect black.
      - **Unmasked slot**: green (attn=0) → red (attn=1) overlay.

    Parameters
    ----------
    rgb_frame     : (H, W, 3) uint8
    slot_masks    : (S, H, W) bool — hard videosaur masks
    attn_per_slot : (S,) float — normalised attention values for this frame
    mask_per_slot : (S,) bool — True = visible in the binary mask, False = masked
    """
    frame = cv2.resize(rgb_frame, (vis_size, vis_size)).astype(np.float32)
    overlay = frame.copy()
    S = slot_masks.shape[0]

    for s in range(S):
        region = slot_masks[s].astype(bool)  # (H, W)
        if not region.any():
            continue

        val = float(attn_per_slot[s])
        is_visible = bool(mask_per_slot[s])

        if not is_visible:
            # Masked slot → gray
            gray = np.array([128, 128, 128], dtype=np.float32)
            overlay[region] = frame[region] * (1 - alpha) + gray * alpha
        elif val == -1.0:
            # Invalid → perfect black
            overlay[region] = 0.0
        else:
            # Green (0) → Red (1)
            r = val * 255.0
            g = (1.0 - val) * 255.0
            color = np.array([r, g, 0.0], dtype=np.float32)
            overlay[region] = frame[region] * (1 - alpha) + color * alpha

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Add black borders around each slot region
    for s in range(S):
        region = slot_masks[s].astype(np.uint8)  # (H, W)
        if not region.any():
            continue
        
        # Find contours and draw black borders
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 0), thickness=2)
    
    return overlay


def save_attention_videos(
    attn_dict: dict[str, np.ndarray],
    rgb_frames: list[np.ndarray],
    slot_masks_all: np.ndarray,
    mask: torch.Tensor,
    video_name: str,
    mask_name: str,
    collision_frame: int,
    timestep: int,
    num_slots: int,
    num_timesteps: int,
    output_dir: str,
    fps: int = 4,
):
    """
    Save one MP4 video per masked token.

    Each video is a 16-frame sequence with per-slot attention overlaid
    on the RGB frames using videosaur slot masks.

    Directory: {output_dir}/{video_name}/{mask_name}/{collision_frame}/video/
    """
    vid_dir = os.path.join(output_dir, video_name, mask_name, str(collision_frame), "video")
    os.makedirs(vid_dir, exist_ok=True)

    mask_ts = mask.T.numpy()  # (T, S) — True=visible

    for key, attn_map in attn_dict.items():
        # attn_map: (T, S)
        frames_out = []
        for t in range(num_timesteps):
            overlay = create_attention_overlay_frame(
                rgb_frame=rgb_frames[t],
                slot_masks=slot_masks_all[t],  # (S, H, W)
                attn_per_slot=attn_map[t],      # (S,)
                mask_per_slot=mask_ts[t],        # (S,)
            )
            frames_out.append(overlay)

        fname = f"{video_name}_{mask_name}_f{collision_frame}_at{timestep}_{key}.mp4"
        path = os.path.join(vid_dir, fname)
        with imageio.get_writer(path, fps=fps) as writer:
            for frame in frames_out:
                writer.append_data(frame)
        print(f"  Saved video: {path}")


# ---------------------------------------------------------------------------
# Slot-index reference image
# ---------------------------------------------------------------------------

def create_slot_reference_image(
    videosaur_model,
    reference_frame: torch.Tensor,
    config,
    output_path: str,
    num_slots: int = 7,
    device: str = "cuda",
):
    """
    Create a reference image showing videosaur slot segmentations with
    slot indices drawn on each segment.

    Parameters
    ----------
    reference_frame : (C, H, W) float32 in [0, 1]
    """
    from src.third_party.videosaur.videosaur.visualizations import (
        draw_segmentation_masks_on_image,
        color_map,
    )
    import torchvision.transforms as tvt

    # Get slot masks (pass single frame as batch of 1)
    hard_masks_all = get_videosaur_masks_for_frames(
        videosaur_model, reference_frame.unsqueeze(0), config, device
    )  # (1, num_slots, H, W)
    hard_masks = hard_masks_all[0]  # (num_slots, H, W)

    # Prepare image for overlay (uint8, 3×H×W)
    img_resized = tvt.Resize((224, 224))(reference_frame)
    img_uint8 = (img_resized * 255).to(torch.uint8)  # (3, 224, 224)

    cmap = color_map(num_slots)
    overlay = draw_segmentation_masks_on_image(
        img_uint8, hard_masks.bool(), colors=cmap, alpha=0.5
    )  # (3, 224, 224) uint8
    overlay_np = overlay.permute(1, 2, 0).cpu().numpy()  # (224, 224, 3)

    # Draw slot index text on each slot's centroid
    pil_img = Image.fromarray(overlay_np)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for s in range(num_slots):
        slot_mask = hard_masks[s].numpy()  # (H, W)
        ys, xs = np.where(slot_mask > 0.5)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        text = str(s)
        # Draw with outline for visibility
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                draw.text((cx + dx, cy + dy), text, fill=(0, 0, 0), font=font)
        draw.text((cx, cy), text, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pil_img.save(output_path, quality=95)
    print(f"  Saved slot reference: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Attention Probing for Causal JEPA")
    parser.add_argument("--video_path", type=str, default='/cs/data/people/hnam16/data/clevrer_for_savi/videos/train/video_07000-08000/video_07039.mp4',
                        help="Path to the video file")
    parser.add_argument("--mask_name", type=str, default="mask_slot5",
                        help=f"Mask name from mask_config. Available: {list_masks()}")
    parser.add_argument("--slot_pkl", type=str, default='/cs/data/people/hnam16/clevrer_videosaur_slots.pkl',
                        help="Path to the slot pickle file")
    parser.add_argument("--annotation_path", type=str, default='/cs/data/people/hnam16/data/clevrer_for_savi/annotations/train/annotation_07000-08000/annotation_07039.json',
                        help="Path to the CLEVRER annotation JSON file")
    parser.add_argument("--videosaur_ckpt", type=str, default='/cs/data/people/hnam16/clevrer_videosaur_model.ckpt',
                        help="Path to the videosaur checkpoint (.ckpt)")
    parser.add_argument("--cjepa_ckpt", type=str, default='/cs/data/people/hnam16/clevrer_savi_4_epoch_30_object.ckpt',
                        help="Path to the C-JEPA checkpoint (_object.ckpt)")
    parser.add_argument("--timestep", type=int, default=4,
                        help="Which timestep position (0-15) the collision frame should occupy")
    parser.add_argument("--videosaur_config", type=str, default='src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml',
                        help="Path to the videosaur model config YAML. "
                             "Default: src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml")
    parser.add_argument("--output_dir", type=str, default="probing/outputs",
                        help="Output directory for CSVs and images")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--layer_idx", type=int, default=-1,
                        help="Transformer layer index for attention extraction (-1 = last)")
    parser.add_argument("--frameskip", type=int, default=1,
                        help="Frame sub-sampling factor (default 1)")
    parser.add_argument("--num_slots", type=int, default=7)
    parser.add_argument("--window_size", type=int, default=16,
                        help="Number of timesteps in the probing window")
    args = parser.parse_args()
    configs = get_default_config()
    model_name = args.cjepa_ckpt.split("/")[-1].split(".")[0]
    args.output_dir = os.path.join(args.output_dir, model_name)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Validate timestep ---
    assert 0 <= args.timestep < args.window_size, (
        f"timestep must be in [0, {args.window_size - 1}], got {args.timestep}"
    )

    # --- Get video name (without extension) for output filenames ---
    video_basename = os.path.basename(args.video_path)
    video_name = os.path.splitext(video_basename)[0]

    # --- 1. Load binary mask ---
    print(f"\n[1] Loading mask '{args.mask_name}' ...")
    mask = get_mask(args.mask_name)  # (7, 16) bool
    print(f"    Mask shape: {mask.shape}, masked positions: {(~mask).sum().item()}")

    # --- 2. Read video ---
    print(f"\n[2] Reading video: {args.video_path} ...")
    all_video_frames = read_video_frames(args.video_path)  # (T, C, H, W)
    print(f"    Video shape: {all_video_frames.shape}")

    # --- 3. Load slots ---
    print(f"\n[3] Loading slots from: {args.slot_pkl} ...")
    all_slots = load_slots_for_video(args.slot_pkl, video_basename)
    print(f"    Slots shape: {all_slots.shape}")  # (T, num_slots, slot_dim)

    # --- 4. Load collision frames ---
    print(f"\n[4] Reading annotation: {args.annotation_path} ...")
    collision_frames = load_collision_frames(args.annotation_path)
    print(f"    Collision frames: {collision_frames}")
    if not collision_frames:
        print("    WARNING: No collisions found in annotation. Exiting.")
        return

    # --- 5. Load videosaur model ---
    print(f"\n[5] Loading videosaur model ...")
    vs_config_path = args.videosaur_config or os.path.join(
        REPO_ROOT, "src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml"
    )
    videosaur_model, vs_config = load_videosaur_model(
        args.videosaur_ckpt, vs_config_path, n_slots=args.num_slots, device=device
    )
    print("    Videosaur loaded.")

    # --- 6. Load C-JEPA predictor ---
    print(f"\n[6] Loading C-JEPA predictor from: {args.cjepa_ckpt} ...")
    num_mask_slots = int(args.cjepa_ckpt.split("/")[-1].split("_")[2])
    assert num_mask_slots in [0, 1, 2, 3, 4, 5, 6, 7], f"Unexpected num_mask_slots parsed from checkpoint name: {num_mask_slots}"
    predictor = load_cjepa_predictor(args.cjepa_ckpt, num_mask_slots, configs, device=device)
    print(f"    Predictor loaded. mask_token shape: {predictor.mask_token.shape}")
    print(f"    time_pos_embed shape: {predictor.time_pos_embed.shape}")

    # --- 7. Create slot reference image ---
    print(f"\n[7] Creating slot reference image ...")
    # Use the first collision frame as reference
    ref_frame_idx = collision_frames[0]
    if ref_frame_idx < all_video_frames.shape[0]:
        ref_frame = all_video_frames[ref_frame_idx]  # (C, H, W)
    else:
        ref_frame = all_video_frames[0]
    ref_path = os.path.join(args.output_dir, video_name, f"{video_name}_slot_reference.jpeg")
    create_slot_reference_image(
        videosaur_model, ref_frame, vs_config, ref_path,
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
        x_flat = prepare_masked_input(slots_window, mask, predictor, device=device)
        print(f"  Input shape (flat): {x_flat.shape}")

        # 8c. Forward + attention
        attn_dict = forward_with_attention(
            x_flat, predictor, mask,
            num_slots=args.num_slots,
            num_timesteps=args.window_size,
            layer_idx=args.layer_idx,
        )
        print(f"  Extracted attention for {len(attn_dict)} masked tokens")

        # 8d. Save CSVs
        print(f"\n  Saving CSVs ...")
        save_attention_csv(
            attn_dict, video_name, args.mask_name,
            col_frame, args.timestep, args.output_dir,
        )

        # 8e. Get RGB frames and videosaur slot masks for the window
        print(f"\n  Computing videosaur slot masks for window frames ...")
        rgb_frames = get_video_frames_for_indices(all_video_frames, frame_indices)

        # Build a (T, C, H, W) tensor of the window frames for videosaur
        window_frames_tensor = []
        for idx in frame_indices:
            if 0 <= idx < all_video_frames.shape[0]:
                window_frames_tensor.append(all_video_frames[idx])
            else:
                # Black frame placeholder
                window_frames_tensor.append(torch.zeros_like(all_video_frames[0]))
        window_frames_tensor = torch.stack(window_frames_tensor)  # (T, C, H, W)

        slot_masks_all = get_videosaur_masks_for_frames(
            videosaur_model, window_frames_tensor, vs_config, device=device,
        )  # (T, num_slots, 224, 224) bool
        slot_masks_np = slot_masks_all.numpy()  # (T, S, H, W)

        # 8f. Save attention overlay videos
        print(f"\n  Saving attention overlay videos ...")
        save_attention_videos(
            attn_dict, rgb_frames, slot_masks_np, mask,
            video_name, args.mask_name,
            col_frame, args.timestep,
            args.num_slots, args.window_size,
            args.output_dir,
        )

    print(f"\n{'='*60}")
    print(f"Done! Outputs saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
