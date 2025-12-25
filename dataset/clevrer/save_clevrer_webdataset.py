import os
import json
import random
import webdataset as wds
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
from torchcodec.decoders import VideoDecoder

def load_mp4_as_numpy(path):
    """Read mp4 using cv2 → return numpy array (F, H, W, C)."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        # raise ValueError(f"No frames in video: {path}")
        return None

    return np.stack(frames, axis=0) 


def load_mp4_as_numpy_torchcodec(path):
    """Read mp4 using torchcodec.VideoDecoder → return numpy array (F, H, W, C).

    This matches the output of load_mp4_as_numpy (uint8 RGB frames) but uses
    VideoDecoder which can be faster and avoids intermediate disk reads.
    """
    try:
        decoder = VideoDecoder(path)
    except Exception:
        return None

    # decoder supports indexing to get all frames: [F, C, H, W]
    try:
        frames_t = decoder[:]
    except Exception:
        return None

    if frames_t is None or len(frames_t) == 0:
        return None

    # Move to CPU and convert to numpy with channels-last [F, H, W, C]
    frames_np = frames_t.permute(0, 2, 3, 1).cpu().numpy()

    # Ensure uint8 in [0,255]
    if np.issubdtype(frames_np.dtype, np.floating):
        # assume floats are in [0,1]
        if frames_np.max() <= 1.01:
            frames_np = (frames_np * 255.0).round().astype(np.uint8)
        else:
            frames_np = frames_np.round().astype(np.uint8)
    else:
        frames_np = frames_np.astype(np.uint8)

    return frames_np

def make_shards(input_dir, split, out_dir, maxcount=32):
    split_dir = os.path.join(input_dir, split)
    video_files = sorted(glob(os.path.join(split_dir, "*.mp4")))

    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, f"clevrer-{split}-%06d.tar")

    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        for idx, path in enumerate(tqdm(video_files)):
            sample_key = f"{idx:06d}"
            sample = {"__key__": sample_key}

            video = load_mp4_as_numpy_torchcodec(path)
            if idx==0:
                video_orig = load_mp4_as_numpy(path)
                assert video.all() == video_orig.all(), f" mismatch between cv2 and torchcodec reading"
            if video is None:
                print(f"[WARN] Skipping empty video: {path}")
                continue

            sample = {"__key__": sample_key}
            sample["video.npy"] = video.astype(np.uint8)
            # Optional: store metadata such as folder name
            meta = {"original_path": path}
            sample["meta.json"] = json.dumps(meta).encode("utf-8")

            sink.write(sample)



    print(f"[OK] Wrote {len(video_files)} samples into shards at {out_dir}")

# Example:
make_shards("/cs/data/people/hnam16/data/clevrer", "train_videos", "/cs/data/people/hnam16/data/clevrer_wds/train", maxcount=64)
make_shards("/cs/data/people/hnam16/data/clevrer", "val_videos", "/cs/data/people/hnam16/data/clevrer_wds/validation", maxcount=64)
# make_shards("../../../data/clevrer", "test", "../../../data/clevrer_wds/test", maxcount=64)

