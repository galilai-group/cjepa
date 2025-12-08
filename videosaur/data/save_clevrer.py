import os
import json
import random
import webdataset as wds
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np

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

def make_shards(input_dir, split, out_dir, maxcount=32):
    split_dir = os.path.join(input_dir, split)
    video_files = sorted(glob(os.path.join(split_dir, "*", "*.mp4")))

    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, f"clevrer-{split}-%06d.tar")

    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        for idx, path in enumerate(tqdm(video_files)):
            sample_key = f"{idx:06d}"
            sample = {"__key__": sample_key}

            video = load_mp4_as_numpy(path)
            if video is None:
                print(f"[WARN] Skipping empty video: {path}")
                continue

            sample = {"__key__": sample_key}
            sample["video.npy"] = video.astype(np.uint8)
            # Optional: store metadata such as folder name
            meta = {"original_path": path}
            sample["meta.json"] = json.dumps(meta).encode("utf-8")

            sink.write(sample)

            # if sample_key == "000150" or sample_key ==150:
            #     break

    print(f"[OK] Wrote {len(video_files)} samples into shards at {out_dir}")

# Example:
make_shards("../../data", "train", "../../data/clevrer_wds/train", maxcount=64)

# make_shards("../../../data/clevrer", "validation", "../../../data/clevrer_wds/validation", maxcount=64)
# make_shards("../../../data/clevrer", "test", "../../../data/clevrer_wds/test", maxcount=64)

