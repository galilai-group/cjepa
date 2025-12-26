import os
import json
import random
import webdataset as wds
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
from torchcodec.decoders import VideoDecoder



def make_shards(input_dir, split, out_dir, maxcount=512):
    split_dir = input_dir
    video_files = sorted(glob(os.path.join(split_dir, "*.mp4")))

    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, f"pusht-{split}-%06d.tar")

    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        written = 0
        for idx, path in enumerate(tqdm(video_files)):
            sample_key = f"{idx:06d}"

            with open(path, "rb") as fh:
                raw = fh.read()

            sample = {"__key__": sample_key}
            # Store raw mp4 bytes so downstream consumers can call dataset.decode('rgb')
            sample["video.mp4"] = raw
            # Optional: store metadata such as folder name
            meta = {"original_path": path}
            sample["meta.json"] = json.dumps(meta).encode("utf-8")

            sink.write(sample)
            written += 1



    print(f"[OK] Wrote {written} samples into shards at {out_dir}")

# Example:
# make_shards("/users/hnam16/scratch/.stable_worldmodel/pusht_expert_train_video/videos", "train", "/users/hnam16/scratch/data/pusht_wds_mp4/train", maxcount=512)
make_shards("/users/hnam16/scratch/.stable_worldmodel/pusht_expert_val_video/videos", "val", "/users/hnam16/scratch/data/pusht_wds_mp4/validation", maxcount=10)


