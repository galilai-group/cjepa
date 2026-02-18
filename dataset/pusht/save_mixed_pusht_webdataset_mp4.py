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
    all_video_files = []
    for split_dir in input_dir:
        video_files = sorted(glob(os.path.join(split_dir, "*.mp4")))
        all_video_files.extend(video_files)
    
    print(f"Found total {len(all_video_files)} video files.")
    # shuffle the combined list of video files
    random.shuffle(all_video_files)

    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, f"pusht-{split}-%06d.tar")

    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        written = 0
        for idx, path in enumerate(tqdm(all_video_files)):
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


train_mixing = [
    "path/to/generated/train", # This folder should contain video_00000.mp4 - video_09999.mp4 from generated data.
    "path/to/expert/train" ]  # This folder should only contain video_10000.mp4 - video_18684.mp4 from pusht_expert_train
make_shards(train_mixing, "train", "your/path/to/data/pusht_mixed_ind_noise_wds_mp4/train", maxcount=512)

validation_mixing = [
    "path/to/generated/validation"
] # This folder should contain video_10000.mp4 - video_10999.mp4 from generated data. 
make_shards(validation_mixing, "val", "your/path/to/data/pusht_mixed_ind_noise_wds_mp4/validation", maxcount=128)
