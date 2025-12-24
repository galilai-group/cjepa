"""
Extract slot features from CLEVRER videos and save to disk for fast loading.
Saves: slot_features, labels, video_indices, video_num, and config info per video.
"""
import os
from joblib import Parallel, delayed
from pathlib import Path
import torch
import numpy as np
import json
from tqdm import tqdm
import hydra
import torchvision.transforms.v2 as transforms
import glob
import pickle as pkl

from visualization.utils import read_video
from videosaur.videosaur import  models

# Output directory for features
FEATURE_DIR = "/cs/data/people/hnam16/data/clevrer_feature"
Path(FEATURE_DIR).mkdir(parents=True, exist_ok=True)

DINO_PATCH_SIZE = 14
TRAIN_PATH = "/cs/data/people/hnam16/data/clevrer/videos/video_*.mp4"
VAL_PATH = "/cs/data/people/hnam16/data/clevrer/validation/*/video_*.mp4"
TEST_PATH = "/cs/data/people/hnam16/data/clevrer/test/*/video_*.mp4"



def extract_and_save_slot_features(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_model = load_causal_model_from_checkpoint(cfg)
    world_model = world_model.to(device)
    world_model.eval()

    train_file = sorted(glob.glob(TRAIN_PATH))
    val_file = sorted(glob.glob(VAL_PATH))
    test_file = sorted(glob.glob(TEST_PATH))

    # Match train pipeline: Resize + ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((196, 196)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ckpt_name = Path(cfg.checkpoint_path).stem
    out_dir = Path(FEATURE_DIR) / ckpt_name
    os.makedirs(out_dir, exist_ok=True)

    feat_dict = {
        "train": {},
        "val": {},
        "test": {}
    }

    def process_video(video_file, split):
        video_name = Path(video_file).name
        video_frames, video_indices = read_video(video_file, num_frameskip=1, return_idx=True)
        video_frames = torch.stack([transform(frame) for frame in video_frames], dim=0).to(device)


        # Initial encoding for the first history_size frames
        pixels = video_frames.unsqueeze(0)
        frame = {"pixels": pixels.to(device)}
        with torch.no_grad():
            x = world_model.encode(frame, target='embed', pixels_key="pixels")
            slot_embed = x["embed"][0, :, :, :].cpu()  # (frame, num_slots, dim)

        feat_dict[split][video_name] = slot_embed.numpy(dtype=np.float32) 

    for split, files in zip(["train", "val", "test"], [train_file, val_file, test_file]):
        print(f"Processing {split} set with {len(files)} videos...")
        Parallel(n_jobs=8, prefer="processes")(
            delayed(process_video)(vf, split) for vf in tqdm(files, desc="Extracting slot features (parallel)")
        )
    
    # save features to disk as pkl file
    out_file = out_dir / f"SLOT128_{ckpt_name}.pkl"
    with open(out_file, "wb") as f:
        pkl.dump(feat_dict[split], f)
    print(f"Saved {split} features to {out_file}")

@hydra.main(version_base=None, config_path="../configs", config_name="config_test_causal")
def main(cfg):
    extract_and_save_slot_features(cfg)

if __name__ == "__main__":
    main()
