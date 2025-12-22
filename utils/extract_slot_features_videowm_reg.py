"""
Extract slot features from CLEVRER videos using VideoWM_reg and save to disk for fast loading.
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

from visualization.utils import read_video
from visualization.model import load_videoreg_model_from_checkpoint

# Output directory for features
FEATURE_DIR = "/cs/data/people/hnam16/data/clevrer_feature"
Path(FEATURE_DIR).mkdir(parents=True, exist_ok=True)

VIDEO_PATH = "/cs/data/people/hnam16/data/clevrer/videos/video_*.mp4"
ANNOT_PATH = "/cs/data/people/hnam16/data/clevrer_annotation"
NUM_FRAMESKIP = 5


def extract_and_save_slot_features(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_model = load_videoreg_model_from_checkpoint(cfg)
    world_model = world_model.to(device)
    world_model.eval()

    video_files = sorted(glob.glob(VIDEO_PATH))
    transform = transforms.Compose([transforms.Resize((196, 196))])

    ckpt_name = Path(cfg.checkpoint_path).stem.split('.ckpt')[0]
    out_dir = Path(FEATURE_DIR) / ckpt_name
    os.makedirs(out_dir, exist_ok=True)

    def process_video(video_file):
        video_num = Path(video_file).stem.split('_')[-1]
        out_path = out_dir / f"slot_features_{video_num}.npz"
        # 이 함수는 반드시 없는 파일만 들어오게 됨

        annot_file = Path(ANNOT_PATH) / f"annotation_{video_num}.json"
        if not annot_file.exists():
            # Try subfolders
            found = False
            for subdir in Path(ANNOT_PATH).iterdir():
                if subdir.is_dir():
                    candidate = subdir / f"annotation_{video_num}.json"
                    if candidate.exists():
                        annot_file = candidate
                        found = True
                        break
            if not found:
                print(f"No annotation for {video_file}, skipping.")
                return
        with open(annot_file, 'r') as f:
            annotations = json.load(f)
        collision_frames = [item['frame_id'] for item in annotations.get('collision', [])]

        video_frames, video_indices = read_video(video_file, NUM_FRAMESKIP, return_idx=True)
        video_frames = torch.stack([transform(frame) for frame in video_frames], dim=0).to(device)

        feat_collection = []
        history_size = cfg.dinowm.history_size

        # Initial encoding for the first history_size frames
        pixels = video_frames[0:0 + history_size].unsqueeze(0)
        frame = {"pixels": pixels.to(device)}
        with torch.no_grad():
            x = world_model.encode(frame, target='embed', pixels_key="pixels")
            feat_embed = x["embed"][0, :history_size, :, :].cpu()  # (num_slots, dim)
            for t in range(feat_embed.shape[0]):
                feat_collection.append(feat_embed[t, :, :])  # (num_slots, dim)

        for i in range(len(video_frames) - history_size):
            pixels = video_frames[i:i + history_size].unsqueeze(0)
            frame = {"pixels": pixels.to(device)}
            with torch.no_grad():
                x = world_model.encode(frame, target='embed', pixels_key="pixels")
                input_embed = x["embed"][:, :history_size, :, :]
                feat_embed = world_model.predict(input_embed)[0, -1, :, :].cpu()   # (num_slots, dim)
                feat_collection.append(feat_embed)
        if not feat_collection:
            print(f"No slots for {video_file}, skipping.")
            return
        feats = torch.stack(feat_collection, dim=0)  # (num_frames, num_slots, dim)
        video_indices_ = video_indices[:]
        labels = torch.zeros(len(video_indices_))
        for c_frame in collision_frames:
            for i, v_idx in enumerate(video_indices_):
                if abs(v_idx - c_frame) <= cfg.collision_tolerance * NUM_FRAMESKIP:
                    labels[i] = 1.0
        np.savez_compressed(
            out_path,
            slots=feats.numpy(),
            labels=labels.numpy(),
            video_indices=np.array(video_indices_),
            video_num=video_num,
            history_size=history_size,
            num_frameskip=NUM_FRAMESKIP,
            collision_tolerance=cfg.collision_tolerance,
            slot_shape=feats.shape,
            slot_dim=feats.shape[-1],
            num_slots=feats.shape[1],
        )
        with open(out_dir / f"slot_features_{video_num}_config.json", "w") as f:
            json.dump({
                "video_file": str(video_file),
                "annot_file": str(annot_file),
                "history_size": history_size,
                "num_frameskip": NUM_FRAMESKIP,
                "collision_tolerance": cfg.collision_tolerance,
                "slot_shape": list(feats.shape),
                "slot_dim": feats.shape[-1],
                "num_slots": feats.shape[1],
            }, f, indent=2)

    files_to_process = []
    for video_file in video_files:
        video_num = Path(video_file).stem.split('_')[-1]
        out_path = out_dir / f"slot_features_{video_num}.npz"
        if not out_path.exists():
            files_to_process.append(video_file)
        else:
            print(f"[SKIP] {out_path} already exists.")

    if files_to_process:
        Parallel(n_jobs=4, prefer="processes")(
            delayed(process_video)(vf) for vf in tqdm(files_to_process, desc="Extracting slot features (parallel)")
        )
    else:
        print("All slot feature files already exist. Nothing to do.")


@hydra.main(version_base=None, config_path="../configs", config_name="config_test")
def main(cfg):
    extract_and_save_slot_features(cfg)

if __name__ == "__main__":
    main()
