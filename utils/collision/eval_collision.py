from pathlib import Path
import hydra
import torch
from loguru import logger as logging
import numpy as np
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.v2 as transforms
import glob

from sklearn.decomposition import PCA

import stable_pretraining as spt
import stable_worldmodel as swm
from custom_models.dinowm_causal import CausalWM
from custom_models.cjepa_predictor import MaskedSlotPredictor
from videosaur.videosaur import  models

from visualization.utils import eval_colision, read_video
from visualization.model import load_causal_model_from_checkpoint



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DINO_PATCH_SIZE = 14
VIDEO_PATH = "/cs/data/people/hnam16/data/clevrer/videos/video_08*.mp4"
ANNOT_PATH = "/cs/data/people/hnam16/data/clevrer_annotation/annotation_08000-09000"
NUM_FRAMESKIP=5
TEST = "pred" # "pred" or "gt"


def slot_interaction(cfg):
    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    world_model = load_causal_model_from_checkpoint(cfg)
    world_model = world_model.to(device)

    video_files = sorted(glob.glob(VIDEO_PATH))

    # process and write results as csv 
    section_name = ANNOT_PATH.split('/')[-1].split('_')[-1]
    ckpt_name = Path(cfg.checkpoint_path).stem.split('.ckpt')[0]
    csv_name = f"video_{section_name}_collision_results_{TEST}_{ckpt_name}.csv"
    with open(csv_name, 'w') as f:
        f.write("video_num, num_collision_gt, gt_collision_frames, detected_collision_framees, precision, recall, f1_score, accuracy\n")

    collision_gt = []
    collision_pred = []
    for video_file in video_files:
        video_frames, video_indices = read_video(video_file, NUM_FRAMESKIP, return_idx=True)
        video_num = Path(video_file).stem.split('_')[-1]

        transform = transforms.Compose([transforms.Resize((196, 196))])
        video_frames = torch.stack([transform(frame) for frame in video_frames], dim=0).to(device)

        logging.info(f"Video frames length: {video_frames.shape[0]}")
        logging.info(f"Discard first {cfg.dinowm.history_size} frames for history")

        # Process video frames
        pred_collection = []
        gt_collection = []
        for i in range(len(video_frames) - cfg.dinowm.history_size):
            pixels = video_frames[i:i + cfg.dinowm.history_size].unsqueeze(0)
            frame = { "pixels": pixels.to(device) }
            with torch.no_grad():
                x = world_model.encode(frame, target='embed', pixels_key="pixels")
                input_embedding = x["embed"][:, : cfg.dinowm.history_size, :, :] 
                pred_embedding = world_model.predict(input_embedding, use_inference_function=True)
                # print(pred_embedding.shape, x["embed"].shape)
                
                pred = pred_embedding[0, -1, :, :].cpu().numpy().squeeze()
                pred -= pred.mean(axis=0, keepdims=True) 
                gt = x["embed"][0, -1, :, :].cpu().numpy().squeeze()
                gt -= gt.mean(axis=0, keepdims=True)
                pred_collection.append(pred) # batchsize is always 1
                gt_collection.append(gt) # batchsize is always 1
        logging.info(f"Collection length: {len(pred_collection)}")

        pred_stacked = np.stack(pred_collection, axis=0) # source: (num_frames, num_slots, embedding_dim)
        gt_stacked = np.stack(gt_collection, axis=0) # source: (num_frames, num_slots, embedding_dim)
        video_indices = video_indices[cfg.dinowm.history_size : ]

        annot_file = Path(ANNOT_PATH) / f"annotation_{video_num}.json"

        if TEST == "gt":
            result = eval_colision(
                gt_stacked, 
                video_indices, 
                annot_dir=annot_file
            )
        else:
            result = eval_colision(
                pred_stacked, 
                video_indices, 
                annot_dir=annot_file
            )
        with open(csv_name, 'a') as f:
            line = (
                f"{video_num},"
                f"{result['num_collision_gt']},"
                f"\"{result['gt_collision_frames']}\","
                f"\"{result['detected_collision_frames']}\","
                f"{result['precision']:.3f},"
                f"{result['recall']:.3f},"
                f"{result['f1_score']:.3f},"
                f"{result['accuracy']:.3f}\n"
            )
            f.write(line)        
        collision_gt.append(result['gt_collision_frames'])
        collision_pred.append(len(result['detected_collision_frames']))
    


@hydra.main(version_base=None, config_path="../configs", config_name="config_test_causal")
def run(cfg):
    """Entry point for evaluation."""
    logging.info(f"Slot interaction for {VIDEO_PATH}")

    
    slot_interaction(cfg)


if __name__ == "__main__":
    run()
