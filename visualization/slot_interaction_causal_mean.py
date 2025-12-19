from pathlib import Path
import hydra
import torch
from loguru import logger as logging
import numpy as np
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.v2 as transforms

from sklearn.decomposition import PCA

import stable_pretraining as spt
import stable_worldmodel as swm
from custom_models.dinowm_causal import CausalWM
from custom_models.cjepa_predictor import MaskedSlotPredictor
from videosaur.videosaur import  models

from visualization.utils import save_video_2d, save_video_3d, read_video
from visualization.model import load_causal_model_from_checkpoint



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DINO_PATCH_SIZE = 14
VIDEO_PATH = "/cs/data/people/hnam16/data/clevrer/videos/video_08000.mp4"
PCA_COMPONENTS = 2
NUM_FRAMESKIP=5



def slot_interaction(cfg):
    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    world_model = load_causal_model_from_checkpoint(cfg)
    world_model = world_model.to(device)
    video_frames = read_video(VIDEO_PATH, NUM_FRAMESKIP)

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
    
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    # fit PCA on the ground truth data
    pca.fit(gt_stacked.reshape(-1, gt_stacked.shape[-1])) # this ensures that the dimension reduction is done on the axis of the last dimension: embedding_dim
    pred_pca = pca.transform(pred_stacked.reshape(-1, pred_stacked.shape[-1])).reshape(pred_stacked.shape[0], -1, PCA_COMPONENTS) # reshape back to (num_frames, num_slots, 3)
    gt_pca = pca.transform(gt_stacked.reshape(-1, gt_stacked.shape[-1])).reshape(gt_stacked.shape[0], -1, PCA_COMPONENTS) # reshape back to (num_frames, num_slots, 3)
    video_frames_plot = video_frames[cfg.dinowm.history_size:, :, :, :]  # Align with predictions: (num_frames, C, H, W)

    # Create output path
    vidname  = VIDEO_PATH.split('/')[-1].split('.')[0]
    ckpt_name = cfg.checkpoint_path.split('/')[-1].split('.')[0]
    output_path = Path(cfg.get('output_dir', './eval_results')) / f"mean_slot_interaction_{vidname}_{ckpt_name}_pca{PCA_COMPONENTS}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if PCA_COMPONENTS == 3:
        save_video_3d(pred_pca, gt_pca, video_frames_plot, output_path=str(output_path))
    elif PCA_COMPONENTS == 2:
        save_video_2d(pred_pca, gt_pca, video_frames_plot, output_path=str(output_path))

@hydra.main(version_base=None, config_path="../configs", config_name="config_test_causal")
def run(cfg):
    """Entry point for evaluation."""
    logging.info(f"Slot interaction for {VIDEO_PATH}")

    
    slot_interaction(cfg)


if __name__ == "__main__":
    run()
