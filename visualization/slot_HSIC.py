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
from visualization.utils import save_video_2d, save_video_3d, read_video
from visualization.model import load_OC_model_from_checkpoint
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



DINO_PATCH_SIZE = 14
NUM_FRAMESKIP=5
VIDEO_PATH = "/cs/data/people/hnam16/data/clevrer/videos/video_08000.mp4"
PCA_COMPONENTS = 2
   

def hsic(p1: np.ndarray, p2: np.ndarray, sigma: float = None) -> float:
    """
    Args:
        p1: First matrix of shape (T, d1) - T samples, d1 features
        p2: Second matrix of shape (T, d2) - T samples, d2 features
        sigma: Bandwidth for RBF kernel. If None, uses median heuristic.
        
    Returns:
        HSIC value (scalar) - higher means more dependent
    """
    assert p1.shape[0] == p2.shape[0], "p1 and p2 must have same number of samples (T)"
    
    T = p1.shape[0]
    
    # Compute pairwise squared distances
    def squared_distances(X: np.ndarray) -> np.ndarray:
        """Compute pairwise squared Euclidean distances."""
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i @ x_j
        sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
        return sq_norms + sq_norms.T - 2 * X @ X.T
    
    # Compute RBF (Gaussian) kernel
    def rbf_kernel(X: np.ndarray, sigma: float) -> np.ndarray:
        """Compute RBF kernel matrix."""
        sq_dists = squared_distances(X)
        return np.exp(-sq_dists / (2 * sigma ** 2))
    
    # Median heuristic for bandwidth selection
    def median_heuristic(X: np.ndarray) -> float:
        """Compute bandwidth using median heuristic."""
        sq_dists = squared_distances(X)
        # Take median of non-zero distances
        triu_indices = np.triu_indices(T, k=1)
        distances = np.sqrt(sq_dists[triu_indices])
        return np.median(distances) + 1e-8
    
    # Determine bandwidth
    if sigma is None:
        sigma1 = median_heuristic(p1)
        sigma2 = median_heuristic(p2)
    else:
        sigma1 = sigma2 = sigma
    
    # Compute kernel matrices
    K = rbf_kernel(p1, sigma1)
    L = rbf_kernel(p2, sigma2)
    
    # Centering matrix H = I - (1/T) * 1 * 1^T
    H = np.eye(T) - np.ones((T, T)) / T
    
    # Centered kernels
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    # HSIC = (1 / (T-1)^2) * trace(K_centered @ L_centered)
    hsic_value = np.trace(K_centered @ L_centered) / ((T - 1) ** 2)
    
    return hsic_value


def normalized_hsic(p1: np.ndarray, p2: np.ndarray, sigma: float = None) -> float:
    """
    Compute normalized HSIC (CKA - Centered Kernel Alignment).
    Normalized to [0, 1] range where 0 = independent, 1 = perfectly dependent.
    
    Args:
        p1: First matrix of shape (T, d1)
        p2: Second matrix of shape (T, d2)
        sigma: Bandwidth for RBF kernel. If None, uses median heuristic.
        
    Returns:
        Normalized HSIC value in [0, 1]
    """
    hsic_xy = hsic(p1, p2, sigma)
    hsic_xx = hsic(p1, p1, sigma)
    hsic_yy = hsic(p2, p2, sigma)
    
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-8)


def save_video_hsic(mat_overall: np.ndarray, temporal: list, video_frames: torch.Tensor,
                    output_path: str = "hsic_video.mp4", duration: float = 3.5):
    """
    Create a video with 3 subplots: RGB video, temporal HSIC heatmap, overall HSIC heatmap.
    
    Args:
        mat_overall: (num_slots, num_slots) overall HSIC matrix (static)
        temporal: List of (num_slots, num_slots) HSIC matrices for each frame
        video_frames: (num_frames, C, H, W) RGB video frames
        output_path: Output video file path
        duration: Target duration in seconds
    """
    num_frames = len(temporal)
    num_slots = mat_overall.shape[0]
    
    logging.info(f"Creating HSIC video with {num_frames} frames, {num_slots} slots")
    
    # Calculate FPS to match target duration
    target_fps = int(num_frames / duration)
    logging.info(f"Target FPS: {target_fps} for {duration}s duration")
    
    # Stack temporal matrices for consistent color scaling
    temporal_stacked = np.stack(temporal, axis=0)  # (T, num_slots, num_slots)
    
    # Get global min/max for consistent colormap across temporal frames (only upper triangle)
    upper_tri_mask = np.triu(np.ones((num_slots, num_slots), dtype=bool), k=1)
    temporal_upper = temporal_stacked[:, upper_tri_mask]
    temporal_min = temporal_upper.min()
    temporal_max = temporal_upper.max()
    overall_upper = mat_overall[upper_tri_mask]
    overall_min = overall_upper.min()
    overall_max = overall_upper.max()
    
    # Use global min/max for both to make them comparable
    vmin = min(temporal_min, overall_min)
    vmax = max(temporal_max, overall_max)
    
    # Create masked arrays to hide lower triangle and diagonal
    lower_tri_mask = ~upper_tri_mask  # includes diagonal and lower triangle
    mat_overall_masked = np.ma.array(mat_overall, mask=lower_tri_mask)
    temporal_masked = [np.ma.array(t, mask=lower_tri_mask) for t in temporal]
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6), facecolor='#2a2a2a')
    
    # Subplot 1: RGB Video
    ax_video = fig.add_subplot(131)
    ax_video.set_facecolor('#2a2a2a')
    ax_video.axis('off')
    ax_video.set_title('RGB Video', color='white', fontsize=14, fontweight='bold', pad=10)
    
    # Subplot 2: Temporal HSIC heatmap
    ax_temporal = fig.add_subplot(132)
    ax_temporal.set_facecolor('#2a2a2a')
    ax_temporal.set_title('Temporal HSIC (sliding window)', color='white', fontsize=14, fontweight='bold', pad=10)
    
    # Subplot 3: Overall HSIC heatmap (static)
    ax_overall = fig.add_subplot(133)
    ax_overall.set_facecolor('#2a2a2a')
    ax_overall.set_title('Overall HSIC (all frames)', color='white', fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Initialize video frame
    video_im = ax_video.imshow(np.zeros((video_frames.shape[2], video_frames.shape[3], 3)))
    
    # Initialize temporal heatmap with masked array
    temporal_im = ax_temporal.imshow(
        temporal_masked[0], cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='equal'
    )
    # Set background color for masked areas (lower triangle + diagonal)
    ax_temporal.set_facecolor('#1a1a1a')
    ax_temporal.set_xticks(range(num_slots))
    ax_temporal.set_yticks(range(num_slots))
    ax_temporal.set_xticklabels([f'S{i+1}' for i in range(num_slots)], color='white', fontsize=9)
    ax_temporal.set_yticklabels([f'S{i+1}' for i in range(num_slots)], color='white', fontsize=9)
    ax_temporal.tick_params(colors='white')
    
    # Add colorbar for temporal
    cbar_temporal = plt.colorbar(temporal_im, ax=ax_temporal, fraction=0.046, pad=0.04)
    cbar_temporal.ax.yaxis.set_tick_params(color='white')
    cbar_temporal.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar_temporal.ax.axes, 'yticklabels'), color='white')
    
    # Draw static overall heatmap with masked array
    overall_im = ax_overall.imshow(
        mat_overall_masked, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='equal'
    )
    # Set background color for masked areas (lower triangle + diagonal)
    ax_overall.set_facecolor('#1a1a1a')
    ax_overall.set_xticks(range(num_slots))
    ax_overall.set_yticks(range(num_slots))
    ax_overall.set_xticklabels([f'S{i+1}' for i in range(num_slots)], color='white', fontsize=9)
    ax_overall.set_yticklabels([f'S{i+1}' for i in range(num_slots)], color='white', fontsize=9)
    ax_overall.tick_params(colors='white')
    
    # Add colorbar for overall
    cbar_overall = plt.colorbar(overall_im, ax=ax_overall, fraction=0.046, pad=0.04)
    cbar_overall.ax.yaxis.set_tick_params(color='white')
    cbar_overall.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar_overall.ax.axes, 'yticklabels'), color='white')
    
    # Add value annotations to overall heatmap
    for i in range(num_slots):
        for j in range(num_slots):
            if mat_overall[i, j] != 0:  # Only annotate non-zero (upper triangle)
                text_color = 'white' if mat_overall[i, j] < (vmax + vmin) / 2 else 'black'
                ax_overall.text(j, i, f'{mat_overall[i, j]:.2f}', 
                               ha='center', va='center', color=text_color, fontsize=8)
    
    def update(frame_idx):
        # Update RGB video
        frame = video_frames[frame_idx].permute(1, 2, 0).cpu().numpy()
        if frame.max() > 1:
            frame = frame / 255.0
        frame = np.clip(frame, 0, 1)
        video_im.set_array(frame)
        
        # Update temporal heatmap
        temporal_im.set_array(temporal_masked[frame_idx])
        
        # Update frame counter
        ax_video.set_xlabel(f'Frame: {frame_idx + 1}/{num_frames}', color='white', fontsize=10)
        
        return [video_im, temporal_im]
    
    # Create animation
    logging.info("Generating HSIC animation...")
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=1000 / target_fps,
        blit=False
    )
    
    # Save video
    logging.info(f"Saving video to {output_path}...")
    writer = animation.FFMpegWriter(fps=target_fps, bitrate=5000)
    anim.save(output_path, writer=writer, dpi=150)
    
    plt.close(fig)
    logging.info(f"HSIC video saved successfully: {output_path}")


def slot_interaction(cfg):
    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    world_model = load_OC_model_from_checkpoint(cfg)
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
            pred_embedding = world_model.predict(input_embedding, use_inference_function=False)
            print(pred_embedding.shape, x["embed"].shape)
            
            pred = pred_embedding[0, -1, :, :].cpu().numpy().squeeze()
            gt = x["embed"][0, -1, :, :].cpu().numpy().squeeze()
            pred_collection.append(pred) # batchsize is always 1
            gt_collection.append(gt) # batchsize is always 1
    logging.info(f"Collection length: {len(pred_collection)}")

    pred_stacked = np.stack(pred_collection, axis=0) # source: (num_frames, num_slots, embedding_dim)
    gt_stacked = np.stack(gt_collection, axis=0) # source: (num_frames, num_slots, embedding_dim)
    pred_stacked -= pred_stacked.mean(axis=0, keepdims=True)  # (T, slots, d) -> average on axis T
    gt_stacked -= gt_stacked.mean(axis=0, keepdims=True)

    num_slots = gt_stacked.shape[1]

    mat_overall = np.zeros((num_slots, num_slots))
    for i in range(num_slots):
        for j in range(i+1, num_slots):
            mat_overall[i, j] = normalized_hsic(pred_stacked[:,i,:], pred_stacked[:, j, :])

    # Min-Max normalize mat_overall to emphasize relative differences
    # Only consider upper triangle (non-zero values)
    upper_tri_mask = np.triu(np.ones_like(mat_overall, dtype=bool), k=1)
    upper_vals = mat_overall[upper_tri_mask]
    if upper_vals.max() - upper_vals.min() > 1e-8:
        mat_overall[upper_tri_mask] = (upper_vals - upper_vals.min()) / (upper_vals.max() - upper_vals.min())
    
    logging.info(f"Overall HSIC range after normalization: [{mat_overall[upper_tri_mask].min():.3f}, {mat_overall[upper_tri_mask].max():.3f}]")

    # Use sliding window for temporal HSIC (HSIC requires T >= 2)
    window_size = 5  # Use past window_size frames for each temporal HSIC calculation
    temporal = []
    T = gt_stacked.shape[0]
    
    for t in range(T):
        mat = np.zeros((num_slots, num_slots))
        # Define window: use frames from max(0, t-window_size+1) to t+1
        start_idx = max(0, t - window_size + 1)
        end_idx = t + 1
        
        # Need at least 2 frames for HSIC
        if end_idx - start_idx >= 2:
            for i in range(num_slots):
                for j in range(i+1, num_slots):
                    p1 = pred_stacked[start_idx:end_idx, i, :]  # (window, d)
                    p2 = pred_stacked[start_idx:end_idx, j, :]  # (window, d)
                    mat[i, j] = normalized_hsic(p1, p2)
        # else: mat stays zeros for first frame
        
        temporal.append(mat)

    # Min-Max normalize temporal matrices (excluding first frame which is all zeros)
    upper_tri_mask = np.triu(np.ones((num_slots, num_slots), dtype=bool), k=1)
    temporal_stacked = np.stack(temporal, axis=0)  # (T, num_slots, num_slots)
    
    # Find first valid frame (with non-zero values)
    first_valid_idx = 1  # First frame is always zeros (T < 2)
    temporal_upper_valid = temporal_stacked[first_valid_idx:, upper_tri_mask]  # exclude first frame
    
    global_min = temporal_upper_valid.min()
    global_max = temporal_upper_valid.max()
    
    if global_max - global_min > 1e-8:
        # Normalize each temporal matrix (skip first frame, keep it as zeros)
        for t in range(first_valid_idx, len(temporal)):
            upper_vals = temporal[t][upper_tri_mask]
            temporal[t][upper_tri_mask] = (upper_vals - global_min) / (global_max - global_min)
    
    logging.info(f"Temporal HSIC range after normalization (excl. first frame): [{global_min:.3f}, {global_max:.3f}] -> [0, 1]")

    video_frames_plot = video_frames[cfg.dinowm.history_size:, :, :, :]  # Align with predictions: (num_frames, C, H, W)

    # Create output path
    vidname  = VIDEO_PATH.split('/')[-1].split('.')[0]
    ckpt_name = cfg.checkpoint_path.split('/')[-1].split('.')[0]
    output_path = Path(cfg.get('output_dir', './eval_results')) / f"HSIC_slot_interaction_{vidname}_{ckpt_name}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_video_hsic(mat_overall, temporal, video_frames_plot, output_path=str(output_path))
    

@hydra.main(version_base=None, config_path="../configs", config_name="config_test_oc")
def run(cfg):
    """Entry point for evaluation."""
    logging.info(f"Slot interaction for {VIDEO_PATH}")

    
    slot_interaction(cfg)


if __name__ == "__main__":
    run()
