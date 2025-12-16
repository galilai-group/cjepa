from pathlib import Path
import hydra
import torch
from loguru import logger as logging
import numpy as np
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.v2 as transforms

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import stable_pretraining as spt
import stable_worldmodel as swm
from custom_models.dinowm_oc import OCWM
from videosaur.videosaur import  models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DINO_PATCH_SIZE = 14
NUM_FRAMESKIP=5
VIDEO_PATH = "/users/hnam16/scratch/clevrer/videos/video_08000.mp4"


def compute_slot_velocities(pca_data: np.ndarray) -> np.ndarray:
    """
    Compute velocity (magnitude of movement) for each slot at each timestep.
    
    Args:
        pca_data: (num_frames, num_slots, 3) PCA coordinates
        
    Returns:
        velocities: (num_frames, num_slots) velocity magnitudes
                   First frame velocity is set to 0
    """
    # Compute frame-to-frame differences
    diffs = np.diff(pca_data, axis=0)  # (num_frames-1, num_slots, 3)
    velocities = np.linalg.norm(diffs, axis=2)  # (num_frames-1, num_slots)
    
    # Pad first frame with zeros
    velocities = np.concatenate([np.zeros((1, pca_data.shape[1])), velocities], axis=0)
    
    return velocities


def get_neon_color(base_color, intensity: float, max_intensity: float = 1.0):
    """
    Create a neon-like color effect based on intensity.
    
    Args:
        base_color: Base RGB color (tuple or array)
        intensity: Current intensity value (0 to 1, relative within slot)
        max_intensity: Maximum intensity for scaling
        
    Returns:
        Modified RGB color with neon effect
    """
    base_rgb = np.array(to_rgb(base_color))
    
    # Intensity determines how "neon" the color becomes
    # High intensity -> brighter, more saturated, slight white blend
    neon_factor = intensity ** 0.5  # Square root for smoother transition
    
    # Increase brightness and add glow effect
    neon_rgb = base_rgb + neon_factor * (1.0 - base_rgb) * 0.7
    
    # Clamp to valid range
    neon_rgb = np.clip(neon_rgb, 0, 1)
    
    return tuple(neon_rgb)


def save_video(pred_pca: np.ndarray, gt_pca: np.ndarray, video_frames: torch.Tensor, 
               output_path: str = "slot_interaction.mp4", fps: int = 42, duration: float = 3.0):
    """
    Create synchronized video with 3 subplots: RGB video, Pred PCA, GT PCA.
    
    Args:
        pred_pca: (num_frames, num_slots, 3) predicted PCA coordinates
        gt_pca: (num_frames, num_slots, 3) ground truth PCA coordinates
        video_frames: (num_frames, C, H, W) RGB video frames
        output_path: Output video file path
        fps: Frames per second for output video
        duration: Target duration in seconds
    """
    num_frames = pred_pca.shape[0]
    num_slots = pred_pca.shape[1]
    
    logging.info(f"Creating video with {num_frames} frames, {num_slots} slots")
    
    # Calculate FPS to match target duration
    target_fps = int(num_frames / duration)
    logging.info(f"Target FPS: {target_fps} for {duration}s duration")
    
    # Get slot colors from seaborn palette
    slot_colors = sns.color_palette("husl", num_slots)
    
    # Compute velocities for neon effect (relative per slot)
    pred_velocities = compute_slot_velocities(pred_pca)  # (num_frames, num_slots)
    gt_velocities = compute_slot_velocities(gt_pca)
    
    # Normalize velocities per slot (relative intensity)
    pred_vel_normalized = np.zeros_like(pred_velocities)
    gt_vel_normalized = np.zeros_like(gt_velocities)
    
    for slot_idx in range(num_slots):
        pred_slot_vel = pred_velocities[:, slot_idx]
        gt_slot_vel = gt_velocities[:, slot_idx]
        
        # Normalize to [0, 1] relative to each slot's own max
        pred_max = pred_slot_vel.max() + 1e-8
        gt_max = gt_slot_vel.max() + 1e-8
        
        pred_vel_normalized[:, slot_idx] = pred_slot_vel / pred_max
        gt_vel_normalized[:, slot_idx] = gt_slot_vel / gt_max
    
    # Compute axis limits for consistent 3D view
    all_pca = np.concatenate([pred_pca.reshape(-1, 3), gt_pca.reshape(-1, 3)], axis=0)
    pca_min, pca_max = all_pca.min(axis=0), all_pca.max(axis=0)
    pca_range = pca_max - pca_min
    pca_center = (pca_max + pca_min) / 2
    max_range = pca_range.max() * 0.6  # Add some padding
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6), facecolor='#2a2a2a')
    
    # Subplot 1: RGB Video
    ax_video = fig.add_subplot(131)
    ax_video.set_facecolor('#2a2a2a')
    ax_video.axis('off')
    ax_video.set_title('RGB Video', color='white', fontsize=14, fontweight='bold', pad=10)
    
    # Subplot 2: Pred PCA
    ax_pred = fig.add_subplot(132, projection='3d', facecolor='#3a3a3a')
    ax_pred.set_facecolor('#3a3a3a')
    ax_pred.set_title('Predicted Slots (PCA)', color='white', fontsize=14, fontweight='bold', pad=10)
    
    # Subplot 3: GT PCA
    ax_gt = fig.add_subplot(133, projection='3d', facecolor='#3a3a3a')
    ax_gt.set_facecolor('#3a3a3a')
    ax_gt.set_title('Ground Truth Slots (PCA)', color='white', fontsize=14, fontweight='bold', pad=10)
    
    # Style 3D axes
    for ax in [ax_pred, ax_gt]:
        ax.set_xlim([pca_center[0] - max_range, pca_center[0] + max_range])
        ax.set_ylim([pca_center[1] - max_range, pca_center[1] + max_range])
        ax.set_zlim([pca_center[2] - max_range, pca_center[2] + max_range])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#555555')
        ax.yaxis.pane.set_edgecolor('#555555')
        ax.zaxis.pane.set_edgecolor('#555555')
        ax.tick_params(colors='#888888', labelsize=8)
        ax.set_xlabel('PC1', color='#888888', fontsize=10)
        ax.set_ylabel('PC2', color='#888888', fontsize=10)
        ax.set_zlabel('PC3', color='#888888', fontsize=10)
        ax.grid(True, alpha=0.2, color='#666666')
    
    plt.tight_layout()
    
    # Initialize plot elements
    video_im = ax_video.imshow(np.zeros((video_frames.shape[2], video_frames.shape[3], 3)))
    
    # Storage for trajectory lines and points
    pred_lines = []
    pred_points = []
    gt_lines = []
    gt_points = []
    
    for slot_idx in range(num_slots):
        color = slot_colors[slot_idx]
        
        # Initialize empty lines for trajectories
        pred_line, = ax_pred.plot([], [], [], color=color, linewidth=1.5, alpha=0.6)
        pred_point, = ax_pred.plot([], [], [], 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=0.5)
        
        gt_line, = ax_gt.plot([], [], [], color=color, linewidth=1.5, alpha=0.6)
        gt_point, = ax_gt.plot([], [], [], 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=0.5)
        
        pred_lines.append(pred_line)
        pred_points.append(pred_point)
        gt_lines.append(gt_line)
        gt_points.append(gt_point)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=slot_colors[i], markersize=8, 
                                   label=f'Slot {i+1}') for i in range(num_slots)]
    ax_pred.legend(handles=legend_elements, loc='upper left', fontsize=8, 
                   facecolor='#3a3a3a', edgecolor='#555555', labelcolor='white')
    
    def update(frame_idx):
        # Update RGB video
        frame = video_frames[frame_idx].permute(1, 2, 0).cpu().numpy()
        # Normalize if needed
        if frame.max() > 1:
            frame = frame / 255.0
        frame = np.clip(frame, 0, 1)
        video_im.set_array(frame)
        
        # Update trajectories for each slot
        for slot_idx in range(num_slots):
            base_color = slot_colors[slot_idx]
            
            # Get trajectory up to current frame
            pred_traj = pred_pca[:frame_idx + 1, slot_idx, :]
            gt_traj = gt_pca[:frame_idx + 1, slot_idx, :]
            
            # Update trajectory lines
            if len(pred_traj) > 1:
                pred_lines[slot_idx].set_data(pred_traj[:, 0], pred_traj[:, 1])
                pred_lines[slot_idx].set_3d_properties(pred_traj[:, 2])
                
                gt_lines[slot_idx].set_data(gt_traj[:, 0], gt_traj[:, 1])
                gt_lines[slot_idx].set_3d_properties(gt_traj[:, 2])
            
            # Get neon intensity for current frame
            pred_intensity = pred_vel_normalized[frame_idx, slot_idx]
            gt_intensity = gt_vel_normalized[frame_idx, slot_idx]
            
            # Apply neon effect to current point
            pred_neon_color = get_neon_color(base_color, pred_intensity)
            gt_neon_color = get_neon_color(base_color, gt_intensity)
            
            # Update current position points with neon effect
            pred_points[slot_idx].set_data([pred_pca[frame_idx, slot_idx, 0]], 
                                           [pred_pca[frame_idx, slot_idx, 1]])
            pred_points[slot_idx].set_3d_properties([pred_pca[frame_idx, slot_idx, 2]])
            pred_points[slot_idx].set_color(pred_neon_color)
            pred_points[slot_idx].set_markersize(8 + pred_intensity * 8)  # Size grows with intensity
            
            gt_points[slot_idx].set_data([gt_pca[frame_idx, slot_idx, 0]], 
                                         [gt_pca[frame_idx, slot_idx, 1]])
            gt_points[slot_idx].set_3d_properties([gt_pca[frame_idx, slot_idx, 2]])
            gt_points[slot_idx].set_color(gt_neon_color)
            gt_points[slot_idx].set_markersize(8 + gt_intensity * 8)
        
        # Add frame counter
        ax_video.set_xlabel(f'Frame: {frame_idx + 1}/{num_frames}', color='white', fontsize=10)
        
        return [video_im] + pred_lines + pred_points + gt_lines + gt_points
    
    # Create animation
    logging.info("Generating animation...")
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, 
        interval=1000 / target_fps,  # milliseconds per frame
        blit=False
    )
    
    # Save video
    logging.info(f"Saving video to {output_path}...")
    writer = animation.FFMpegWriter(fps=target_fps, bitrate=5000)
    anim.save(output_path, writer=writer, dpi=150)
    
    plt.close(fig)
    logging.info(f"Video saved successfully: {output_path}")

        
   

def read_video():
    '''
    Docstring for read_video
    input: video file path
    output: video frames for a whole video with a given frameskip
    '''
    video = VideoDecoder(VIDEO_PATH)
    return video[0 : -1 : NUM_FRAMESKIP].to(device)

def load_model_from_checkpoint(cfg):

    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path must be specified in config!")
    ckpt_path = Path(cfg.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    logging.info(f"Loading checkpoint from: {ckpt_path}")

    model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = model.encoder 
    slot_attention = model.processor 
    initializer = model.initializer
    embedding_dim = cfg.videosaur.SLOT_DIM 
    num_patches = cfg.videosaur.NUM_SLOTS

    if cfg.training_type == "wm":
        embedding_dim += cfg.dinowm.proprio_embed_dim + cfg.dinowm.action_embed_dim  # Total embedding size
    logging.info(f"Patches: {num_patches}, Embedding dim: {embedding_dim}")
    
    # For VideoWM, no action/proprio encoders
    predictor = swm.wm.dinowm.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim,
        **cfg.predictor,
    )

    # Build action and proprioception encoders
    if cfg.training_type == "video":    
        action_encoder = None
        proprio_encoder = None
        logging.info(f"[Video Only] Action encoder: None, Proprio encoder: None")
    else :
        effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
        action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=cfg.dinowm.action_embed_dim)
        proprio_encoder = swm.wm.dinowm.Embedder(in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.dinowm.proprio_embed_dim)
        logging.info(f"Action dim: {effective_act_dim}, Proprio dim: {cfg.dinowm.proprio_dim}")
    
    world_model = OCWM(
        encoder=spt.backbone.EvalOnly(encoder),
        slot_attention=spt.backbone.EvalOnly(slot_attention),
        initializer = spt.backbone.EvalOnly(initializer),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
    )
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    ckpt_state = checkpoint.model.state_dict()
    
    # Load into world_model
    missing, unexpected = world_model.load_state_dict(ckpt_state, strict=False)
    logging.info(f"Missing keys: {len(missing) if missing else 0}")
    logging.info(f"Unexpected keys: {len(unexpected) if unexpected else 0}")
    
    # Set to eval model
    world_model.eval()
    
    return world_model


def slot_interaction(cfg):
    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    world_model = load_model_from_checkpoint(cfg)
    world_model = world_model.to(device)
    video_frames = read_video()

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
            pred_embedding = world_model.predict(input_embedding)
            pred_collection.append(pred_embedding[0, 0, :, :].cpu().numpy().squeeze()) # batchsize is always 1
            gt_collection.append(x["embed"][0, 1, :, :].cpu().numpy().squeeze()) # batchsize is always 1
    logging.info(f"Collection length: {len(pred_collection)}")

    pred_stacked = np.stack(pred_collection, axis=0) # source: (num_frames, num_slots, embedding_dim)
    gt_stacked = np.stack(gt_collection, axis=0) # source: (num_frames, num_slots, embedding_dim)
    pca = PCA(n_components=3, random_state=42)
    # fit PCA on the ground truth data
    pca.fit(gt_stacked.reshape(-1, gt_stacked.shape[-1])) # this ensures that the dimension reduction is done on the axis of the last dimension: embedding_dim
    pred_pca = pca.transform(pred_stacked.reshape(-1, pred_stacked.shape[-1])).reshape(pred_stacked.shape[0], -1, 3) # reshape back to (num_frames, num_slots, 3)
    gt_pca = pca.transform(gt_stacked.reshape(-1, gt_stacked.shape[-1])).reshape(gt_stacked.shape[0], -1, 3) # reshape back to (num_frames, num_slots, 3)
    video_frames_plot = video_frames[1:, :, :, :]  # Align with predictions: (num_frames, C, H, W)

    # Create output path
    output_path = Path(cfg.get('output_dir', './eval_results')) / "slot_interaction.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_video(pred_pca, gt_pca, video_frames_plot, output_path=str(output_path))

@hydra.main(version_base=None, config_path="../configs", config_name="config_test_oc")
def run(cfg):
    """Entry point for evaluation."""
    logging.info(f"Slot interaction for {VIDEO_PATH}")

    
    slot_interaction(cfg)


if __name__ == "__main__":
    run()
