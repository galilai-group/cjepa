from pathlib import Path
import hydra
import torch
from loguru import logger as logging
import numpy as np
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.v2 as transforms
import json

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import stable_pretraining as spt
import stable_worldmodel as swm
from custom_models.dinowm_causal import CausalWM
from custom_models.cjepa_predictor import MaskedSlotPredictor
from videosaur.videosaur import  models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_video(video_path, num_frameskip, return_idx=False):
    '''
    Docstring for read_video
    input: video file path
    output: video frames for a whole video with a given frameskip
    '''
    video = VideoDecoder(video_path)
    if return_idx:
        idx = np.arange(len(video))[0 : -1 : num_frameskip]
        return video[0 : -1 : num_frameskip].to(device), idx
    else:
        return video[0 : -1 : num_frameskip].to(device)

def eval_colision(stacked_slots, stacked_video_frames_idx, annot_dir=None):
    """
    Evaluate collision events between slots based on predicted and ground truth slot trajectories.
    
    Args:
        stacked_slots: (num_frames, num_slots, embedding_dim) predicted slot embeddings
        stacked_video_frames_idx: List of frame indices corresponding to stacked_slots
        annot_dir: Path to annotations JSON file containing collision events
    """
    # read annotations json
    if annot_dir is not None:
        with open(annot_dir, 'r') as f:
            annotations = json.load(f)
        collision = annotations['collision']
        # assert len(collision) == 1
        num_collision = len(collision)
        collision_frames = [item['frame_id'] for item in collision]

        # find first nearest and second nearest frames in stacked_video_frames_idx
        collision_dict = {}
        allowed_collision_gt = []
        for c_frame in collision_frames:
            diffs = np.abs(np.array(stacked_video_frames_idx) - c_frame)
            nearest_idx = np.argmin(diffs)
            if nearest_idx == 0:
                second_nearest_idx = 1
            elif nearest_idx == len(stacked_video_frames_idx) - 1:
                second_nearest_idx = len(stacked_video_frames_idx) - 2
            else:
                if diffs[nearest_idx - 1] < diffs[nearest_idx + 1]:
                    second_nearest_idx = nearest_idx - 1
                else:
                    second_nearest_idx = nearest_idx + 1
            collision_dict[c_frame] = (nearest_idx, second_nearest_idx)
            allowed_collision_gt.append(int(stacked_video_frames_idx[nearest_idx]))
            allowed_collision_gt.append(int(stacked_video_frames_idx[second_nearest_idx]))
        
        detected_collision_frames = detect_collision_frames(stacked_slots, num_collision, stacked_video_frames_idx)

        true_positives = []
        false_positives = []
        for tuple_pair in detected_collision_frames:
            if tuple_pair[0] in allowed_collision_gt or tuple_pair[1] in allowed_collision_gt:
                true_positives.append(tuple_pair)
            else:
                false_positives.append(tuple_pair)
        false_negatives_len = num_collision - len(true_positives)


        # calculate acc, prec, recision, f1
        # true_positives = set(detected_collision_frames).intersection(set(allowed_collision_gt))
        # false_positives = set(detected_collision_frames) - set(allowed_collision_gt)
        # false_negatives = set(allowed_collision_gt) - set(detected_collision_frames)
        precision = len(true_positives) / (len(true_positives) + len(false_positives) + 1e-8)
        recall = len(true_positives) / (len(true_positives) + false_negatives_len + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        acc = len(true_positives) / num_collision
    

    return {
        'num_collision_gt': int(num_collision),
        'gt_collision_frames': [x for x in allowed_collision_gt],
        'detected_collision_frames': [x for x in detected_collision_frames],
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(acc)
    }
        


        
def detect_collision_frames(slot_embeddings: np.ndarray, num_collisions: int, stacked_video_frames_idx) -> list:
    """
    Detect collision frames based on sudden changes in all slot embeddings.
    
    Computes the total change (sum of all slot changes) at each frame transition,
    then returns the top-k frames with the largest total changes.
    
    Args:
        slot_embeddings: (num_frames, num_slots, embedding_dim) slot embeddings over time
        num_collisions: Number of collision events to detect (returns top-k frames)
        stacked_video_frames_idx: List of original frame indices corresponding to slot_embeddings
        
    Returns:
        collision_frames: List of frame indices (sorted by change magnitude, descending)
                         Each index represents frame t where large change occurred between t and t+1
    """
    num_frames, num_slots, embed_dim = slot_embeddings.shape
    
    if num_frames < 2:
        return []
    
    # Compute frame-to-frame differences for each slot
    # diffs[t] = embeddings[t+1] - embeddings[t]
    diffs = slot_embeddings[1:] - slot_embeddings[:-1]  # (num_frames-1, num_slots, embed_dim)
    
    # Compute L2 norm of change for each slot at each timestep
    change_per_slot = np.linalg.norm(diffs, axis=2)  # (num_frames-1, num_slots)
    
    # Sum across all slots to get total change per frame
    total_change_per_frame = np.sum(change_per_slot, axis=1)  # (num_frames-1,)
    sorted_vals = np.argsort(total_change_per_frame)[::-1]
    
    # Get top-k frames with largest total changes, but should be paired with t and t+1
    collision_frames = []
    for t in range(num_collisions):
        top_index = sorted_vals[0]
        # handle edge cases
        if top_index == 0:
            top_index_pair = 1
        elif top_index == len(total_change_per_frame) -1:
            top_index_pair = top_index -1
        elif top_index + 1 not in sorted_vals:
            top_index_pair = top_index -1
        elif top_index -1 not in sorted_vals:
            top_index_pair = top_index +1
        else:
            top_index_pair = top_index + 1 if total_change_per_frame[top_index + 1] > total_change_per_frame[top_index - 1] else top_index -1
        # remove top_index, top_index_pair from sorted
        indices = np.where((sorted_vals != top_index) & (sorted_vals != top_index_pair))
        sorted_vals = sorted_vals[indices]
        collision_frames.append(((int(stacked_video_frames_idx[top_index]), int(stacked_video_frames_idx[top_index_pair]))))



    # num_to_detect = min(num_collisions, len(total_change_per_frame))
    # top_indices = np.argsort(total_change_per_frame)[::-1][:num_to_detect]
    
    # # Return as sorted list of frame indices
    # collision_frames = [int(idx) for idx in top_indices]
    
    return collision_frames


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


def save_video_2d(pred_pca: np.ndarray, gt_pca: np.ndarray, video_frames: torch.Tensor, 
               output_path: str = "slot_interaction.mp4", fps: int = 42, duration: float = 3.0):
    """
    Create synchronized video with 3 subplots: RGB video, Pred PCA, GT PCA (2D version).
    
    Args:
        pred_pca: (num_frames, num_slots, 2) predicted PCA coordinates
        gt_pca: (num_frames, num_slots, 2) ground truth PCA coordinates
        video_frames: (num_frames, C, H, W) RGB video frames
        output_path: Output video file path
        fps: Frames per second for output video
        duration: Target duration in seconds
    """
    num_frames = pred_pca.shape[0]
    num_slots = pred_pca.shape[1]
    
    logging.info(f"Creating 2D video with {num_frames} frames, {num_slots} slots")
    
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
    
    # Compute axis limits separately for pred and gt (adaptive per-axis)
    pred_flat = pred_pca.reshape(-1, 2)
    gt_flat = gt_pca.reshape(-1, 2)
    
    pred_min, pred_max = pred_flat.min(axis=0), pred_flat.max(axis=0)
    pred_range = pred_max - pred_min
    pred_padding = pred_range * 0.1
    pred_axis_min = pred_min - pred_padding
    pred_axis_max = pred_max + pred_padding
    
    gt_min, gt_max = gt_flat.min(axis=0), gt_flat.max(axis=0)
    gt_range = gt_max - gt_min
    gt_padding = gt_range * 0.1
    gt_axis_min = gt_min - gt_padding
    gt_axis_max = gt_max + gt_padding
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6), facecolor='#2a2a2a')
    
    # Subplot 1: RGB Video
    ax_video = fig.add_subplot(131)
    ax_video.set_facecolor('#2a2a2a')
    ax_video.axis('off')
    ax_video.set_title('RGB Video', color='white', fontsize=14, fontweight='bold', pad=10)
    
    # Subplot 2: Pred PCA (2D)
    ax_pred = fig.add_subplot(132, facecolor='#3a3a3a')
    ax_pred.set_facecolor('#3a3a3a')
    ax_pred.set_title('Predicted Slots (PCA)', color='white', fontsize=14, fontweight='bold', pad=10)
    
    # Subplot 3: GT PCA (2D)
    ax_gt = fig.add_subplot(133, facecolor='#3a3a3a')
    ax_gt.set_facecolor('#3a3a3a')
    ax_gt.set_title('Ground Truth Slots (PCA)', color='white', fontsize=14, fontweight='bold', pad=10)
    
    # Style 2D axes with separate limits
    ax_pred.set_xlim([pred_axis_min[0], pred_axis_max[0]])
    ax_pred.set_ylim([pred_axis_min[1], pred_axis_max[1]])
    
    ax_gt.set_xlim([gt_axis_min[0], gt_axis_max[0]])
    ax_gt.set_ylim([gt_axis_min[1], gt_axis_max[1]])
    
    for ax in [ax_pred, ax_gt]:
        ax.tick_params(colors='#888888', labelsize=8)
        ax.set_xlabel('PC1', color='#888888', fontsize=10)
        ax.set_ylabel('PC2', color='#888888', fontsize=10)
        ax.grid(True, alpha=0.2, color='#666666')
        ax.spines['bottom'].set_color('#555555')
        ax.spines['top'].set_color('#555555')
        ax.spines['left'].set_color('#555555')
        ax.spines['right'].set_color('#555555')
        ax.set_aspect('auto', adjustable='box')  # auto aspect for better visibility
    
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
        
        # Initialize empty lines for trajectories (2D)
        pred_line, = ax_pred.plot([], [], color=color, linewidth=1.5, alpha=0.6)
        pred_point, = ax_pred.plot([], [], 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=0.5)
        
        gt_line, = ax_gt.plot([], [], color=color, linewidth=1.5, alpha=0.6)
        gt_point, = ax_gt.plot([], [], 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=0.5)
        
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
            
            # Update trajectory lines (2D)
            if len(pred_traj) > 1:
                pred_lines[slot_idx].set_data(pred_traj[:, 0], pred_traj[:, 1])
                gt_lines[slot_idx].set_data(gt_traj[:, 0], gt_traj[:, 1])
            
            # Get neon intensity for current frame
            pred_intensity = pred_vel_normalized[frame_idx, slot_idx]
            gt_intensity = gt_vel_normalized[frame_idx, slot_idx]
            
            # Apply neon effect to current point
            pred_neon_color = get_neon_color(base_color, pred_intensity)
            gt_neon_color = get_neon_color(base_color, gt_intensity)
            
            # Update current position points with neon effect (2D)
            pred_points[slot_idx].set_data([pred_pca[frame_idx, slot_idx, 0]], 
                                           [pred_pca[frame_idx, slot_idx, 1]])
            pred_points[slot_idx].set_color(pred_neon_color)
            pred_points[slot_idx].set_markersize(8 + pred_intensity * 8)  # Size grows with intensity
            
            gt_points[slot_idx].set_data([gt_pca[frame_idx, slot_idx, 0]], 
                                         [gt_pca[frame_idx, slot_idx, 1]])
            gt_points[slot_idx].set_color(gt_neon_color)
            gt_points[slot_idx].set_markersize(8 + gt_intensity * 8)
        
        # Add frame counter
        ax_video.set_xlabel(f'Frame: {frame_idx + 1}/{num_frames}', color='white', fontsize=10)
        
        return [video_im] + pred_lines + pred_points + gt_lines + gt_points
    
    # Create animation
    logging.info("Generating 2D animation...")
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
    logging.info(f"2D Video saved successfully: {output_path}")


def save_video_3d(pred_pca: np.ndarray, gt_pca: np.ndarray, video_frames: torch.Tensor, 
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
    
    # Compute axis limits separately for pred and gt (adaptive per-axis)
    pred_flat = pred_pca.reshape(-1, 3)
    gt_flat = gt_pca.reshape(-1, 3)
    
    pred_min, pred_max = pred_flat.min(axis=0), pred_flat.max(axis=0)
    pred_range = pred_max - pred_min
    pred_padding = pred_range * 0.1
    pred_axis_min = pred_min - pred_padding
    pred_axis_max = pred_max + pred_padding
    
    gt_min, gt_max = gt_flat.min(axis=0), gt_flat.max(axis=0)
    gt_range = gt_max - gt_min
    gt_padding = gt_range * 0.1
    gt_axis_min = gt_min - gt_padding
    gt_axis_max = gt_max + gt_padding
    
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
    
    # Style 3D axes with separate limits
    ax_pred.set_xlim([pred_axis_min[0], pred_axis_max[0]])
    ax_pred.set_ylim([pred_axis_min[1], pred_axis_max[1]])
    ax_pred.set_zlim([pred_axis_min[2], pred_axis_max[2]])
    
    ax_gt.set_xlim([gt_axis_min[0], gt_axis_max[0]])
    ax_gt.set_ylim([gt_axis_min[1], gt_axis_max[1]])
    ax_gt.set_zlim([gt_axis_min[2], gt_axis_max[2]])
    
    for ax in [ax_pred, ax_gt]:
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