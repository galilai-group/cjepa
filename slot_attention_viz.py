from pathlib import Path
import hydra
import torch
from loguru import logger as logging
import numpy as np
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.v2 as transforms
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

import stable_pretraining as spt
import stable_worldmodel as swm
from custom_models.dinowm_oc import OCWM
from videosaur.videosaur import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DINO_PATCH_SIZE = 14
NUM_FRAMESKIP = 5
VIDEO_PATH = "/users/hnam16/scratch/clevrer/videos/video_08000.mp4"


def extract_attention_weights(model, batch):
    """
    Extract attention weights from slot attention module.
    
    Args:
        model: OCWM model with slot_attention module
        batch: Input batch with 'pixels' key
        
    Returns:
        attention_weights: (T, num_slots, num_patches) attention weights over time
        slots: (T, num_slots, D) slot embeddings
        patches: (T, num_patches, D) patch embeddings
    """
    attention_weights_collection = []
    slots_collection = []
    patches_collection = []
    
    with torch.no_grad():
        # Encode frames to get patch embeddings
        encoded = model.encode(batch, target='embed', pixels_key='pixels')
        full_embed = encoded['embed']  # (B, T, num_patches, D)
        
        B, T, P, D = full_embed.shape
        
        # For each timestep, run slot attention and extract attention weights
        for t in range(T):
            patches_t = full_embed[:, t, :, :]  # (B, num_patches, D)
            
            # Initialize slots - initializer only needs batch size (integer)
            batch_size = B  # We already have B from unpacking shape above
            slots_init = model.initializer.modules(batch_size)  # (B, num_slots, slot_dim)
            
            # Run slot attention - need to hook into attention computation
            # Slot attention computes: attn = softmax(q @ k^T / sqrt(d))
            # where q comes from slots, k comes from patches
            
            # Access the slot attention module
            slot_attn = model.slot_attention.modules  # Unwrap EvalOnly
            
            # Store original forward to restore later
            slots = slots_init
            inputs = patches_t
            
            # Run slot attention iterations and capture attention from last iteration
            for iteration in range(slot_attn.n_iters):
                slots_prev = slots
                
                # Normalization
                if hasattr(slot_attn, 'norm_slots'):
                    slots_for_attn = slot_attn.norm_slots(slots)
                else:
                    slots_for_attn = slots
                    
                if hasattr(slot_attn, 'norm_inputs'):
                    inputs_for_attn = slot_attn.norm_inputs(inputs)
                else:
                    inputs_for_attn = inputs
                
                # Compute attention weights
                # Q from slots: (B, num_slots, dim)
                q = slot_attn.to_q(slots_for_attn)
                
                # K, V from inputs (patches): (B, num_patches, dim)
                k = slot_attn.to_k(inputs_for_attn)
                v = slot_attn.to_v(inputs_for_attn)
                
                # Multi-head attention (simplified - using single head for visualization)
                # Attention scores: (B, num_slots, num_patches)
                scale = slot_attn.dim_head ** -0.5
                attn_logits = torch.einsum('bnd,bmd->bnm', q, k) * scale
                attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, num_slots, num_patches)
                
                # Weighted sum
                updates = torch.einsum('bnm,bmd->bnd', attn_weights, v)
                
                # MLP update
                if slot_attn.use_mlp:
                    slots = slots_prev + slot_attn.mlp(updates)
                else:
                    slots = slots_prev + updates
            
            # Store final attention weights from last iteration
            attention_weights_collection.append(attn_weights.squeeze(0).cpu().numpy())  # (num_slots, num_patches)
            slots_collection.append(slots.squeeze(0).cpu().numpy())  # (num_slots, D)
            patches_collection.append(patches_t.squeeze(0).cpu().numpy())  # (num_patches, D)
    
    attention_weights = np.stack(attention_weights_collection, axis=0)  # (T, num_slots, num_patches)
    slots = np.stack(slots_collection, axis=0)  # (T, num_slots, D)
    patches = np.stack(patches_collection, axis=0)  # (T, num_patches, D)
    
    return attention_weights, slots, patches


def create_attention_video(attention_weights: np.ndarray, video_frames: torch.Tensor,
                          output_path: str = "slot_attention_viz.mp4", duration: float = 3.0):
    """
    Create video showing RGB frames alongside slot attention heatmaps.
    
    Args:
        attention_weights: (T, num_slots, num_patches) attention weights
        video_frames: (T, C, H, W) RGB video frames
        output_path: Output video file path
        duration: Target duration in seconds
    """
    num_frames, num_slots, num_patches = attention_weights.shape
    
    logging.info(f"Creating attention video: {num_frames} frames, {num_slots} slots, {num_patches} patches")
    
    # Calculate FPS
    target_fps = int(num_frames / duration)
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(20, 10), facecolor='#1a1a1a')
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Top row: RGB video (large, left) + Average attention heatmap (right)
    ax_video = fig.add_subplot(gs[0:2, 0:2])
    ax_avg_attn = fig.add_subplot(gs[0:2, 2:4])
    
    # Bottom row: Individual slot attention heatmaps (7 slots)
    ax_slots = []
    for i in range(num_slots):
        ax = fig.add_subplot(gs[2, i % 4] if i < 4 else gs[2, i % 4])
        ax_slots.append(ax)
    
    # Style axes
    ax_video.set_facecolor('#1a1a1a')
    ax_video.axis('off')
    ax_video.set_title('RGB Video', color='white', fontsize=14, fontweight='bold')
    
    ax_avg_attn.set_facecolor('#1a1a1a')
    ax_avg_attn.set_title('Average Attention (All Slots)', color='white', fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(ax_slots):
        ax.set_facecolor('#1a1a1a')
        ax.set_title(f'Slot {i+1}', color='white', fontsize=10, fontweight='bold')
    
    # Initialize plot elements
    video_im = ax_video.imshow(np.zeros((video_frames.shape[2], video_frames.shape[3], 3)))
    
    # Attention heatmaps need to be reshaped to spatial grid
    # Assuming patches are in spatial order (e.g., 14x14 grid for 196 patches)
    patch_h = patch_w = int(np.sqrt(num_patches))
    logging.info(f"Reshaping patches to {patch_h}x{patch_w} grid")
    
    # Initialize heatmaps
    avg_attn_im = ax_avg_attn.imshow(np.zeros((patch_h, patch_w)), 
                                      cmap='hot', vmin=0, vmax=1, interpolation='bilinear')
    ax_avg_attn.axis('off')
    plt.colorbar(avg_attn_im, ax=ax_avg_attn, fraction=0.046, pad=0.04)
    
    slot_attn_ims = []
    for i, ax in enumerate(ax_slots):
        im = ax.imshow(np.zeros((patch_h, patch_w)), 
                      cmap='viridis', vmin=0, vmax=1, interpolation='bilinear')
        ax.axis('off')
        slot_attn_ims.append(im)
    
    def update(frame_idx):
        # Update RGB video
        frame = video_frames[frame_idx].permute(1, 2, 0).cpu().numpy()
        if frame.max() > 1:
            frame = frame / 255.0
        frame = np.clip(frame, 0, 1)
        video_im.set_array(frame)
        
        # Get attention weights for current frame
        attn_frame = attention_weights[frame_idx]  # (num_slots, num_patches)
        
        # Average attention across all slots
        avg_attn = attn_frame.mean(axis=0).reshape(patch_h, patch_w)
        avg_attn_im.set_array(avg_attn)
        
        # Individual slot attention
        for slot_idx in range(num_slots):
            if slot_idx < len(slot_attn_ims):
                slot_attn = attn_frame[slot_idx].reshape(patch_h, patch_w)
                slot_attn_ims[slot_idx].set_array(slot_attn)
        
        # Add frame counter
        ax_video.set_xlabel(f'Frame: {frame_idx + 1}/{num_frames}', 
                           color='white', fontsize=12, fontweight='bold')
        
        return [video_im, avg_attn_im] + slot_attn_ims
    
    # Create animation
    logging.info("Generating attention visualization animation...")
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=1000 / target_fps,
        blit=False
    )
    
    # Save video
    logging.info(f"Saving video to {output_path}...")
    writer = animation.FFMpegWriter(fps=target_fps, bitrate=8000)
    anim.save(output_path, writer=writer, dpi=120)
    
    plt.close(fig)
    logging.info(f"Attention video saved: {output_path}")


def create_cross_slot_attention_matrix(attention_weights: np.ndarray, 
                                       output_path: str = "cross_slot_attention.mp4",
                                       duration: float = 3.0):
    """
    Create video showing cross-slot attention patterns over time.
    Computes slot-to-slot similarity based on their attention patterns.
    
    Args:
        attention_weights: (T, num_slots, num_patches) attention weights
        output_path: Output video file path
        duration: Target duration in seconds
    """
    num_frames, num_slots, num_patches = attention_weights.shape
    target_fps = int(num_frames / duration)
    
    logging.info("Creating cross-slot attention matrix video...")
    
    # Create figure
    fig, (ax_matrix, ax_time) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#1a1a1a')
    fig.suptitle('Slot-to-Slot Attention Pattern Similarity', 
                 color='white', fontsize=16, fontweight='bold')
    
    # Compute cross-slot similarity for all frames
    cross_slot_similarities = np.zeros((num_frames, num_slots, num_slots))
    for t in range(num_frames):
        attn_t = attention_weights[t]  # (num_slots, num_patches)
        # Cosine similarity between slot attention patterns
        norms = np.linalg.norm(attn_t, axis=1, keepdims=True) + 1e-8
        attn_normalized = attn_t / norms
        similarity = attn_normalized @ attn_normalized.T  # (num_slots, num_slots)
        cross_slot_similarities[t] = similarity
    
    # Initialize matrix heatmap
    matrix_im = ax_matrix.imshow(np.zeros((num_slots, num_slots)), 
                                 cmap='RdYlBu_r', vmin=-1, vmax=1, 
                                 interpolation='nearest')
    ax_matrix.set_xlabel('Slot Index', color='white', fontsize=12, fontweight='bold')
    ax_matrix.set_ylabel('Slot Index', color='white', fontsize=12, fontweight='bold')
    ax_matrix.set_title('Cross-Slot Attention Similarity', color='white', fontsize=14, fontweight='bold')
    ax_matrix.set_xticks(range(num_slots))
    ax_matrix.set_yticks(range(num_slots))
    ax_matrix.set_xticklabels([f'S{i+1}' for i in range(num_slots)], color='white')
    ax_matrix.set_yticklabels([f'S{i+1}' for i in range(num_slots)], color='white')
    ax_matrix.tick_params(colors='white')
    plt.colorbar(matrix_im, ax=ax_matrix, label='Similarity', 
                 orientation='vertical', pad=0.02)
    
    # Time series plot showing mean off-diagonal similarity (interaction strength)
    ax_time.set_facecolor('#2a2a2a')
    ax_time.set_xlabel('Time (Frame)', color='white', fontsize=12, fontweight='bold')
    ax_time.set_ylabel('Mean Cross-Slot Similarity', color='white', fontsize=12, fontweight='bold')
    ax_time.set_title('Slot Interaction Strength Over Time', color='white', fontsize=14, fontweight='bold')
    ax_time.tick_params(colors='white')
    ax_time.spines['bottom'].set_color('white')
    ax_time.spines['left'].set_color('white')
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)
    ax_time.grid(True, alpha=0.3, color='#666666')
    
    # Compute mean off-diagonal similarity over time (slot interaction)
    mean_cross_similarities = []
    for t in range(num_frames):
        sim_matrix = cross_slot_similarities[t]
        # Exclude diagonal (self-similarity)
        mask = ~np.eye(num_slots, dtype=bool)
        mean_off_diag = sim_matrix[mask].mean()
        mean_cross_similarities.append(mean_off_diag)
    
    time_line, = ax_time.plot([], [], color='cyan', linewidth=2, label='Mean Interaction')
    time_marker, = ax_time.plot([], [], 'o', color='red', markersize=10, 
                                markeredgecolor='white', markeredgewidth=2)
    ax_time.set_xlim(0, num_frames)
    ax_time.set_ylim(min(mean_cross_similarities) - 0.05, max(mean_cross_similarities) + 0.05)
    ax_time.legend(loc='upper right', facecolor='#2a2a2a', edgecolor='white', 
                   labelcolor='white', fontsize=10)
    
    plt.tight_layout()
    
    def update(frame_idx):
        # Update similarity matrix
        sim_matrix = cross_slot_similarities[frame_idx]
        matrix_im.set_array(sim_matrix)
        
        # Update time series
        time_line.set_data(range(frame_idx + 1), mean_cross_similarities[:frame_idx + 1])
        time_marker.set_data([frame_idx], [mean_cross_similarities[frame_idx]])
        
        # Add frame counter
        ax_matrix.text(0.02, 0.98, f'Frame: {frame_idx + 1}/{num_frames}',
                      transform=ax_matrix.transAxes, color='white', fontsize=11,
                      fontweight='bold', verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        return [matrix_im, time_line, time_marker]
    
    # Create animation
    logging.info("Generating cross-slot attention animation...")
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=1000 / target_fps,
        blit=False
    )
    
    # Save video
    logging.info(f"Saving cross-slot attention video to {output_path}...")
    writer = animation.FFMpegWriter(fps=target_fps, bitrate=6000)
    anim.save(output_path, writer=writer, dpi=120)
    
    plt.close(fig)
    logging.info(f"Cross-slot attention video saved: {output_path}")


def read_video():
    """Read video from file path."""
    video = VideoDecoder(VIDEO_PATH)
    return video[0: -1: NUM_FRAMESKIP].to(device)


def load_model_from_checkpoint(cfg):
    """Load OCWM model from checkpoint (same as slot_interaction_2.py)."""
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

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    ckpt_state = checkpoint.model.state_dict()
    missing, unexpected = world_model.load_state_dict(ckpt_state, strict=False)
    logging.info(f"Missing keys: {len(missing) if missing else 0}")
    logging.info(f"Unexpected keys: {len(unexpected) if unexpected else 0}")

    world_model.eval()
    return world_model


def visualize_slot_attention(cfg):
    """Main function to visualize slot attention."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    world_model = load_model_from_checkpoint(cfg)
    world_model = world_model.to(device)

    video_frames = read_video()
    transform = transforms.Compose([transforms.Resize((196, 196))])
    video_frames = torch.stack([transform(frame) for frame in video_frames], dim=0).to(device)

    logging.info(f"Video frames shape: {video_frames.shape}")

    # Take a subset of frames for visualization (e.g., 30 frames)
    num_viz_frames = min(30, len(video_frames))
    video_subset = video_frames[:num_viz_frames].unsqueeze(0)  # (1, T, C, H, W)

    logging.info(f"Extracting attention weights for {num_viz_frames} frames...")
    batch = {"pixels": video_subset}
    attention_weights, slots, patches = extract_attention_weights(world_model, batch)

    logging.info(f"Attention weights shape: {attention_weights.shape}")
    logging.info(f"Slots shape: {slots.shape}")
    logging.info(f"Patches shape: {patches.shape}")

    # Create output directory
    output_dir = Path(cfg.get('output_dir', './eval_results'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    attn_viz_path = output_dir / "slot_attention_viz.mp4"
    create_attention_video(attention_weights, video_frames[:num_viz_frames], 
                          output_path=str(attn_viz_path), duration=3.0)

    cross_attn_path = output_dir / "cross_slot_attention.mp4"
    create_cross_slot_attention_matrix(attention_weights, 
                                      output_path=str(cross_attn_path), duration=3.0)

    logging.info("✓ Slot attention visualization complete!")


@hydra.main(version_base=None, config_path="./", config_name="config_test_oc")
def run(cfg):
    """Entry point for slot attention visualization."""
    logging.info(f"Visualizing slot attention for {VIDEO_PATH}")
    visualize_slot_attention(cfg)


if __name__ == "__main__":
    run()
