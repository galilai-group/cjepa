"""
Interaction Dynamics Visualization: Vector Fields and Temporal Similarity.

This module implements advanced dynamics visualization methods:
  1. Vector Field of Future Prediction (JEPA-style flow visualization)
  2. Temporal Self-Similarity Matrix (showing interaction-induced structural changes)

Both highlight how the model captures transient interaction events.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger as logging


class Arrow3D(FancyArrowPatch):
    """3D arrow patch for quiver-style visualization in 3D space."""
    
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 1), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        
        # Project to 2D
        proj = proj3d.proj_transform((x1, x1+dx), (y1, y1+dy), (z1, z1+dz), self.axes.M)
        self.set_positions((proj[0][0], proj[1][0]), (proj[0][1], proj[1][1]))
        
        FancyArrowPatch.draw(self, renderer)
        self._children = []

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        
        proj = proj3d.proj_transform((x1, x1+dx), (y1, y1+dy), (z1, z1+dz), self.axes.M)
        self.set_positions((proj[0][0], proj[1][0]), (proj[0][1], proj[1][1]))
        
        return np.min(proj[2])


def plot_vector_field_prediction(
    embeddings: torch.Tensor,
    predictions: torch.Tensor,
    output_path: Path,
    video_name: str = "video",
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    Visualize the model's predicted flow in latent space.
    
    **JEPA Concept - "Vector Field of Future":**
      For each latent state z_t, the model predicts z_{t+1}.
      The vector (z_{t+1} - z_t) represents the model's learned dynamics.
    
    **Visualization:**
      - Project to 2D PCA plane.
      - Draw arrows from z_t to z_{t+1}.
      - Arrow color: velocity magnitude (prediction confidence/uncertainty).
      - Arrow length: magnitude of state change (prediction jump).
    
    **Interpretation Patterns:**
      - Laminar flow (parallel arrows): predictable, smooth motion.
      - Turbulent/diverging arrows: uncertain regions (often interaction events).
      - Large arrows: high-magnitude state transitions.
    
    **Physics Insight:**
      "In stable regions, the vector field shows laminar flow - all points move
       similarly, indicating the model has learned a smooth attractor.
       During interactions, the field becomes turbulent or chaotic, showing
       the model recognizes difficult-to-predict state changes."
    
    Args:
        embeddings: Latent states z_t, shape (T, D).
        predictions: Predicted next states z'_{t+1}, shape (T, D).
        output_path: Directory to save figures.
        video_name: Name for the output file.
        figsize: Figure size.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    predictions_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    
    # Project to 2D using PCA
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings_np)
    pred_2d = pca.transform(predictions_np)
    
    # Compute flow vectors
    flow = pred_2d - emb_2d  # (T, 2)
    flow_magnitude = np.linalg.norm(flow, axis=1)
    flow_magnitude_norm = (flow_magnitude - flow_magnitude.min()) / (flow_magnitude.max() - flow_magnitude.min() + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot trajectory as background
    ax.plot(emb_2d[:, 0], emb_2d[:, 1], 'k-', alpha=0.2, linewidth=0.5, label='Trajectory')
    
    # Sample points for better visibility
    sample_rate = max(1, len(emb_2d) // 50)  # ~50 arrows for clarity
    indices = np.arange(0, len(emb_2d), sample_rate)
    
    # Plot arrows with color based on magnitude
    for idx in indices:
        x, y = emb_2d[idx]
        dx, dy = flow[idx]
        
        color_val = flow_magnitude_norm[idx]
        color = plt.cm.Spectral_r(color_val)  # Red (high) to Blue (low)
        
        ax.arrow(
            x, y, dx, dy,
            head_width=0.05,
            head_length=0.05,
            fc=color,
            ec='black',
            linewidth=1.0,
            alpha=0.7,
        )
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Prediction Magnitude (State Change)', shrink=0.8)
    
    # Scatter current and predicted positions
    scatter1 = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c='blue', s=30, alpha=0.5, label='Current State z_t')
    scatter2 = ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c='green', s=30, alpha=0.5, label='Predicted z\'_{t+1}')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=11, fontweight='bold')
    ax.set_title(
        f'Vector Field of Future Predictions: {video_name}\n(Red=Large Changes, Blue=Small Changes)',
        fontsize=13,
        fontweight='bold',
        pad=20,
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_file = output_path / f"{video_name}_vector_field.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"Saved vector field plot: {output_file}")
    plt.close()


def plot_temporal_similarity_matrix(
    embeddings: torch.Tensor,
    output_path: Path,
    video_name: str = "video",
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """
    Plot temporal self-similarity matrix: M[i,j] = Similarity(z_i, z_j).
    
    **Concept - "Self-Similarity Structure":**
      A similarity matrix reveals the temporal structure of the latent space.
      - Diagonal band: smooth temporal progression.
      - Checkerboard pattern or blocks: phase changes (interaction events).
    
    **Pattern Interpretation:**
      - Strong diagonal (repeated similar states): stable attractor regions.
      - Block structure breaking: sudden state transitions (interactions).
      - Off-diagonal blocks: repeated motion patterns or periodic behavior.
    
    **Physics Insight:**
      "Similar latent states cluster together. When interactions occur,
       the state jumps to a new region of latent space.
       The matrix's structure visualizes these state space partitions."
    
    Args:
        embeddings: Latent states z_t, shape (T, D).
        output_path: Directory to save figures.
        video_name: Name for the output file.
        figsize: Figure size.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    
    # Compute cosine similarity matrix
    # Normalize embeddings
    emb_norm = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity (dot product of normalized vectors)
    similarity_matrix = np.dot(emb_norm, emb_norm.T)  # (T, T)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot similarity matrix
    im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto', origin='lower')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    # Labels and title
    ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time Step', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Temporal Self-Similarity Matrix: {video_name}\n(Blocks=Phases, Diagonal=Continuity)',
        fontsize=13,
        fontweight='bold',
        pad=20,
    )
    
    # Add diagonal reference line
    T = len(embeddings)
    ax.plot([0, T], [0, T], 'r--', linewidth=1.5, alpha=0.5, label='Temporal progression')
    
    # Highlight interaction regions by computing temporal gradient
    # High gradient along diagonal = sudden state change
    diag_diff = np.abs(np.diff(np.diag(similarity_matrix)))
    
    # Find indices with large changes
    threshold = np.percentile(diag_diff, 90)
    change_indices = np.where(diag_diff > threshold)[0]
    
    if len(change_indices) > 0:
        for idx in change_indices[:5]:  # Highlight top 5 interactions
            ax.axvline(x=idx, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax.axhline(y=idx, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax.grid(False)
    plt.tight_layout()
    output_file = output_path / f"{video_name}_similarity_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"Saved similarity matrix plot: {output_file}")
    plt.close()


def plot_temporal_difference_heatmap(
    embeddings: torch.Tensor,
    output_path: Path,
    video_name: str = "video",
    window_size: int = 5,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot temporal difference heatmap: D[i,j] = ||z_i - z_j||.
    
    **Concept:**
      Shows pairwise distances between latent states over time.
      Unlike similarity matrices, this directly shows state divergence.
    
    **Pattern:**
      - Blue diagonal: most similar states are adjacent (temporal continuity).
      - Yellow/red blocks off-diagonal: states that are far apart in latent space
        but not adjacent in time (revisited states, cycle patterns).
      - Sudden boundary changes: interaction-induced state transitions.
    
    Args:
        embeddings: Latent states z_t, shape (T, D).
        output_path: Directory to save figures.
        video_name: Name for the output file.
        window_size: If set, smooth with this window size for noise reduction.
        figsize: Figure size.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    
    # Compute pairwise L2 distance matrix
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(embeddings_np, metric='euclidean'))
    
    # Optional: smooth with median filter for noise reduction
    if window_size > 1:
        from scipy.ndimage import median_filter
        distances = median_filter(distances, size=window_size)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(distances, cmap='YlOrRd', aspect='auto', origin='lower')
    cbar = plt.colorbar(im, ax=ax, label='L2 Distance in Latent Space')
    
    ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time Step', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Temporal Distance Matrix: {video_name}\n(Blue=Continuous, Yellow/Red=Transitions)',
        fontsize=13,
        fontweight='bold',
        pad=20,
    )
    
    # Highlight main diagonal (should be small)
    T = len(embeddings)
    ax.plot([0, T], [0, T], 'c--', linewidth=2, alpha=0.7, label='Temporal order')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    output_file = output_path / f"{video_name}_distance_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"Saved distance matrix plot: {output_file}")
    plt.close()
