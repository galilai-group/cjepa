"""
Latent Space Trajectory Visualization using Physics-Inspired Concepts.

This module implements two powerful dynamics visualization methods:
  1. Latent Energy Trajectory: 3D PCA trajectory with velocity coloring
  2. Phase Space Plot: State-Space dynamics showing stability and transitions

Both methods highlight interaction events through rapid latent changes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Tuple, Dict, Optional
from loguru import logger as logging


def compute_temporal_velocity(embeddings: torch.Tensor) -> np.ndarray:
    """
    Compute temporal velocity (magnitude of latent change).
    
    Args:
        embeddings: Tensor of shape (T, D) representing latent trajectory.
        
    Returns:
        velocity: Array of shape (T,) where velocity[t] = ||z_{t+1} - z_t||_2.
                 First element is set to velocity[1] for consistency.
    
    Interpretation:
        - Low velocity (blue): stable regions with minimal latent change.
        - High velocity (red): interaction events with rapid latent transitions.
    """
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    
    # Compute pairwise differences
    diffs = np.diff(embeddings_np, axis=0)  # (T-1, D)
    velocity = np.linalg.norm(diffs, axis=1)  # (T-1,)
    
    # Pad with first velocity for T-length output
    velocity = np.concatenate([[velocity[0]], velocity])
    
    return velocity


def fit_pca_3d(embeddings: torch.Tensor) -> Tuple[np.ndarray, PCA]:
    """
    Project high-dimensional embeddings to 3D using PCA.
    
    PCA preserves temporal continuity better than t-SNE/UMAP,
    making it ideal for trajectory visualization.
    
    Args:
        embeddings: Tensor of shape (T, D).
        
    Returns:
        trajectory_3d: Array of shape (T, 3) - the 3D trajectory.
        pca: Fitted PCA object for reference.
    """
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    
    pca = PCA(n_components=3)
    trajectory_3d = pca.fit_transform(embeddings_np)
    
    logging.info(f"PCA: {pca.explained_variance_ratio_[:3]} = {pca.explained_variance_ratio_[:3].sum():.4f} variance explained")
    
    return trajectory_3d, pca


def plot_latent_energy_trajectory(
    embeddings: torch.Tensor,
    output_path: Path,
    video_name: str = "video",
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    Plot 3D latent trajectory with velocity-based coloring.
    
    **Visualization Concept:**
      - X, Y, Z: First three principal components from PCA.
      - Color: Latent velocity (||z_{t+1} - z_t||) indicating interaction intensity.
      - Blue regions: stable motion with smooth latent evolution.
      - Red regions: interaction events with rapid latent transitions.
    
    **Interpretation:**
      "The model maintains stable representations during passive motion (blue segments),
       but exhibits sharp latent shifts during physical interactions like collisions (red spikes).
       This demonstrates sensitivity to dynamics and state changes."
    
    Args:
        embeddings: Tensor of shape (T, D) - full video latent trajectory.
        output_path: Directory to save figures.
        video_name: Name for the output file.
        figsize: Figure size (width, height).
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Compute trajectory and velocity
    trajectory_3d, pca = fit_pca_3d(embeddings)
    velocity = compute_temporal_velocity(embeddings)
    
    # Normalize velocity for coloring [0, 1]
    velocity_norm = (velocity - velocity.min()) / (velocity.max() - velocity.min() + 1e-8)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory segments with velocity coloring
    num_points = len(trajectory_3d)
    for i in range(num_points - 1):
        color_value = velocity_norm[i]
        # Blue (low velocity) to Red (high velocity)
        color = plt.cm.cool(1 - color_value)  # cool colormap: blue->cyan->green
        ax.plot(
            trajectory_3d[i:i+2, 0],
            trajectory_3d[i:i+2, 1],
            trajectory_3d[i:i+2, 2],
            color=color,
            linewidth=2.5,
            alpha=0.9,
        )
    
    # Add scatter points for better visibility, sized by velocity
    size_base = 30
    sizes = 20 + size_base * velocity_norm
    scatter = ax.scatter(
        trajectory_3d[:, 0],
        trajectory_3d[:, 1],
        trajectory_3d[:, 2],
        c=velocity_norm,
        s=sizes,
        cmap='cool',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5,
    )
    
    # Colorbar for velocity
    cbar = plt.colorbar(scatter, ax=ax, label='Latent Velocity (Interaction Intensity)', shrink=0.8)
    
    # Labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=11, fontweight='bold')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=11, fontweight='bold')
    ax.set_title(
        f'Latent Energy Trajectory: {video_name}\n(Blue=Stable, Red=Interaction Events)',
        fontsize=13,
        fontweight='bold',
        pad=20,
    )
    
    # Improve visibility
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / f"{video_name}_latent_trajectory_3d.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"Saved latent trajectory plot: {output_file}")
    plt.close()


def plot_phase_space(
    embeddings: torch.Tensor,
    output_path: Path,
    video_name: str = "video",
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot Phase Space: State Norm vs Temporal Derivative.
    
    **Physics Concept:**
      This is a "Phase Space Diagram" - a fundamental tool in dynamical systems.
      - X-axis: State Norm (||z_t||) = current energy/complexity.
      - Y-axis: Temporal Derivative (||z_{t+1} - z_t||) = rate of change.
    
    **Visualization Patterns:**
      - Stable regions: Small loops around an "attractor" (stable state).
      - Interaction events: Excursions away from attractor, showing state transitions.
      - Relaxation: System returns to stable state after disturbance (closing loops).
    
    **Interpretation:**
      "During normal video progression, the system oscillates in a stable attractor region.
       When physical interactions occur (t=45-50), the system is 'kicked' to a high-energy state.
       The trajectory's return to the attractor demonstrates model awareness of transient events."
    
    Args:
        embeddings: Tensor of shape (T, D).
        output_path: Directory to save figures.
        video_name: Name for the output file.
        figsize: Figure size.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    
    # Compute metrics
    state_norm = np.linalg.norm(embeddings_np, axis=1)  # (T,)
    velocity = compute_temporal_velocity(embeddings)  # (T,)
    time_indices = np.arange(len(embeddings))
    
    # Normalize for consistent coloring
    time_norm = (time_indices - time_indices.min()) / (time_indices.max() - time_indices.min() + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot phase trajectory with time coloring
    scatter = ax.scatter(
        state_norm,
        velocity,
        c=time_norm,
        s=50,
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5,
    )
    
    # Connect consecutive points with lines
    for i in range(len(state_norm) - 1):
        ax.plot(
            state_norm[i:i+2],
            velocity[i:i+2],
            color=plt.cm.viridis(time_norm[i]),
            linewidth=1.5,
            alpha=0.5,
        )
    
    # Colorbar for time progression
    cbar = plt.colorbar(scatter, ax=ax, label='Time Progression (Normalized)', shrink=0.8)
    
    # Labels and title
    ax.set_xlabel('State Norm (||z_t||)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temporal Derivative (||z_{t+1} - z_t||)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Phase Space Diagram: {video_name}\n(Attractors=Stable Regions, Excursions=Interactions)',
        fontsize=13,
        fontweight='bold',
        pad=20,
    )
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Annotate extrema for reference
    max_vel_idx = np.argmax(velocity)
    ax.annotate(
        f'Max interaction\n(t={max_vel_idx})',
        xy=(state_norm[max_vel_idx], velocity[max_vel_idx]),
        xytext=(10, 20),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'),
        fontsize=10,
    )
    
    plt.tight_layout()
    output_file = output_path / f"{video_name}_phase_space.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"Saved phase space plot: {output_file}")
    plt.close()


def plot_slot_interaction_trajectory(
    slot_embeddings_list: list,
    output_path: Path,
    video_name: str = "video",
    slot_names: Optional[list] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    Plot multiple object trajectories in 3D latent space (Disentangled Slot Interaction).
    
    **Concept - "Latent Billiards":**
      When objects interact physically, their latent trajectories should show:
      - Approach: trajectories converge toward each other.
      - Interaction: trajectories overlap or touch.
      - Bounce: trajectories diverge after interaction.
    
    This demonstrates that the model captures object-level dynamics, not just scene-level.
    
    Args:
        slot_embeddings_list: List of tensors, each shape (T, D) for one object.
        output_path: Directory to save figures.
        video_name: Name for the output file.
        slot_names: List of names for each slot (e.g., ['Ball', 'Cube']).
        figsize: Figure size.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if slot_names is None:
        slot_names = [f"Slot {i}" for i in range(len(slot_embeddings_list))]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(slot_embeddings_list)))
    
    # Project each slot's trajectory to 3D
    for slot_idx, (embeddings, name, color) in enumerate(zip(slot_embeddings_list, slot_names, colors)):
        trajectory_3d, _ = fit_pca_3d(embeddings)
        
        # Plot trajectory
        ax.plot(
            trajectory_3d[:, 0],
            trajectory_3d[:, 1],
            trajectory_3d[:, 2],
            color=color,
            linewidth=2.5,
            label=name,
            alpha=0.8,
        )
        
        # Scatter for start and end points
        ax.scatter(
            [trajectory_3d[0, 0]],
            [trajectory_3d[0, 1]],
            [trajectory_3d[0, 2]],
            color=color,
            s=150,
            marker='o',
            edgecolors='black',
            linewidth=2,
            alpha=0.9,
        )
        ax.scatter(
            [trajectory_3d[-1, 0]],
            [trajectory_3d[-1, 1]],
            [trajectory_3d[-1, 2]],
            color=color,
            s=150,
            marker='X',
            edgecolors='black',
            linewidth=2,
            alpha=0.9,
        )
    
    ax.set_xlabel('PC1', fontsize=11, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
    ax.set_zlabel('PC3', fontsize=11, fontweight='bold')
    ax.set_title(
        f'Slot Interaction Dynamics: {video_name}\n(Circles=Start, X=End)',
        fontsize=13,
        fontweight='bold',
        pad=20,
    )
    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / f"{video_name}_slot_interaction_3d.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"Saved slot interaction plot: {output_file}")
    plt.close()
