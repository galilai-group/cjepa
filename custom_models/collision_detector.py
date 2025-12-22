import torch
import torch.nn as nn
import torch.nn.functional as F


class CollisionDetectorConv(nn.Module):
    """
    Temporal Convolution based collision detector.
    
    Takes a sequence of frame representations and outputs per-frame collision scores.
    Uses 1D convolutions along the temporal dimension to capture local dynamics.
    
    Args:
        num_patches: Number of patches (or slots) per frame
        dim: Dimension of each patch embedding
        hidden_dim: Hidden dimension for temporal processing
        kernel_size: Temporal convolution kernel size (default: 3)
    """
    
    def __init__(self, num_patches: int, dim: int, hidden_dim: int = 128, kernel_size: int = 3):
        super().__init__()
        
        self.num_patches = num_patches
        self.dim = dim
        
        # Pool patches into a single frame representation
        self.patch_pool = nn.Sequential(
            nn.Linear(num_patches * dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Temporal convolutions to capture dynamics
        padding = kernel_size // 2
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=kernel_size, padding=padding)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, P, D) - batch of frame sequences
               B: batch size
               T: number of frames
               P: number of patches/slots
               D: embedding dimension
               
        Returns:
            scores: (B, T) - per-frame collision scores (higher = more likely collision)
        """
        B, T, P, D = x.shape
        
        # Flatten patches: (B, T, P*D)
        x = x.view(B, T, P * D)
        
        # Pool to hidden dim: (B, T, hidden_dim)
        x = self.patch_pool(x)
        
        # Permute for conv1d: (B, hidden_dim, T)
        x = x.permute(0, 2, 1)
        
        # Temporal convolution: (B, 1, T)
        scores = self.temporal_conv(x)
        
        # Squeeze: (B, T)
        scores = scores.squeeze(1)
        
        return scores
    
    def predict_collisions(self, x: torch.Tensor, num_collisions: int) -> torch.Tensor:
        """
        Predict the top-k frames most likely to contain collisions.
        
        Args:
            x: (B, T, P, D) - batch of frame sequences
            num_collisions: Number of collision frames to return
            
        Returns:
            collision_frames: (B, num_collisions) - indices of predicted collision frames
        """
        scores = self.forward(x)  # (B, T)
        
        # Get top-k frame indices
        _, top_indices = torch.topk(scores, k=num_collisions, dim=1)
        
        return top_indices


class CollisionDetectorConvDiff(nn.Module):
    """
    Collision detector that explicitly models frame-to-frame differences.
    Combines learned difference weighting with temporal convolution.
    
    Args:
        num_patches: Number of patches (or slots) per frame
        dim: Dimension of each patch embedding
        hidden_dim: Hidden dimension for temporal processing
    """
    
    def __init__(self, num_patches: int, dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.num_patches = num_patches
        self.dim = dim
        
        # Learn which differences matter
        self.diff_encoder = nn.Sequential(
            nn.Linear(num_patches * dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Temporal conv on differences
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, P, D) - batch of frame sequences
               
        Returns:
            scores: (B, T) - per-frame collision scores
        """
        B, T, P, D = x.shape
        
        # Flatten patches: (B, T, P*D)
        x = x.view(B, T, P * D)
        
        # Compute frame-to-frame differences: (B, T-1, P*D)
        diff = x[:, 1:] - x[:, :-1]
        
        # Encode differences: (B, T-1, hidden_dim)
        diff_encoded = self.diff_encoder(diff.abs())
        
        # Permute for conv1d: (B, hidden_dim, T-1)
        diff_encoded = diff_encoded.permute(0, 2, 1)
        
        # Temporal convolution: (B, 1, T-1)
        scores = self.temporal_conv(diff_encoded).squeeze(1)
        
        # Pad first frame with zeros: (B, T)
        scores = F.pad(scores, (1, 0), value=scores.min().item())
        
        return scores
    
    def predict_collisions(self, x: torch.Tensor, num_collisions: int) -> torch.Tensor:
        """
        Predict the top-k frames most likely to contain collisions.
        """
        scores = self.forward(x)
        _, top_indices = torch.topk(scores, k=num_collisions, dim=1)
        return top_indices


def build_collision_detector(model_type: str = "conv", **kwargs) -> nn.Module:
    """
    Factory function to build collision detector.
    
    Args:
        model_type: "conv" or "conv_diff"
        **kwargs: Arguments passed to the model constructor
        
    Returns:
        Collision detector model
    """
    if model_type == "conv":
        return CollisionDetectorConv(**kwargs)
    elif model_type == "conv_diff":
        return CollisionDetectorConvDiff(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
