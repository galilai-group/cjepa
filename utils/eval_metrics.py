import torch
import torch.nn.functional as F
from typing import Union
import numpy as np


def rankme(embeddings: Union[torch.Tensor, np.ndarray]) -> float:
    """
    RankMe: Effective rank metric based on SVD entropy.
    
    Measures the effective number of learned dimensions in the embedding space.
    Stable estimation requires: num_samples >> embedding_dimension.
    
    Args:
        embeddings: Tensor of shape (N, D) where N is num_samples, D is embedding_dim.
                   N should be >= 512 for stable estimation (preferably > D).
    
    Returns:
        rankme_score (float): Exponential of normalized entropy.
                             Range: [1, embedding_dim]
                             Higher is better (more diverse, full-rank representations).
                             Value of D means perfectly rank-full embedding space.
        """
    with torch.no_grad():
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        else:
            embeddings = embeddings.float()
        
        # Ensure 2D
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        # Center the embeddings
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Compute singular values
        s = torch.linalg.svdvals(embeddings)
        
        # Normalize to probability distribution (L2 norm for SVD)
        s = s[s > 1e-10]  # Remove near-zero singular values
        p = s / (torch.sum(s) + 1e-10)
        
        # Compute entropy
        epsilon = 1e-10
        entropy = -torch.sum(p * torch.log(p + epsilon))
        
        # RankMe score: exponential of entropy
        rankme_score = torch.exp(entropy).item()
    
    return rankme_score


def frechet_joint_distance(embeddings_real: Union[torch.Tensor, np.ndarray],
                          embeddings_pred: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Fréchet Distance in Embedding Space: Compare temporal distribution of real vs predicted embeddings.
    
    Measures how well the predictor captures the distribution of future embeddings.
    Lower values indicate better prediction accuracy.
    
    Args:
        embeddings_real: Real future embeddings, shape (N, D).
        embeddings_pred: Predicted future embeddings, shape (N, D).
    
    Returns:
        frechet_dist (float): Fréchet distance between distributions.
                             Range: [0, ∞)
                             Lower is better. Value 0 means distributions are identical.
    
    Formula: FD = ||μ_real - μ_pred||_2 + Tr(Σ_real + Σ_pred - 2(Σ_real Σ_pred)^0.5)
             This is the Fréchet distance assuming Gaussians.
    """
    with torch.no_grad():
        if isinstance(embeddings_real, np.ndarray):
            embeddings_real = torch.tensor(embeddings_real, dtype=torch.float32)
        if isinstance(embeddings_pred, np.ndarray):
            embeddings_pred = torch.tensor(embeddings_pred, dtype=torch.float32)
        
        embeddings_real = embeddings_real.float()
        embeddings_pred = embeddings_pred.float()
        
        # Compute means
        mu_real = embeddings_real.mean(dim=0)
        mu_pred = embeddings_pred.mean(dim=0)
        
        # Compute covariances
        cov_real = torch.cov(embeddings_real.T)
        cov_pred = torch.cov(embeddings_pred.T)
        
        # Mean difference term
        mean_diff = torch.norm(mu_real - mu_pred)
        
        # Covariance trace term
        trace_real = torch.trace(cov_real)
        trace_pred = torch.trace(cov_pred)
        
        # Compute sqrt of product of covariances
        try:
            # Matrix square root via eigendecomposition
            eigvals_real, eigvecs_real = torch.linalg.eigh(cov_real)
            eigvals_pred, eigvecs_pred = torch.linalg.eigh(cov_pred)
            
            # Ensure non-negative eigenvalues
            eigvals_real = torch.clamp(eigvals_real, min=0)
            eigvals_pred = torch.clamp(eigvals_pred, min=0)
            
            sqrt_cov_real = eigvecs_real @ torch.diag(torch.sqrt(eigvals_real)) @ eigvecs_real.T
            sqrt_cov_prod = sqrt_cov_real @ cov_pred @ sqrt_cov_real
            
            eigvals_prod, _ = torch.linalg.eigh(sqrt_cov_prod)
            eigvals_prod = torch.clamp(eigvals_prod, min=0)
            trace_prod = torch.sum(torch.sqrt(eigvals_prod))
        except:
            # Fallback if eigendecomposition fails
            trace_prod = 0
        
        frechet_dist = (mean_diff ** 2 + trace_real + trace_pred - 2 * trace_prod).item()
        frechet_dist = max(0, frechet_dist)  # Ensure non-negative
    
    return float(frechet_dist)


def feature_rollout_degradation(embeddings_history: list,
                                ground_truth_embeddings: list) -> dict:
    """
    Feature Rollout Degradation: Measure how prediction accuracy degrades over time.
    
    Predicts multiple steps ahead and compares embeddings at each step to ground truth.
    Shows how the world model's latent predictions diverge from reality.
    
    Args:
        embeddings_history: List of predicted embeddings for steps [0, 1, 2, ..., K].
                          Each element is shape (N, D).
        ground_truth_embeddings: List of real embeddings for corresponding steps.
                                Same structure as embeddings_history.
    
    Returns:
        degradation_report (dict): 
            - 'mse_per_step': List of MSE values at each rollout step.
                             Range: [0, ∞)
                             Lower is better. Perfect prediction = 0.
            - 'cosine_sim_per_step': List of cosine similarity at each step.
                                    Range: [-1, 1], typically [0, 1] for embeddings.
                                    Higher is better. Value 1 means identical direction.
            - 'degradation_rate': (MSE_final - MSE_initial) / MSE_initial.
                                 Range: [0, ∞)
                                 Lower is better (slower divergence).
    
    Interpretation:
        - Fast MSE growth → model diverges quickly from reality.
        - MSE plateau → model settles into local pattern (possibly mode collapse).
        - High cosine similarity → predictions remain directionally correct.
    """
    with torch.no_grad():
        mse_per_step = []
        cosine_sim_per_step = []
        
        for pred_emb, gt_emb in zip(embeddings_history, ground_truth_embeddings):
            if isinstance(pred_emb, np.ndarray):
                pred_emb = torch.tensor(pred_emb, dtype=torch.float32)
            if isinstance(gt_emb, np.ndarray):
                gt_emb = torch.tensor(gt_emb, dtype=torch.float32)
            
            pred_emb = pred_emb.float()
            gt_emb = gt_emb.float()
            
            # MSE between predictions and ground truth
            mse = F.mse_loss(pred_emb, gt_emb).item()
            mse_per_step.append(mse)
            
            # Cosine similarity (average across batch)
            # Normalize embeddings
            pred_norm = F.normalize(pred_emb, p=2, dim=-1)
            gt_norm = F.normalize(gt_emb, p=2, dim=-1)
            
            # Cosine similarity
            cos_sim = torch.sum(pred_norm * gt_norm, dim=-1).mean().item()
            cosine_sim_per_step.append(cos_sim)
        
        # Compute degradation rate
        if len(mse_per_step) > 1:
            degradation_rate = (mse_per_step[-1] - mse_per_step[0]) / (mse_per_step[0] + 1e-10)
        else:
            degradation_rate = 0
        
        report = {
            'mse_per_step': mse_per_step,
            'cosine_sim_per_step': cosine_sim_per_step,
            'degradation_rate': degradation_rate,
        }
    
    return report



