import torch
from loguru import logger as logging
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


# ============================================================================
# Hungarian Matching Loss
# ============================================================================
def hungarian_matching_loss_AP(
    pred: torch.Tensor,
    target: torch.Tensor,
    cost_type: str = "mse",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute per-time-step Hungarian matching loss for slot-based predictions.
    
    For each time step independently, finds the optimal bipartite matching
    between predicted slots and target slots, then computes the loss using
    matched pairs. This handles slot permutation invariance.
    
    Args:
        pred: Predicted slot embeddings (B, T, N, D)
              - B: batch size
              - T: number of time steps (prediction horizon)
              - N: number of slots per frame
              - D: slot embedding dimension
        target: Target slot embeddings (B, T, N, D)
        cost_type: Type of cost for matching, "mse" or "cosine"
        reduction: "mean" or "sum" for final loss aggregation
    
    Returns:
        loss: Scalar loss suitable for backpropagation
    
    Algorithm:
        1. For each (batch, time) pair:
           - Build cost matrix C[i,j] = ||pred[i] - target[j]||^2
           - Run Hungarian algorithm to find optimal permutation π
           - Compute loss using matched pairs
        2. Average (or sum) over all (B, T) pairs
    """
    B, T, N, D = pred.shape
    assert target.shape == pred.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
    
    device = pred.device
    total_loss = torch.tensor(0.0, device=device, dtype=pred.dtype)
    
    # Detach for cost matrix computation (Hungarian is non-differentiable)
    pred_detached = pred.detach()
    target_detached = target.detach()
    
    for b in range(B):
        for t in range(T):
            # Extract slots at this (batch, time) position
            pred_slots = pred[b, t]           # (N, D)
            target_slots = target[b, t]       # (N, D)
            pred_slots_det = pred_detached[b, t]
            target_slots_det = target_detached[b, t]
            
            # Build cost matrix C[i,j] = cost(pred[i], target[j])
            if cost_type == "mse":
                # C[i,j] = ||pred[i] - target[j]||^2
                # Expand for pairwise computation: (N, 1, D) - (1, N, D) = (N, N, D)
                diff = pred_slots_det.unsqueeze(1) - target_slots_det.unsqueeze(0)  # (N, N, D)
                cost_matrix = (diff ** 2).sum(dim=-1)  # (N, N)
            elif cost_type == "cosine":
                # C[i,j] = 1 - cosine_similarity(pred[i], target[j])
                pred_norm = F.normalize(pred_slots_det, dim=-1)  # (N, D)
                target_norm = F.normalize(target_slots_det, dim=-1)  # (N, D)
                cosine_sim = torch.mm(pred_norm, target_norm.t())  # (N, N)
                cost_matrix = 1.0 - cosine_sim
            else:
                raise ValueError(f"Unknown cost_type: {cost_type}")
            
            # Run Hungarian algorithm to find optimal matching
            # Returns (row_indices, col_indices) where row i is matched to col[i]
            cost_np = cost_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            
            # col_ind[i] gives the target index that should be matched to pred index i
            # So we permute target according to col_ind
            matched_target = target_slots[col_ind]  # (N, D)
            
            # Compute loss on matched pairs (this is differentiable w.r.t. pred)
            if cost_type == "mse":
                pair_loss = F.mse_loss(pred_slots, matched_target, reduction='sum')
            elif cost_type == "cosine":
                # For cosine, we use 1 - cosine_similarity as loss
                cos_sim = F.cosine_similarity(pred_slots, matched_target, dim=-1)  # (N,)
                pair_loss = (1.0 - cos_sim).sum()
            
            total_loss = total_loss + pair_loss
    
    # Reduction
    if reduction == "mean":
        total_loss = total_loss / (B * T * N)
    elif reduction == "sum":
        pass  # Already summed
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    

    return {"pixels_loss": total_loss}



def hungarian_matching_loss_with_proprio(
    pred: torch.Tensor,
    target: torch.Tensor,
    pixels_dim: int,
    proprio_dim: int = 0,
    cost_type: str = "mse",
    reduction: str = "mean",
) -> dict:
    """
    Hungarian matching loss with separate losses for pixels and proprio.
    
    Matching is done based on pixels (slot embeddings) only,
    but loss is computed for both pixels and proprio using the same matching.
    
    Args:
        pred: (B, T, N, D) where D = pixels_dim + proprio_dim + action_dim
        target: (B, T, N, D)
        pixels_dim: Dimension of pixel/slot embedding
        proprio_dim: Dimension of proprio embedding (0 if not used)
        cost_type: "mse" or "cosine"
        reduction: "mean" or "sum"
    
    Returns:
        dict with "pixels_loss", "proprio_loss" (if proprio_dim > 0), and "total_loss"
    """
    B, T, N, D = pred.shape
    assert target.shape == pred.shape
    
    device = pred.device
    total_pixels_loss = torch.tensor(0.0, device=device, dtype=pred.dtype)
    total_proprio_loss = torch.tensor(0.0, device=device, dtype=pred.dtype)
    
    # Extract pixels part for matching
    pred_pixels = pred[..., :pixels_dim]           # (B, T, N, pixels_dim)
    target_pixels = target[..., :pixels_dim]
    
    pred_pixels_det = pred_pixels.detach()
    target_pixels_det = target_pixels.detach()
    
    # Extract proprio part if available
    if proprio_dim > 0:
        pred_proprio = pred[..., pixels_dim:pixels_dim + proprio_dim]
        target_proprio = target[..., pixels_dim:pixels_dim + proprio_dim]
    
    for b in range(B):
        for t in range(T):
            # Build cost matrix using ONLY pixels (slot embeddings)
            pred_slots = pred_pixels_det[b, t]      # (N, pixels_dim)
            target_slots = target_pixels_det[b, t]  # (N, pixels_dim)
            
            if cost_type == "mse":
                diff = pred_slots.unsqueeze(1) - target_slots.unsqueeze(0)
                cost_matrix = (diff ** 2).sum(dim=-1)
            elif cost_type == "cosine":
                pred_norm = F.normalize(pred_slots, dim=-1)
                target_norm = F.normalize(target_slots, dim=-1)
                cost_matrix = 1.0 - torch.mm(pred_norm, target_norm.t())
            else:
                raise ValueError(f"Unknown cost_type: {cost_type}")
            
            # Hungarian matching
            cost_np = cost_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            
            # Apply matching to pixels
            matched_target_pixels = target_pixels[b, t, col_ind]
            pixels_loss = F.mse_loss(pred_pixels[b, t], matched_target_pixels, reduction='sum')
            total_pixels_loss = total_pixels_loss + pixels_loss
            
            # Apply same matching to proprio
            if proprio_dim > 0:
                matched_target_proprio = target_proprio[b, t, col_ind]
                proprio_loss = F.mse_loss(pred_proprio[b, t], matched_target_proprio, reduction='sum')
                total_proprio_loss = total_proprio_loss + proprio_loss
    
    # Reduction
    num_elements_pixels = B * T * N * pixels_dim
    num_elements_proprio = B * T * N * proprio_dim if proprio_dim > 0 else 1
    
    if reduction == "mean":
        total_pixels_loss = total_pixels_loss / num_elements_pixels
        if proprio_dim > 0:
            total_proprio_loss = total_proprio_loss / num_elements_proprio
    
    result = {
        "pixels_loss": total_pixels_loss,
        "total_loss": total_pixels_loss,
    }
    
    if proprio_dim > 0:
        result["proprio_loss"] = total_proprio_loss
        result["total_loss"] = total_pixels_loss + total_proprio_loss
    
    return result


# ============================================================================
# Hungarian Matching for Planning (Cost Computation)
# ============================================================================
def hungarian_cost(
    preds: torch.Tensor,
    goal: torch.Tensor,
    cost_type: str = "mse",
    pixels_dim: int = None,
) -> torch.Tensor:
    """
    Compute per-sample Hungarian matched cost between predictions and goal.
    
    Used for planning: computes the minimum-cost matching between predicted
    slots and goal slots for each sample independently.
    Matching is based on pixels_dim only (if specified).
    
    Args:
        preds: Predicted slot embeddings (B, N, S, D)
               - B: batch size (num action candidates)
               - N: num candidates (usually 1 for this use case)
               - S: number of slots
               - D: slot embedding dimension
        goal: Goal slot embeddings (B, N, S, D)
        cost_type: "mse" or "cosine"
        pixels_dim: If specified, use only first pixels_dim dimensions for matching and cost.
                   Otherwise use full D.
    
    Returns:
        cost: (B, N) tensor of matched costs per sample
    """
    # Handle shapes - preds and goal should be (B, N, S, D) or (B, N, T, S, D)
    # For criterion, we usually pass preds[:, :, -1:] which is (B, N, 1, S, D)
    
    if preds.ndim == 5:
        # (B, N, T, S, D) -> take last time step
        preds = preds[:, :, -1]  # (B, N, S, D)
        goal = goal[:, :, -1]    # (B, N, S, D)
    
    B, N, S, D = preds.shape
    assert goal.shape == preds.shape, f"Shape mismatch: preds {preds.shape} vs goal {goal.shape}"
    
    # Determine matching dimension
    match_dim = pixels_dim if pixels_dim is not None else D
    
    device = preds.device
    costs = torch.zeros(B, N, device=device, dtype=preds.dtype)
    
    preds_det = preds.detach()
    goal_det = goal.detach()
    
    for b in range(B):
        for n in range(N):
            pred_slots_full = preds_det[b, n]  # (S, D)
            goal_slots_full = goal_det[b, n]   # (S, D)
            
            # Use only pixels_dim for matching
            pred_slots = pred_slots_full[:, :match_dim]  # (S, match_dim)
            goal_slots = goal_slots_full[:, :match_dim]  # (S, match_dim)
            
            # Build cost matrix using only pixels portion
            if cost_type == "mse":
                diff = pred_slots.unsqueeze(1) - goal_slots.unsqueeze(0)  # (S, S, match_dim)
                cost_matrix = (diff ** 2).sum(dim=-1)  # (S, S)
            elif cost_type == "cosine":
                pred_norm = F.normalize(pred_slots, dim=-1)
                goal_norm = F.normalize(goal_slots, dim=-1)
                cost_matrix = 1.0 - torch.mm(pred_norm, goal_norm.t())
            else:
                raise ValueError(f"Unknown cost_type: {cost_type}")
            
            # Hungarian matching
            cost_np = cost_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            
            # Compute matched cost (on pixels only)
            matched_goal = goal_slots[col_ind]  # (S, match_dim)
            
            if cost_type == "mse":
                sample_cost = F.mse_loss(pred_slots, matched_goal, reduction='mean')
            elif cost_type == "cosine":
                cos_sim = F.cosine_similarity(pred_slots, matched_goal, dim=-1)
                sample_cost = (1.0 - cos_sim).mean()
            
            costs[b, n] = sample_cost
    
    return costs


def reorder_slots_to_match(
    pred: torch.Tensor,
    reference: torch.Tensor,
    cost_type: str = "mse",
    pixels_dim: int = None,
) -> torch.Tensor:
    """
    Reorder predicted slots to match the order of reference slots.
    
    Used during rollout to maintain slot consistency across prediction steps.
    Matching is based on pixels_dim only (if specified), but the full embedding
    (including proprio/action) is reordered together.
    
    Args:
        pred: Predicted slots (B, T, S, D) or (B, S, D)
        reference: Reference slots to match against (B, S, D_ref)
                   Usually the last frame of the current history
                   D_ref can be pixels_dim (for matching) or full D
        cost_type: "mse" or "cosine"
        pixels_dim: If specified, use only first pixels_dim dimensions for matching.
                   The full embedding is still reordered.
    
    Returns:
        reordered_pred: Same shape as pred, with slots reordered
    """
    has_time_dim = pred.ndim == 4
    
    if has_time_dim:
        B, T, S, D = pred.shape
    else:
        B, S, D = pred.shape
        T = 1
        pred = pred.unsqueeze(1)  # (B, 1, S, D)
    
    D_ref = reference.shape[-1]
    assert reference.shape[:2] == (B, S), f"Reference shape mismatch: {reference.shape}, expected (B={B}, S={S}, D)"
    
    # Determine matching dimension
    if pixels_dim is not None:
        match_dim = pixels_dim
    else:
        match_dim = D_ref  # Use reference dimension for matching
    
    device = pred.device
    reordered = torch.zeros_like(pred)
    
    # For each batch, find the matching between reference and pred's first frame
    # Then apply the same permutation to all time steps
    pred_det = pred.detach()
    ref_det = reference.detach()
    
    for b in range(B):
        ref_slots = ref_det[b]  # (S, D_ref)
        # Use first predicted frame for matching (only pixels_dim portion)
        pred_slots_full = pred_det[b, 0]  # (S, D)
        pred_slots = pred_slots_full[:, :match_dim]  # (S, match_dim)
        ref_slots_match = ref_slots[:, :match_dim] if ref_slots.shape[-1] > match_dim else ref_slots  # (S, match_dim)
        
        # Build cost matrix: cost[i,j] = distance(pred[i], ref[j])
        if cost_type == "mse":
            diff = pred_slots.unsqueeze(1) - ref_slots_match.unsqueeze(0)
            cost_matrix = (diff ** 2).sum(dim=-1)  # (S, S)
        elif cost_type == "cosine":
            pred_norm = F.normalize(pred_slots, dim=-1)
            ref_norm = F.normalize(ref_slots_match, dim=-1)
            cost_matrix = 1.0 - torch.mm(pred_norm, ref_norm.t())
        else:
            raise ValueError(f"Unknown cost_type: {cost_type}")
        
        # Hungarian: find matching that minimizes cost(pred[i], ref[perm[i]])
        # We want to reorder pred so that pred[perm[i]] matches ref[i]
        cost_np = cost_matrix.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        
        # row_ind[i] should go to position col_ind[i]
        # We need inverse permutation: position col_ind[i] gets row_ind[i]
        inv_perm = torch.zeros(S, dtype=torch.long, device=device)
        inv_perm[col_ind] = torch.tensor(row_ind, device=device)
        
        # Apply same permutation to all time steps
        for t in range(T):
            reordered[b, t] = pred[b, t, inv_perm]
    
    if not has_time_dim:
        reordered = reordered.squeeze(1)
    
    return reordered


def reorder_slots_to_match_AP(
    pred: torch.Tensor,
    reference: torch.Tensor,
    cost_type: str = "mse",
    slot_dim: int = None,
) -> torch.Tensor:
    """
    Args:
        pred: Predicted slots (B, T, S, D) or (B, S, D)
        reference: Reference slots to match against (B, S_ref, D)
                   Usually the last frame of the current history
                   S_ref can be pixels_dim (for matching) or full S
        cost_type: "mse" or "cosine"
        slot_dim: If specified, use only first slotdim dimensions for matching.
                   The full embedding is still reordered.
    
    Returns:
        reordered_pred: Same shape as pred, with slots reordered
    """
    has_time_dim = pred.ndim == 4
    
    if has_time_dim:
        B, T, S, D = pred.shape
    else:
        B, S, D = pred.shape
        T = 1
        pred = pred.unsqueeze(1)  # (B, 1, S, D)
    
    S_ref = reference.shape[-2]
    # assert reference.shape[:2] == (B, S), f"Reference shape mismatch: {reference.shape}, expected (B={B}, S={S}, D)"
    
    # Determine matching dimension
    if slot_dim is not None:
        match_dim = slot_dim
    else:
        match_dim = S_ref  # Use reference dimension for matching
    
    device = pred.device
    reordered = torch.zeros_like(pred)
    
    # For each batch, find the matching between reference and pred's first frame
    # Then apply the same permutation to all time steps
    pred_det = pred.detach()
    ref_det = reference.detach()
    
    for b in range(B):
        ref_slots = ref_det[b]  # (S_ref, D)
        # Use first predicted frame for matching (only first slot_dim slots)
        pred_slots_full = pred_det[b, 0]  # (S, D)
        pred_slots = pred_slots_full[:match_dim, :]  # (match_dim, D)
        ref_slots_match = ref_slots[:match_dim, :] if ref_slots.shape[-2] > match_dim else ref_slots  # (match_dim, D)
        
        # Build cost matrix: cost[i,j] = distance(pred[i], ref[j])
        # Only comparing object slots (first match_dim slots in axis=-2)
        if cost_type == "mse":
            diff = pred_slots.unsqueeze(1) - ref_slots_match.unsqueeze(0)
            cost_matrix = (diff ** 2).sum(dim=-1)  # (match_dim, match_dim)
        elif cost_type == "cosine":
            pred_norm = F.normalize(pred_slots, dim=-1)
            ref_norm = F.normalize(ref_slots_match, dim=-1)
            cost_matrix = 1.0 - torch.mm(pred_norm, ref_norm.t())
        else:
            raise ValueError(f"Unknown cost_type: {cost_type}")
        
        # Hungarian: find matching that minimizes cost(pred[i], ref[perm[i]])
        cost_np = cost_matrix.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        
        # Create permutation for object slots only (size = match_dim, not S)
        # inv_perm[col_ind[i]] = row_ind[i] means position col_ind[i] gets slot row_ind[i]
        obj_perm = torch.zeros(match_dim, dtype=torch.long, device=device)
        obj_perm[col_ind] = torch.tensor(row_ind, device=device)
        
        # Apply permutation to all time steps
        for t in range(T):
            # Reorder only object slots (first match_dim slots in axis=-2)
            reordered[b, t, :match_dim, :] = pred[b, t, obj_perm, :]
            # Keep proprio and action slots unchanged (slots after match_dim)
            reordered[b, t, match_dim:, :] = pred[b, t, match_dim:, :]
    
    if not has_time_dim:
        reordered = reordered.squeeze(1)
    
    return reordered