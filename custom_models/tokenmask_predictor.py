import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from torchvision import transforms
import torchvision.transforms.v2 as transforms
from einops import rearrange, repeat
from torch import distributed as dist


class NonCausalTransformer(nn.Module):
    """
    Standard Transformer Encoder with Non-Causal (Full) Attention.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Batch_first=True for (B, Seq, D)
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                # FeedForward Network
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))

    def forward(self, x):
        # x: (B, SeqLen, D)
        for attn, ff in self.layers:
            # Self-attention with no mask (Full Attention)
            attn_out, _ = attn(x, x, x) 
            x = x + attn_out
            x = x + ff(x)
        return self.norm(x)

class TokenMaskedSlotPredictor(nn.Module):
    def __init__(
        self,
        num_slots: int,
        slot_dim: int = 64,
        num_frames: int = 3,
        num_pred: int = 1,
        mask_ratio: float = 0.5,   # N: Target masking ratio (0.0 to 1.0)
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
        causal_mask_predict: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_frames = num_frames
        self.num_pred = num_pred
        self.total_frames = num_frames + num_pred
        self.mask_ratio = mask_ratio
        self.causal_mask_predict = causal_mask_predict
        self.seed = seed
        # 1. Learnable Mask Token (Query Base)
        # Represents the center of the manifold for any missing data
        self.mask_token = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # 2. Time Positional Embedding
        # Shared across all slots, distinct for each timestep (0 to T_total)
        self.time_pos_embed = nn.Parameter(torch.randn(1, self.total_frames, 1, slot_dim))
        
        # 3. ID Projector (The "Anchor" mechanism)
        # Projects the t=0 latent (feature) into a Query (instruction)
        # "Here is what this object looked like at start, predict its future/history."
        self.id_projector = nn.Linear(slot_dim, slot_dim)

        # 4. Backbone (Non-Causal Transformer)
        self.transformer = NonCausalTransformer(
            dim=slot_dim, depth=depth, heads=heads, 
            dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout
        )
        
        # 5. Output Head
        self.to_out = nn.Linear(slot_dim, slot_dim)
        
    def get_masked_slots(self, x):
        """
        Sporadic token-level interpolation-prone masking.
        Uses stratified sampling to ensure masked tokens are uniformly spread
        across (T × S) space, like salt and pepper noise.
        
        Args:
            x: (B, T, S, D)
        
        Returns:
            x_masked: (B, T, S, D) with sporadic tokens replaced by mask_token
            mask_indices: (T, S) bool tensor indicating which (time, slot) positions are masked
        """
        B, T, S, D = x.shape
        total_positions = T * S
        num_masked = int(total_positions * self.mask_ratio)
        
        # Use seed for reproducibility
        rng = np.random.RandomState(self.seed)
        
        # Stratified sampling: divide (T*S) space into equal regions
        # and sample one position from each region to ensure uniform spread
        if num_masked > 0:
            region_size = total_positions / num_masked
            selected_indices = []
            
            for i in range(num_masked):
                region_start = int(i * region_size)
                region_end = int((i + 1) * region_size)
                # Sample one position from this region
                idx = rng.randint(region_start, max(region_start + 1, region_end))
                selected_indices.append(idx)
            
            selected_indices = np.array(selected_indices)
        else:
            selected_indices = np.array([], dtype=int)
        
        # Create (T, S) mask from flat indices
        mask_flat = np.zeros(total_positions, dtype=bool)
        if len(selected_indices) > 0:
            mask_flat[selected_indices] = True
        mask_indices = mask_flat.reshape(T, S)
        
        # Apply masking: replace masked positions with learnable mask_token
        x_masked = x.clone()
        mask_tensor = torch.from_numpy(mask_indices).to(x.device)
        mask_expanded = mask_tensor.unsqueeze(0).unsqueeze(-1).expand(B, T, S, D)  # (B, T, S, D)
        
        # Expand mask_token to match shape
        mask_token_expanded = self.mask_token.expand(B, T, S, D)
        
        x_masked = torch.where(mask_expanded, mask_token_expanded, x_masked)
        
        return x_masked, mask_tensor
    
    def prepare_input(self, x):
        """
        Constructs the input sequence for the Transformer with anchor mechanism.
        
        Logic:
        - t=0: ALWAYS Visible (Identity Anchor)
        - Masked positions: replaced with mask_token + TimePE + AnchorQuery
        - Unmasked positions: real data + TimePE
        - Future: mask_token + TimePE + AnchorQuery
        
        Args:
            x: (B, T_hist, S, D) - Ground Truth History
        Returns:
            full_input: (B, T_total, S, D)
            mask_tensor: (T_hist, S) bool tensor indicating masked positions
        """
        B, T_hist, S, D = x.shape
        T_total = self.total_frames
        device = x.device
        
        # 1. Get Mask Indices (only for history, excludes t=0)
        mask_tensor = self.get_masked_slots(x)
        
        # 2. Prepare Base Components
        # Anchors: First frame of all slots (B, S, D)
        anchors = x[:, 0, :, :] 
        
        # Project anchors to create Identity Queries (B, S, D)
        anchor_queries = self.id_projector(anchors)
        
        # 3. Construct the "Query Grid" (Default for everything)
        # Shape: (B, T_total, S, D)
        # Base = MaskToken + TimePE + AnchorQuery
        # This represents "Predict the state of [Anchor] at [Time]"
        
        # Expand dims for broadcasting
        # MaskToken: (1, 1, 1, D) -> (B, T, S, D)
        tokens_grid = self.mask_token.expand(B, T_total, S, D)
        
        # TimePE: (1, T, 1, D) -> (B, T, S, D)
        pos_grid = self.time_pos_embed.expand(B, T_total, S, D)
        
        # AnchorQueries: (B, 1, S, D) -> (B, T, S, D)
        anchor_grid = anchor_queries.unsqueeze(1).expand(B, T_total, S, D)
        
        # Full Query Input
        query_input = tokens_grid + pos_grid + anchor_grid

        # 4. Construct the "Real Data Grid" (Only available for history)
        # We start by cloning the query input, then overwrite visible parts with real data.
        final_input = query_input.clone()
        
        # (A) ALWAYS overwrite t=0 with Real Data + TimePE for ALL slots
        final_input[:, 0, :, :] = x[:, 0, :, :] + self.time_pos_embed[:, 0, :, :].expand(B, S, D)
        
        # (B) For unmasked positions in history (t=1 to T_hist-1), use real data
        if T_hist > 1:
            for t in range(1, T_hist):
                for s in range(S):
                    if not mask_tensor[t, s]:  # unmasked position
                        final_input[:, t, s, :] = x[:, t, s, :] + self.time_pos_embed[:, t, 0, :]
        
        return final_input, mask_tensor



    @torch.no_grad()
    def inference(self, x):
        """
        Inference function - no masking, predict future only.
        
        Args:
            x: (B, T_hist, S, D) - Fully visible history
        Returns:
            future_prediction: (B, T_pred, S, D)
        """
        B, T_hist, S, D = x.shape
        T_pred = self.num_pred
        T_total = T_hist + T_pred
        inf_time_pos_embed = self.time_pos_embed[:, -T_total:, :, :]
        device = x.device
        
        # 1. Anchor Query (t=0)
        anchors = x[:, 0, :, :]
        anchor_queries = self.id_projector(anchors) # (B, S, D)
        
        # 2. History Part  (NO MASK)
        input_history = x + inf_time_pos_embed[:, :T_hist, :, :]
        
        # Future: mask_token + TimePE + anchor
        tokens_grid = self.mask_token.expand(B, T_pred, S, D)
        pos_grid = inf_time_pos_embed[:, T_hist:T_total, :, :].expand(B, T_pred, S, D)
        anchor_grid = anchor_queries.unsqueeze(1).expand(B, T_pred, S, D)
        input_future = tokens_grid + pos_grid + anchor_grid
        full_input = torch.cat([input_history, input_future], dim=1)
        
        # Flatten
        x_flat = rearrange(full_input, 'b t s d -> b (t s) d')
        out_flat = self.transformer(x_flat)
        
        # Unflatten
        out = rearrange(out_flat, 'b (t s) d -> b t s d', t=T_total, s=S)
        out = self.to_out(out)
        return out[:, T_hist:, :, :]

    def forward(self, x):
        """
        Forward pass with Non-Causal Full Attention.
        
        Args:
            x: (B, T, S, D) - T: history_length, S: num_slots
        
        Returns:
            output: (B, T+num_pred, S, D)
            mask_tensor: (T, S) bool tensor indicating masked positions in history
        """
        B, T_hist, S, D = x.shape
        
        # Prepare input with masking and anchor mechanism
        x_input, mask_tensor = self.prepare_input(x)  # (B, T_total, S, D)
        
        # Flatten for Transformer: (B, T*S, D)
        x_flat = rearrange(x_input, 'b t s d -> b (t s) d')
        
        # 3. Non-Causal Full Attention
        # Every token (History, Future, Masked, Unmasked) attends to every other token.
        out_flat = self.transformer(x_flat)
        
        # 4. Unflatten
        out = rearrange(out_flat, 'b (t s) d -> b t s d', t=self.total_frames, s=S)
        
        # 5. Output Projection
        out = self.to_out(out)

        return out,  mask_tensor
    

class TokenMaskedSlot_AP_Predictor(TokenMaskedSlotPredictor):
    def __init__(
        self,
        num_slots: int,
        slot_dim: int = 64,
        num_frames: int = 3,
        num_pred: int = 1,
        mask_ratio: float = 0.5,   # N: Target masking ratio (0.0 to 1.0)
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
        causal_mask_predict: bool = True,
        seed: int = 42,
    ):
        super().__init__(
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_frames=num_frames,
            num_pred=num_pred,
            mask_ratio=mask_ratio,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            causal_mask_predict=causal_mask_predict,
            seed=seed,
        )
        # self.num_slots = num_slots
        # self.slot_dim = slot_dim
        # self.num_frames = num_frames
        # self.num_pred = num_pred
        # self.total_frames = num_frames + num_pred
        # self.mask_ratio = mask_ratio
        # self.causal_mask_predict = causal_mask_predict
        # self.seed = seed
        
        # 1. Learnable Mask Token (Query Base)
        # Represents the center of the manifold for any missing data
        self.mask_token = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # 2. Time Positional Embedding
        # Shared across all slots, distinct for each timestep (0 to T_total)
        self.time_pos_embed = nn.Parameter(torch.randn(1, self.total_frames, 1, slot_dim))
        
        # 3. ID Projector (The "Anchor" mechanism)
        # Projects the t=0 latent (feature) into a Query (instruction)
        # "Here is what this object looked like at start, predict its future/history."
        self.id_projector = nn.Linear(slot_dim, slot_dim)

        # 4. Backbone (Non-Causal Transformer)
        self.transformer = NonCausalTransformer(
            dim=slot_dim, depth=depth, heads=heads, 
            dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout
        )
        
        # 5. Output Head
        self.to_out = nn.Linear(slot_dim, slot_dim)

    def get_masked_slots(self, x):
        """
        Sporadic token-level interpolation-prone masking.
        Excludes action/proprio slots (last 2 slots) from masking.
        
        Args:
            x: (B, T, S, D)
        
        Returns:
            mask_tensor: (T, S) bool tensor indicating which positions are masked
        """
        B, T, S, D = x.shape
        num_object_slots = S - 2  # exclude AP slots from masking
        total_positions = T * num_object_slots
        num_masked = int(total_positions * self.mask_ratio)
        
        # Use seed for reproducibility
        rng = np.random.RandomState(self.seed)
        
        # Stratified sampling: divide space into equal regions
        # and sample one position from each region to ensure uniform spread
        if num_masked > 0:
            region_size = total_positions / num_masked
            selected_indices = []
            
            for i in range(num_masked):
                region_start = int(i * region_size)
                region_end = int((i + 1) * region_size)
                idx = rng.randint(region_start, max(region_start + 1, region_end))
                selected_indices.append(idx)
            
            selected_indices = np.array(selected_indices)
        else:
            selected_indices = np.array([], dtype=int)
        
        # Create (T, num_object_slots) mask from flat indices, then pad for AP slots
        mask_flat = np.zeros(total_positions, dtype=bool)
        if len(selected_indices) > 0:
            mask_flat[selected_indices] = True
        mask_object = mask_flat.reshape(T, num_object_slots)
        
        # Pad with False for action/proprio slots (never mask them)
        mask_indices = np.zeros((T, S), dtype=bool)
        mask_indices[:, :num_object_slots] = mask_object
        
        mask_tensor = torch.from_numpy(mask_indices).to(x.device)
        
        return mask_tensor

 