import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

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

    def forward(self, x, return_attention=False):
        # x: (B, SeqLen, D)
        attn_weights_list = [] if return_attention else None
        for attn, ff in self.layers:
            # Self-attention with no mask (Full Attention)
            if return_attention:
                attn_out, attn_weights = attn(x, x, x, need_weights=return_attention, average_attn_weights=True)
                attn_weights_list.append(attn_weights)  # (B, SeqLen, SeqLen)
            else:
                attn_out, _ = attn(x, x, x)
            x = x + attn_out
            x = x + ff(x)
        if return_attention:
            return self.norm(x), attn_weights_list
        return self.norm(x)

class MaskedSlotPredictor(nn.Module):
    def __init__(
        self,
        num_slots: int,             # Total number of slots per frame
        slot_dim: int = 64,
        history_frames: int = 3,    # Number of input frames
        pred_frames: int = 1,       # Number of future frames to predict
        num_masked_slots: int = 2,  # N slots to mask (context masking)
        seed: int = 42,             # Random seed for masking
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.total_frames = history_frames + pred_frames
        self.num_masked_slots = num_masked_slots
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

    def get_mask_indices(self, batch_size, device):
        """
        Selects N slots to be masked per sample (or shared across batch).
        Here, we implement shared masking across the batch for simplicity, 
        but it can be easily made per-sample.
        """
        rng = np.random.RandomState(self.seed)
        
        # Select N indices out of num_slots
        masked_indices = rng.choice(self.num_slots, self.num_masked_slots, replace=False)
        
        # Create boolean mask for logic (True = Masked/Target, False = Visible/Context)
        # This is strictly about "Slot" masking. Time masking logic is handled in prepare_input.
        is_slot_masked = torch.zeros(self.num_slots, dtype=torch.bool, device=device)
        is_slot_masked[masked_indices] = True
        
        return is_slot_masked, torch.from_numpy(masked_indices).to(device)

    def prepare_input(self, x):
        """
        Constructs the input sequence for the Transformer.
        
        Logic:
        - t=0: ALWAYS Visible (Identity Anchor).
        - Masked Slots (Target): Visible at t=0, Masked at t=1 ~ T_total.
        - Unmasked Slots (Context): Visible at t=0 ~ T_hist, Masked at Future.
        
        Args:
            x: (B, T_hist, S, D) - Ground Truth History
        Returns:
            full_input: (B, T_total, S, D)
            mask_indices: indices of slots that were masked
        """
        B, T_hist, S, D = x.shape
        T_total = self.total_frames
        device = x.device
        
        # 1. Get Mask Indices
        if self.num_masked_slots > 0 :
            is_slot_masked, masked_indices = self.get_mask_indices(B, device)
        else:
            masked_indices = torch.tensor([], dtype=torch.long, device=device)
        
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
        
        # --- Overwrite Logic ---
        
        # (A) ALWAYS overwrite t=0 with Real Data + TimePE(0) for ALL slots
        # This ensures the Anchor is physically present in the input
        final_input[:, 0, :, :] = x[:, 0, :, :] + self.time_pos_embed[:, 0, :, :]
        
        # (B) For UNMASKED (Context) slots, overwrite history (t=1 to T_hist-1)
        # Filter indices for unmasked slots
        if self.num_masked_slots > 0:
            unmasked_indices = torch.where(~is_slot_masked)[0]
        else:
            unmasked_indices = torch.arange(0, x.shape[2])
        
        if len(unmasked_indices) > 0 and T_hist > 1:
            # Extract real history for unmasked slots
            # x[:, 1:, ...] matches T_hist-1 frames
            real_history = x[:, 1:, unmasked_indices, :]
            
            # Add corresponding TimePE
            history_pos = self.time_pos_embed[:, 1:T_hist, :, :].expand(B, T_hist-1, S, D)
            history_pos_unmasked = history_pos[:, :, unmasked_indices, :]
            
            # Overwrite in final_input
            final_input[:, 1:T_hist, unmasked_indices, :] = real_history + history_pos_unmasked

        # Note: 
        # - Masked slots at t >= 1 remain as "Query Input".
        # - Unmasked slots at t >= T_hist (Future) remain as "Query Input".
        
        return final_input, masked_indices
    
    @torch.no_grad()
    def inference(self, x):
        """
        Inference function.
        Args:
            x: (B, T_hist, S, D) - Fully visible history
        Returns:
            future_prediction: (B, T_pred, S, D)
        """
        B, T_hist, S, D = x.shape
        T_pred = self.pred_frames
        T_total = T_hist + T_pred
        inf_time_pos_embed = self.time_pos_embed[:, -T_total:, :, :]
        
        # 1. Anchor Query (t=0)
        anchors = x[:, 0, :, :]
        anchor_queries = self.id_projector(anchors) # (B, S, D)
        
        # 2. History Part  (NO MASK)
        input_history = x + inf_time_pos_embed[:, :T_hist, :, :]
        
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
    
    @torch.no_grad()
    def attention_probing(self, x, layer_idx=-1):
        """
        Forward pass with attention extraction for probing.
        
        Args:
            x: (B, T_hist, S, D)
            layer_idx: Which transformer layer's attention to return. 
                       -1 means the last layer. Default: -1.
        
        Returns:
            out: (B, T_total, S, D) - Predicted slots
            masked_indices: Indices of masked slots
            attention_dict: Dictionary with attention scores.
                Keys: 
                    - 'masked_slot_{i}_t{t}' for masked slots at history timesteps (t=1 to T_hist-1)
                    - 'future_t{T_hist}_slot_{s}' for all slots at the first future timestep
                Values: (B, T_total, S) normalized attention scores (what that token attends to)
        """
        B, T_hist, S, D = x.shape
        T_total = self.total_frames
        
        # 1. Prepare Input (Mix of Real Data and Queries)
        x_input, masked_indices = self.prepare_input(x)  # (B, T_total, S, D)
        
        # 2. Flatten for Transformer: (B, T*S, D)
        x_flat = rearrange(x_input, 'b t s d -> b (t s) d')
        
        # 3. Non-Causal Full Attention with attention weights
        out_flat, attn_weights_list = self.transformer(x_flat, return_attention=True)
        
        # Select attention from specified layer (default: last layer)
        # attn_weights shape: (B, T*S, T*S)
        attn_weights = attn_weights_list[layer_idx]
        
        # 4. Unflatten output
        out = rearrange(out_flat, 'b (t s) d -> b t s d', t=T_total, s=S)
        
        # 5. Output Projection
        out = self.to_out(out)
        
        # 6. Extract attention scores for specific tokens
        # Reshape attention to (B, T_total, S, T_total, S) for easier indexing
        # attn[b, t_q, s_q, t_k, s_k] = attention from query (t_q, s_q) to key (t_k, s_k)
        attn_reshaped = rearrange(attn_weights, 'b (tq sq) (tk sk) -> b tq sq tk sk', 
                                   tq=T_total, sq=S, tk=T_total, sk=S)
        
        attention_dict = {}
        
        # (A) Attention for masked slots at history timesteps (t=1 to T_hist-1)
        # These are positions where masked slots have query tokens instead of real data
        for slot_idx in masked_indices.tolist():
            for t in range(1, T_hist):  # t=1 to T_hist-1
                key = f'masked_slot_{slot_idx}_t{t}'
                # Get attention: (B, T_total, S) - what this token attends to
                attn_slice = attn_reshaped[:, t, slot_idx, :, :]  # (B, T_total, S)
                attention_dict[key] = attn_slice
        
        # (B) Attention for all slots at the first future timestep (t=T_hist)
        first_future_t = T_hist
        for slot_idx in range(S):
            key = f'future_t{first_future_t}_slot_{slot_idx}'
            # Get attention: (B, T_total, S) - what this token attends to
            attn_slice = attn_reshaped[:, first_future_t, slot_idx, :, :]  # (B, T_total, S)
            attention_dict[key] = attn_slice
        
        return out, masked_indices, attention_dict

    def forward(self, x):
        """
        Args:
            x: (B, T_hist, S, D)
            return_indices: If True, returns the indices of masked slots.
        """
        B, T_hist, S, D = x.shape
        
        # 1. Prepare Input (Mix of Real Data and Queries)
        x_input, masked_indices = self.prepare_input(x) # (B, T_total, S, D)
        
        # 2. Flatten for Transformer: (B, T*S, D)
        x_flat = rearrange(x_input, 'b t s d -> b (t s) d')
        
        # 3. Non-Causal Full Attention
        # Every token (History, Future, Masked, Unmasked) attends to every other token.
        out_flat = self.transformer(x_flat)
        
        # 4. Unflatten
        out = rearrange(out_flat, 'b (t s) d -> b t s d', t=self.total_frames, s=S)
        
        # 5. Output Projection
        out = self.to_out(out)

        return out, masked_indices



class MaskedSlot_AP_Predictor(MaskedSlotPredictor):
    def __init__(
        self,
        num_slots: int,             # Total number of slots per frame
        slot_dim: int = 64,
        history_frames: int = 3,    # Number of input frames
        pred_frames: int = 1,       # Number of future frames to predict
        num_masked_slots: int = 2,  # N slots to mask (context masking)
        seed: int = 42,             # Random seed for masking
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        future_action_conditioning: bool = False,  # If True, includes action conditioning in future frames
    ):
        super().__init__(
            num_slots=num_slots,
            slot_dim=slot_dim,
            history_frames=history_frames,
            pred_frames=pred_frames,
            num_masked_slots=num_masked_slots,
            seed=seed,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.future_action_conditioning = future_action_conditioning

    def get_mask_indices(self, batch_size, device):
        rng = np.random.RandomState(self.seed)
        
        # Select N indices out of num_slots
        masked_indices = rng.choice(self.num_slots-2, self.num_masked_slots, replace=False) # excluding the last 2 slots: proprio and action
        
        # Create boolean mask for logic (True = Masked/Target, False = Visible/Context)
        # This is strictly about "Slot" masking. Time masking logic is handled in prepare_input.
        is_slot_masked = torch.zeros(self.num_slots, dtype=torch.bool, device=device)
        is_slot_masked[masked_indices] = True
        
        return is_slot_masked, torch.from_numpy(masked_indices).to(device)
    
