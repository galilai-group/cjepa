import torch
import torch.nn.functional as F
import numpy as np

# from torchvision import transforms
import torchvision.transforms.v2 as transforms
from einops import rearrange, repeat
from torch import distributed as dist
from torch import nn



class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        num_patches=1,
        num_frames=1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            num_patches=num_patches,
                            num_frames=num_frames,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_patches=1, num_frames=1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

        self.register_buffer("bias", self.generate_mask_matrix(num_patches, num_frames))

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)

        # q, k, v: (B, heads, T, dim_head)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        attn_mask = self.bias[:, :, :T, :T] == 1  # bool mask

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False
        )

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)

    def generate_mask_matrix(self, npatch, nwindow):
        zeros = torch.zeros(npatch, npatch)
        ones = torch.ones(npatch, npatch)
        rows = []
        for i in range(nwindow):
            row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
            rows.append(row)
        mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
        return mask
    


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MaskedSlotPredictor(nn.Module):
    """
    V-JEPA style predictor for masked slot prediction.
    
    Input: (B, T, S, 64) - B: batch, T: history_length, S: num_slots, 64: slot_dim
    Output: (B, T + num_pred, S, 64) - predicts future slots
    
    Masking: M random slots are masked across all timesteps
    cfg.causal_mask_predict: if True, predict masked slots; if False, only predict future
    """
    
    def __init__(
        self,
        num_slots: int,
        slot_dim: int = 64,
        num_frames: int = 3,
        num_pred: int = 1,
        num_masked_slots: int = 2,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
        causal_mask_predict: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_frames = num_frames
        self.num_pred = num_pred
        self.num_masked_slots = num_masked_slots
        self.causal_mask_predict = causal_mask_predict
        self.seed = seed
        
        # Positional embedding for time axis only (slots are permutable)
        # Shape: (1, T + num_pred, slot_dim)
        self.time_pos_embedding = nn.Parameter(
            torch.randn(1, num_frames + num_pred, slot_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer backbone
        self.transformer = Transformer(
            dim=slot_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            num_patches=num_slots,
            num_frames=num_frames + num_pred,
        )
        
        # Output projection
        self.to_out = nn.Linear(slot_dim, slot_dim)
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.randn(1, 1, slot_dim))
        
    def get_masked_slots(self, x):
        """
        Randomly select M slots to mask across all timesteps.
        
        Args:
            x: (B, T, S, 64)
        
        Returns:
            x_masked: (B, T, S, 64) with masked slots replaced by mask_token
            mask_indices: (S,) bool array indicating which slots are masked
            masked_slot_ids: (M,) array of masked slot indices
        """
        B, T, S, D = x.shape
        
        # Use seed for reproducibility
        rng = np.random.RandomState(self.seed)
        
        # Randomly select M slots to mask
        mask_indices = np.zeros(S, dtype=bool)
        masked_slot_ids = rng.choice(S, self.num_masked_slots, replace=False)
        mask_indices[masked_slot_ids] = True
        
        # Create masked version: replace M masked slots with learnable mask token
        x_masked = x.clone()
        # Properly broadcast mask_token: (1, 1, 1, D) -> (B, T, M, D)
        for i, slot_id in enumerate(masked_slot_ids):
            x_masked[:, :, slot_id, :] = self.mask_token  # (1, 1, D) broadcasts to (B, T, D)
        
        return x_masked, torch.from_numpy(mask_indices).to(x.device), masked_slot_ids
    
    def forward(self, x, return_mask_info=False):
        """
        Args:
            x: (B, T, S, 64) - T: history_length, S: num_slots
            return_mask_info: if True, also return (mask_indices, T) for loss computation
        
        Returns:
            pred: (B, T+num_pred, S, 64) or (B, num_pred, S, 64) depending on causal_mask_predict
            (optionally) mask_info: tuple of (mask_indices, num_history_frames) for selective loss
        """
        B, T, S, D = x.shape
        
        # Get masked version and mask info
        x_masked, mask_indices, masked_slot_ids = self.get_masked_slots(x)
        
        # Add temporal positional embedding BEFORE flattening
        # Shape of x_masked: (B, T, S, D)
        # time_pos_embedding: (1, T, D) -> broadcast to (1, T, 1, D)
        x_with_pos = x_masked + self.time_pos_embedding[:, :T, :].unsqueeze(2)  # (B, T, S, D)
        x_with_pos = self.dropout(x_with_pos)
        
        # Flatten: (B, T, S, D) -> (B, T*S, D)
        x_flat = rearrange(x_with_pos, "b t s d -> b (t s) d")
        
        # Process through transformer
        x_transformed = self.transformer(x_flat)  # (B, T*S, D)
        
        # Unflatten back: (B, T*S, D) -> (B, T, S, D)
        x_transformed = rearrange(x_transformed, "b (t s) d -> b t s d", t=T, s=S)
        
        # Generate future predictions
        future_preds = []
        z = x_transformed  # (B, T, S, D)
        
        for step in range(self.num_pred):
            # Use last history frame to predict next
            z_hist = z[:, -self.num_frames:, :, :]  # (B, num_frames, S, D)
            
            # Add temporal positional embedding for prediction step
            # Position in time axis: T + step (from 0-indexed)
            time_idx = self.num_frames + step  # absolute position in sequence
            z_hist_with_pos = z_hist + self.time_pos_embedding[:, -self.num_frames:, :].unsqueeze(2)  # (B, num_frames, S, D)
            
            # Flatten
            z_hist_flat = rearrange(z_hist_with_pos, "b t s d -> b (t s) d")
            
            # Predict next frame
            z_pred = self.transformer(z_hist_flat)  # (B, num_frames*S, D)
            z_pred = rearrange(z_pred, "b (t s) d -> b t s d", t=self.num_frames, s=S)
            z_pred = z_pred[:, -1:, :, :]  # Take only the predicted frame (B, 1, S, D)
            
            future_preds.append(z_pred)
            z = torch.cat([z, z_pred], dim=1)  # (B, T+1, S, D)
        
        # Output: concatenate history and predictions
        output = torch.cat([x_transformed] + future_preds, dim=1)  # (B, T+num_pred, S, D)
        
        if self.causal_mask_predict:
            # Predict both masked slots and future: return full output
            result = output
        else:
            # Only predict future (exclude history timesteps)
            result = output[:, T:, :, :]  # (B, num_pred, S, D)
        
        if return_mask_info:
            # Return (output, mask_indices, num_history_frames)
            return result, mask_indices, T
        else:
            return result
