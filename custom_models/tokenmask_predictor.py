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


class TokenMaskedSlotPredictor(nn.Module):
    """
    masked predictor for prediction with sporadic token-level interpolation-prone masking.
    
    Input: (B, T, S, 128) - B: batch, T: history_length, S: num_slots, 128: slot_dim
    Output: (B, T + num_pred, S, 128) - predicts future slots
    
    Masking Strategy: Token masking strategy
    - mask_ratio (N): target percentage of positions to mask (0.0 to 1.0)
    - Try to locate masked tokens not neighboring each other, which means uniformly spread
        across (T × S) space using multiple spatiotemporal token, like salt and pepper noise.
    - Goal : Make masking interpolation-prone.
    - This is opposite from bulk masking (contiguous region) used in VideoMAE.

    
    cfg.causal_mask_predict: if True, predict masked positions; if False, only predict future
    """
    
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
        self.mask_ratio = mask_ratio
        self.causal_mask_predict = causal_mask_predict
        self.seed = seed
        
        # Calculate block sizes to achieve target mask_ratio with num_mask_blocks blocks
        # Since blocks can overlap, we use a larger block size than simple division
        total_positions = num_frames * num_slots
        target_masked_positions = int(total_positions * mask_ratio)
        

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
    
    def forward(self, x):
        """
        Single-pass transformer prediction using causal masking (Option 1).
        
        Args:
            x: (B, T, S, 128) - T: history_length, S: num_slots
            return_mask_info: if True, also return (mask_indices, T) for loss computation
        
        Returns:
            pred: (B, T+num_pred, S, 128)
            mask_indices shape
        """
        B, T, S, D = x.shape
        
        # Get masked version and mask info (sporadic token-level masking)
        x_masked, mask_indices = self.get_masked_slots(x)
        
        # Create full sequence: history (masked) + future (placeholder zeros)
        # Shape: (B, T + num_pred, S, D)
        future_placeholder = torch.zeros(B, self.num_pred, S, D, device=x.device, dtype=x.dtype)
        x_full = torch.cat([x_masked, future_placeholder], dim=1)  # (B, T+num_pred, S, D)
        
        # Add temporal positional embedding BEFORE flattening
        # time_pos_embedding: (1, T+num_pred, D) -> broadcast to (1, T+num_pred, 1, D)
        x_with_pos = x_full + self.time_pos_embedding[:, :T+self.num_pred, :].unsqueeze(2)  # (B, T+num_pred, S, D)
        x_with_pos = self.dropout(x_with_pos)
        
        # Flatten: (B, T+num_pred, S, D) -> (B, (T+num_pred)*S, D)
        x_flat = rearrange(x_with_pos, "b t s d -> b (t s) d")
        
        # Single transformer pass with causal attention
        # The transformer's causal mask ensures each position only attends to previous positions
        # This allows the model to predict future frames in one pass
        x_transformed = self.transformer(x_flat)  # (B, (T+num_pred)*S, D)
        
        # Unflatten back: (B, (T+num_pred)*S, D) -> (B, T+num_pred, S, D)
        output = rearrange(x_transformed, "b (t s) d -> b t s d", t=T+self.num_pred, s=S)
        
        # Apply output projection
        output = self.to_out(output)  # (B, T+num_pred, S, D)
    

        return output, mask_indices

    
    @torch.no_grad()
    def inference(self, x):
        B, T, S, D = x.shape
        
        # Create full sequence: history (masked) + future (placeholder zeros)
        # Shape: (B, T + num_pred, S, D)
        future_placeholder = torch.zeros(B, self.num_pred, S, D, device=x.device, dtype=x.dtype)
        x_full = torch.cat([x, future_placeholder], dim=1)  # (B, T+num_pred, S, D)
        
        # Add temporal positional embedding BEFORE flattening
        # time_pos_embedding: (1, T+num_pred, D) -> broadcast to (1, T+num_pred, 1, D)
        x_with_pos = x_full + self.time_pos_embedding[:, :T+self.num_pred, :].unsqueeze(2)  # (B, T+num_pred, S, D)
        x_with_pos = self.dropout(x_with_pos)
        
        # Flatten: (B, T+num_pred, S, D) -> (B, (T+num_pred)*S, D)
        x_flat = rearrange(x_with_pos, "b t s d -> b (t s) d")
        
        # Single transformer pass with causal attention
        # The transformer's causal mask ensures each position only attends to previous positions
        # This allows the model to predict future frames in one pass
        x_transformed = self.transformer(x_flat)  # (B, (T+num_pred)*S, D)
        
        # Unflatten back: (B, (T+num_pred)*S, D) -> (B, T+num_pred, S, D)
        output = rearrange(x_transformed, "b (t s) d -> b t s d", t=T+self.num_pred, s=S)
        
        # Apply output projection
        output = self.to_out(output)  # (B, T+num_pred, S, D)
        return output[:, T:, :, :]
    