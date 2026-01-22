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
    V-JEPA style predictor for masked slot prediction with spatiotemporal block masking.
    
    Input: (B, T, S, 64) - B: batch, T: history_length, S: num_slots, 64: slot_dim
    Output: (B, T + num_pred, S, 64) - predicts future slots
    
    Masking Strategy (V-JEPA/V-JEPA2):
    - M spatiotemporal blocks (bulks) are randomly placed
    - Block sizes are automatically calculated to achieve N% masking ratio
    - num_mask_blocks (M): controls how many blocks to place
    - mask_ratio (N): target percentage of positions to mask (0.0 to 1.0)
    - Blocks can overlap, creating diverse masking patterns
    
    Example: num_mask_blocks=3, mask_ratio=0.5
      → 3 blocks placed to cover ~50% of (T × S) positions
    
    cfg.causal_mask_predict: if True, predict masked positions; if False, only predict future
    """
    
    def __init__(
        self,
        num_slots: int,
        slot_dim: int = 64,
        num_frames: int = 3,
        num_pred: int = 1,
        num_mask_blocks: int = 2,  # M: Number of spatiotemporal blocks to mask
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
        self.num_mask_blocks = num_mask_blocks
        self.mask_ratio = mask_ratio
        self.causal_mask_predict = causal_mask_predict
        self.seed = seed
        
        # Calculate block sizes to achieve target mask_ratio with num_mask_blocks blocks
        # Since blocks can overlap, we use a larger block size than simple division
        total_positions = num_frames * num_slots
        target_masked_positions = int(total_positions * mask_ratio)
        
        # Use larger blocks to account for potential overlaps
        # Heuristic: each block should cover slightly more to reach target despite overlaps
        overlap_factor = 1.5  # Assume some overlap, so make blocks bigger
        positions_per_block = max(1, int(target_masked_positions * overlap_factor / num_mask_blocks))
        
        # Calculate block dimensions (try to make them roughly square/rectangular)
        # Prefer temporal dimension to be smaller than spatial for video data
        aspect_ratio = num_slots / num_frames  # S/T ratio
        block_area = positions_per_block
        
        self.temporal_block_size = max(1, int(np.sqrt(block_area / aspect_ratio)))
        self.spatial_block_size = max(1, int(block_area / self.temporal_block_size))
        
        # Ensure blocks don't exceed dimensions
        self.temporal_block_size = min(self.temporal_block_size, num_frames)
        self.spatial_block_size = min(self.spatial_block_size, num_slots)
        
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
        V-JEPA style: Place multiple spatiotemporal blocks (bulks) randomly.
        Each block is a continuous rectangular region in (time, slot) space.
        
        Args:
            x: (B, T, S, 64)
        
        Returns:
            x_masked: (B, T, S, 64) with spatiotemporal blocks replaced by mask_token
            mask_indices: (T, S) bool array indicating which (time, slot) positions are masked
            masked_blocks: List of (t_start, s_start, t_size, s_size) tuples for each block
        """
        B, T, S, D = x.shape
        
        # Use seed for reproducibility
        rng = np.random.RandomState(self.seed)
        
        # Initialize mask matrix
        mask_indices = np.zeros((T, S), dtype=bool)
        masked_blocks = []
        
        # Place multiple spatiotemporal blocks
        for _ in range(self.num_mask_blocks):
            # Randomly select block anchor point (top-left corner)
            # Ensure block fits within boundaries
            t_start = rng.randint(0, max(1, T - self.temporal_block_size + 1))
            s_start = rng.randint(0, max(1, S - self.spatial_block_size + 1))
            
            # Determine actual block size (handle edge cases)
            t_size = min(self.temporal_block_size, T - t_start)
            s_size = min(self.spatial_block_size, S - s_start)
            
            # Mark this block region as masked
            mask_indices[t_start:t_start+t_size, s_start:s_start+s_size] = True
            
            masked_blocks.append((t_start, s_start, t_size, s_size))
        
        # Apply masking to all positions marked in mask_indices
        x_masked = x.clone()
        for t in range(T):
            for s in range(S):
                if mask_indices[t, s]:
                    x_masked[:, t, s, :] = self.mask_token  # (1, 1, D) broadcasts to (B, D)
        
        return x_masked, torch.from_numpy(mask_indices).to(x.device), masked_blocks
    
    def forward(self, x):
        """
        Single-pass transformer prediction using causal masking (Option 1).
        
        Args:
            x: (B, T, S, 64) - T: history_length, S: num_slots
            return_mask_info: if True, also return (mask_indices, T) for loss computation
        
        Returns:
            pred: (B, T+num_pred, S, 64) or (B, num_pred, S, 64) depending on causal_mask_predict
            (optionally) mask_info: tuple of (mask_indices, num_history_frames) for selective loss
                         mask_indices shape: (T, S) for V-JEPA style masking
        """
        B, T, S, D = x.shape
        
        # Get masked version and mask info (V-JEPA: spatiotemporal blocks)
        x_masked, mask_indices, masked_blocks = self.get_masked_slots(x)
        
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
    

class MaskedSlot_AP_Predictor(MaskedSlotPredictor):
    """
    V-JEPA style predictor for masked slot prediction with spatiotemporal block masking.
    
    Input: (B, T, S, 64) - B: batch, T: history_length, S: num_slots, 64: slot_dim
    Output: (B, T + num_pred, S, 64) - predicts future slots
    
    Masking Strategy (V-JEPA/V-JEPA2):
    - M spatiotemporal blocks (bulks) are randomly placed
    - Block sizes are automatically calculated to achieve N% masking ratio
    - num_mask_blocks (M): controls how many blocks to place
    - mask_ratio (N): target percentage of positions to mask (0.0 to 1.0)
    - Blocks can overlap, creating diverse masking patterns
    
    Example: num_mask_blocks=3, mask_ratio=0.5
      → 3 blocks placed to cover ~50% of (T × S) positions
    
    cfg.causal_mask_predict: if True, predict masked positions; if False, only predict future
    """
    
    def __init__(
        self,
        num_slots: int,
        slot_dim: int = 64,
        num_frames: int = 3,
        num_pred: int = 1,
        num_mask_blocks: int = 2,  # M: Number of spatiotemporal blocks to mask
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
            num_mask_blocks=num_mask_blocks,
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
    

        
        # Calculate block sizes to achieve target mask_ratio with num_mask_blocks blocks
        # Since blocks can overlap, we use a larger block size than simple division
        total_positions = num_frames * (num_slots-2)
        target_masked_positions = int(total_positions * mask_ratio)
        
        # Use larger blocks to account for potential overlaps
        # Heuristic: each block should cover slightly more to reach target despite overlaps
        overlap_factor = 1.5  # Assume some overlap, so make blocks bigger
        positions_per_block = max(1, int(target_masked_positions * overlap_factor / num_mask_blocks))
        
        # Calculate block dimensions (try to make them roughly square/rectangular)
        # Prefer temporal dimension to be smaller than spatial for video data
        aspect_ratio = (num_slots-2) / num_frames  # S/T ratio
        block_area = positions_per_block
        
        self.temporal_block_size = max(1, int(np.sqrt(block_area / aspect_ratio)))
        self.spatial_block_size = max(1, int(block_area / self.temporal_block_size)-1)
        
        # Ensure blocks don't exceed dimensions
        self.temporal_block_size = min(self.temporal_block_size, num_frames)
        self.spatial_block_size = min(self.spatial_block_size, num_slots)
        
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

        B, T, S, D = x.shape
        
        # Use seed for reproducibility
        rng = np.random.RandomState(self.seed)
        
        # Initialize mask matrix
        mask_indices = np.zeros((T, S), dtype=bool)
        masked_blocks = []
        
        # Place multiple spatiotemporal blocks
        for _ in range(self.num_mask_blocks):
            # Randomly select block anchor point (top-left corner)
            # Ensure block fits within boundaries
            t_start = rng.randint(0, max(1, T - self.temporal_block_size + 1))
            s_start = rng.randint(0, max(1, S -2 - self.spatial_block_size + 1))
            
            # Determine actual block size (handle edge cases)
            t_size = min(self.temporal_block_size, T - t_start)
            s_size = min(self.spatial_block_size, S -2 - s_start)
            
            # Mark this block region as masked
            mask_indices[t_start:t_start+t_size, s_start:s_start+s_size] = True
            
            masked_blocks.append((t_start, s_start, t_size, s_size))
        
        # Apply masking to all positions marked in mask_indices
        x_masked = x.clone()
        for t in range(T):
            for s in range(S):
                if mask_indices[t, s]:
                    x_masked[:, t, s, :] = self.mask_token  # (1, 1, D) broadcasts to (B, D)
        
        return x_masked, torch.from_numpy(mask_indices).to(x.device), masked_blocks