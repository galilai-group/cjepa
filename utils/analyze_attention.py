"""
With dummy input right now!!!
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from custom_models.cjepa_predictor import MaskedSlotPredictor


class AttentionAnalyzer:
    """Hook-based attention map extraction and analysis."""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.attention_maps = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention maps."""
        for name, module in self.model.named_modules():
            if 'Attention' in module.__class__.__name__:
                hook = module.register_forward_hook(self._attention_hook(name))
                self.hooks.append(hook)
    
    def _attention_hook(self, name):
        def hook(module, input, output):
            # For scaled_dot_product_attention, we need to capture inside the forward
            # Store the module reference for later extraction
            self.attention_maps[name] = module
        return hook
    
    def remove_hooks(self):
        """Clean up hooks."""
        for hook in self.hooks:
            hook.remove()
    
    def extract_attention_weights(self, x, num_slots, num_frames):
        """
        Forward pass with attention weight extraction.
        
        Args:
            x: (B, T, S, D) - input slots
            num_slots: S
            num_frames: T
        
        Returns:
            attn_by_layer: list of attention weight tensors from each layer
        """
        B, T, S, D = x.shape
        
        # Get masked slots
        x_masked, mask_indices, masked_slot_ids = self.model.get_masked_slots(x)
        
        # Add positional embeddings
        x_with_pos = x_masked + self.model.time_pos_embedding[:, :T, :].unsqueeze(2)
        x_with_pos = self.model.dropout(x_with_pos)
        
        # Flatten for transformer
        x_flat = torch.einsum('btsd->bts', x_with_pos).reshape(B, T*S, D)
        
        # Manually forward through transformer to capture attention
        attn_weights_by_layer = []
        x_curr = x_flat
        
        for layer_idx, (attn, ff) in enumerate(self.model.transformer.layers):
            # Capture attention weights manually
            B, N, C = x_curr.shape
            x_norm = attn.norm(x_curr)
            
            qkv = attn.to_qkv(x_norm).chunk(3, dim=-1)
            q, k, v = (torch.einsum('bnd->bhnd', t.reshape(B, N, attn.heads, -1)) 
                      for t in qkv)
            
            # Compute attention weights
            scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
            attn_mask = attn.bias[:, :, :N, :N] == 1
            scores.masked_fill_(~attn_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            
            attn_weights_by_layer.append(attn_weights.detach())
            
            # Apply attention
            out = torch.matmul(attn_weights, v)
            out = torch.einsum('bhnd->bnd', out)
            out = attn.to_out(out)
            x_curr = out + x_curr
            
            # Feed-forward
            x_curr = ff(x_curr) + x_curr
        
        return attn_weights_by_layer, mask_indices, masked_slot_ids, (B, T, S, D)


def analyze_cross_slot_interaction(
    model, 
    x, 
    num_slots,
    num_frames,
    device='cpu',
    layer_idx=None  # which layer to analyze, None=average all
):
    """
    Compute cross-slot attention statistics.
    
    Returns:
        slot_interaction_matrix: (S, S) - average attention from slot i to slot j
        intra_temporal_attn: (T, T) - average attention within same slot across time
        inter_temporal_attn: (T, T) - average attention between different slots across time
    """
    analyzer = AttentionAnalyzer(model, device)
    
    B, T, S, D = x.shape
    attn_weights_list, mask_indices, masked_slot_ids, shape_info = analyzer.extract_attention_weights(
        x, num_slots, num_frames
    )
    analyzer.remove_hooks()
    
    # attn_weights_list: list of (B, heads, T*S, T*S) tensors
    num_layers = len(attn_weights_list)
    
    # Average across batch and heads
    if layer_idx is not None:
        attn = attn_weights_list[layer_idx].mean(dim=(0, 1))  # (T*S, T*S)
    else:
        attn = torch.stack(attn_weights_list).mean(dim=(0, 1, 2))  # average all layers, batch, heads
    
    # Reshape to separate slots and time
    # (T*S, T*S) -> interpret as (T, S, T, S) interaction matrix
    attn_reshaped = attn.reshape(T, S, T, S)  # attn[t1,s1,t2,s2] = attention from (t1,s1) to (t2,s2)
    
    # 1. Cross-slot interaction: average attention TO each slot FROM all slots at same timestep
    slot_interaction_matrix = torch.zeros(S, S, device=device)
    for t in range(T):
        for s_from in range(S):
            for s_to in range(S):
                # Average across all timesteps for attention from slot s_from to s_to
                slot_interaction_matrix[s_from, s_to] += attn_reshaped[:, s_from, t, s_to].mean()
    slot_interaction_matrix /= T
    
    # 2. Intra-temporal (same slot, different times)
    intra_temporal = torch.zeros(T, T, device=device)
    for s in range(S):
        intra_temporal += attn_reshaped[:, s, :, s]
    intra_temporal /= S
    
    # 3. Inter-temporal (different slots, different times)
    inter_temporal = torch.zeros(T, T, device=device)
    for t1 in range(T):
        for s1 in range(S):
            for t2 in range(T):
                for s2 in range(S):
                    if s1 != s2:
                        inter_temporal[t1, t2] += attn_reshaped[t1, s1, t2, s2]
    inter_temporal /= (S * (S - 1) * T)
    
    return {
        'slot_interaction': slot_interaction_matrix.cpu().numpy(),
        'intra_temporal': intra_temporal.cpu().numpy(),
        'inter_temporal': inter_temporal.cpu().numpy(),
        'mask_indices': mask_indices.cpu().numpy(),
        'masked_slot_ids': masked_slot_ids,
    }


def visualize_interaction(analysis_results, save_path='attention_analysis.png'):
    """Visualize cross-slot interaction patterns."""
    slot_interaction = analysis_results['slot_interaction']
    intra_temp = analysis_results['intra_temporal']
    inter_temp = analysis_results['inter_temporal']
    mask_indices = analysis_results['mask_indices']
    masked_ids = analysis_results['masked_slot_ids']
    
    S, _ = slot_interaction.shape
    T, _ = intra_temp.shape
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Cross-slot interaction heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(slot_interaction, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Attention'}, ax=ax1)
    ax1.set_xlabel('To Slot')
    ax1.set_ylabel('From Slot')
    ax1.set_title('Cross-Slot Attention Interaction Matrix')
    
    # Highlight masked slots
    for mid in masked_ids:
        ax1.axhline(mid, color='blue', linewidth=2, alpha=0.7)
        ax1.axvline(mid, color='blue', linewidth=2, alpha=0.7)
    
    # 2. Intra-temporal attention (same slot)
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(intra_temp, cmap='viridis', aspect='auto')
    ax2.set_xlabel('To Timestep')
    ax2.set_ylabel('From Timestep')
    ax2.set_title('Intra-Temporal Attention (Same Slot)')
    plt.colorbar(im, ax=ax2, label='Attention')
    
    # 3. Inter-temporal attention (different slots)
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(inter_temp, cmap='plasma', aspect='auto')
    ax3.set_xlabel('To Timestep')
    ax3.set_ylabel('From Timestep')
    ax3.set_title('Inter-Temporal Attention (Different Slots)')
    plt.colorbar(im, ax=ax3, label='Attention')
    
    # 4. Slot-to-slot attention strength (bar plot)
    ax4 = fig.add_subplot(gs[1, 0])
    interaction_strength = slot_interaction.sum(axis=1)  # total attention from each slot
    colors = ['red' if i in masked_ids else 'blue' for i in range(S)]
    ax4.bar(range(S), interaction_strength, color=colors, alpha=0.7)
    ax4.set_xlabel('Slot Index')
    ax4.set_ylabel('Total Attention Strength')
    ax4.set_title('Slot Attention Strength (Red=Masked)')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Interaction diversity (how evenly slots attend to others)
    ax5 = fig.add_subplot(gs[1, 1])
    # Entropy of attention distribution per slot
    entropy = []
    for i in range(S):
        probs = slot_interaction[i] / (slot_interaction[i].sum() + 1e-8)
        ent = -np.sum(probs * np.log(probs + 1e-8))
        entropy.append(ent)
    colors = ['red' if i in masked_ids else 'blue' for i in range(S)]
    ax5.bar(range(S), entropy, color=colors, alpha=0.7)
    ax5.set_xlabel('Slot Index')
    ax5.set_ylabel('Attention Distribution Entropy')
    ax5.set_title('Cross-Slot Attention Diversity')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = f"""
    === Cross-Slot Interaction Analysis ===
    
    Total Slots: {S}
    Masked Slots: {len(masked_ids)} {list(masked_ids)}
    Timeline: {T} frames
    
    Slot-Interaction Stats:
    • Mean interaction: {slot_interaction.mean():.4f}
    • Max interaction: {slot_interaction.max():.4f}
    • Min interaction: {slot_interaction.min():.4f}
    • Diagonal (self-attn): {np.diag(slot_interaction).mean():.4f}
    • Off-diagonal (cross-attn): {(slot_interaction.sum() - np.diag(slot_interaction).sum()) / (S * (S-1)):.4f}
    
    Temporal Stats:
    • Intra-temp entropy: {-np.sum(intra_temp * np.log(intra_temp + 1e-8)) / np.prod(intra_temp.shape):.4f}
    • Inter-temp entropy: {-np.sum(inter_temp * np.log(inter_temp + 1e-8)) / np.prod(inter_temp.shape):.4f}
    
    Interpretation:
    • Higher off-diagonal → better cross-slot learning
    • Higher cross-temp entropy → more diverse temporal patterns
    • Masked slots should have lower values if learning works
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    return fig


if __name__ == "__main__":
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = MaskedSlotPredictor(
        num_slots=4,
        slot_dim=64,
        num_frames=3,
        num_pred=1,
        num_masked_slots=2,
        depth=3,  # smaller for analysis
        heads=4,
        dim_head=16,
        mlp_dim=128,
    )
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    B, T, S, D = 2, 3, 4, 64
    x = torch.randn(B, T, S, D, device=device)
    
    # Analyze
    results = analyze_cross_slot_interaction(model, x, S, T, device=device)
    
    # Visualize
    visualize_interaction(results)
    
    print("\n=== Analysis Results ===")
    print(f"Slot interaction matrix shape: {results['slot_interaction'].shape}")
    print(f"Masked slots: {results['masked_slot_ids']}")
    print(f"\nSlot interaction matrix:\n{results['slot_interaction']}")
