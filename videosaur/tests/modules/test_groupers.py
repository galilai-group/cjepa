import pytest
import torch

from videosaur.modules import groupers


@pytest.mark.parametrize("use_mlp", [False, True])
def test_slot_attention(use_mlp):
    inp_dim, slot_dim, n_patches, n_slots = 5, 8, 6, 3
    slot_attention = groupers.SlotAttention(inp_dim, slot_dim, use_mlp=use_mlp)

    features = torch.randn(1, n_patches, inp_dim)
    slots = torch.randn(1, n_slots, slot_dim)

    with torch.no_grad():
        outp = slot_attention(slots, features)

    assert outp["slots"].shape == (1, n_slots, slot_dim)
    assert outp["masks"].shape == (1, n_slots, n_patches)
