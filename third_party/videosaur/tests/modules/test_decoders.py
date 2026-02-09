import pytest
import torch

from third_party.videosaur.videosaur.modules import decoders, networks


@pytest.mark.parametrize("eval_upscale", [False, True])
def test_mlp_decoder(eval_upscale):
    inp_dim, outp_dim, n_patches, n_slots = 5, 8, 4, 3

    if eval_upscale:
        hw = int(4 * n_patches**0.5)
        output_size = hw, hw
        n_output_patches = hw * hw
    else:
        output_size = None
        n_output_patches = n_patches

    decoder = decoders.MLPDecoder(
        inp_dim=inp_dim,
        outp_dim=outp_dim,
        hidden_dims=[10],
        n_patches=n_patches,
        eval_output_size=output_size,
    )
    if eval_upscale:
        decoder.eval()

    inp = torch.randn(1, n_slots, inp_dim)
    with torch.no_grad():
        outp = decoder(inp)

    assert outp["reconstruction"].shape == (1, n_output_patches, outp_dim)
    assert outp["masks"].shape == (1, n_slots, n_output_patches)
    assert outp["masks"].min() >= 0.0 and outp["masks"].max() <= 1.0


def test_spatial_broadcast_decoder():
    bs, inp_dim, outp_dim, feat_dim, height, width, n_slots = 2, 4, 6, 3, 5, 4, 3
    backbone = networks.CNNDecoder(inp_dim, [feat_dim, feat_dim], kernel_sizes=3, strides=2)
    decoder = decoders.SpatialBroadcastDecoder(
        inp_dim=inp_dim,
        outp_dim=outp_dim,
        backbone=backbone,
        initial_size=(height, width),
        backbone_dim=feat_dim,
    )

    inp = torch.randn(bs, n_slots, inp_dim)
    with torch.no_grad():
        outp = decoder(inp)

    target_height, target_width = height * 4, width * 4
    assert outp["reconstruction"].shape == (bs, outp_dim, target_height, target_width)
    assert outp["masks"].shape == (bs, n_slots, target_height, target_width)
    assert outp["masks"].min() >= 0.0 and outp["masks"].max() <= 1.0


@pytest.mark.parametrize("eval_upscale", [False, True])
def test_slot_mixer_decoder(eval_upscale):
    bs, inp_dim, outp_dim, embed_dim, feat_dim, n_patches, n_slots = 2, 4, 5, 6, 3, 9, 3
    transformer = networks.TransformerEncoder(embed_dim, n_blocks=1, n_heads=1, memory_dim=inp_dim)
    mlp = networks.MLP(inp_dim, feat_dim, [5], final_activation=True)

    if eval_upscale:
        hw = int(2 * n_patches**0.5)
        output_size = hw, hw
        n_output_patches = hw * hw
    else:
        output_size = None
        n_output_patches = n_patches

    decoder = decoders.SlotMixerDecoder(
        inp_dim=inp_dim,
        outp_dim=outp_dim,
        embed_dim=embed_dim,
        n_patches=n_patches,
        allocator=transformer,
        renderer=mlp,
        renderer_dim=feat_dim,
        eval_output_size=output_size,
    )
    if eval_upscale:
        decoder.eval()

    inp = torch.randn(bs, n_slots, inp_dim)
    with torch.no_grad():
        outp = decoder(inp)

    assert outp["reconstruction"].shape == (bs, n_output_patches, outp_dim)
    assert outp["masks"].shape == (bs, n_slots, n_output_patches)
    assert outp["masks"].min() >= 0.0 and outp["masks"].max() <= 1.0
