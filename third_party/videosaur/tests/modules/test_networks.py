import pytest
import torch

from third_party.videosaur.videosaur.modules import networks


@pytest.mark.parametrize(
    "kernel_sizes,strides,expected_size",
    [
        ([5, 3], 1, 32),
        ([5, 3], [2, 1], 16),
        ([5, 5], 2, 8),
        ([5, 3], 2, 8),
        (5, [2, 2], 8),
    ],
)
def test_cnn_encoder(kernel_sizes, strides, expected_size):
    bs, inp_dim, outp_dim = 2, 3, 5
    encoder = networks.CNNEncoder(
        inp_dim, features=[12, outp_dim], kernel_sizes=kernel_sizes, strides=strides
    )
    with torch.no_grad():
        outp = encoder(torch.randn(bs, inp_dim, 32, 32))
    assert outp.shape == (bs, outp_dim, expected_size, expected_size)


@pytest.mark.parametrize(
    "kernel_sizes,strides,expected_size",
    [
        ([5, 3], 1, 8),
        ([5, 3], [2, 1], 16),
        ([5, 5], 2, 32),
        ([5, 3], 2, 32),
        ([5, 5, 5, 5, 5, 3], [2, 2, 2, 2, 1, 1], 128),
    ],
)
def test_cnn_decoder(kernel_sizes, strides, expected_size):
    bs, inp_dim, outp_dim = 2, 3, 5
    features = [12] * (len(kernel_sizes) - 1) + [outp_dim]
    decoder = networks.CNNDecoder(
        inp_dim, features=features, kernel_sizes=kernel_sizes, strides=strides
    )
    with torch.no_grad():
        outp = decoder(torch.randn(bs, inp_dim, 8, 8))
    assert outp.shape == (bs, outp_dim, expected_size, expected_size)


@pytest.mark.parametrize(
    "qdim,kdim,vdim,inner_dim,same_qkv,same_kv",
    [
        (4, 5, 6, 8, False, False),
        (4, 5, 5, 8, False, False),
        (4, 5, 5, 8, False, True),
        (4, None, None, 8, False, False),
        (4, None, None, 8, False, True),
        (4, None, None, 8, True, True),
    ],
)
@pytest.mark.parametrize("qkv_bias", [False, True])
def test_attention(qdim, kdim, vdim, inner_dim, qkv_bias, same_qkv, same_kv):
    bs, src_len, tgt_len = 2, 3, 4
    attention = networks.Attention(
        dim=qdim, num_heads=2, kdim=kdim, vdim=vdim, inner_dim=inner_dim, qkv_bias=qkv_bias
    )
    kdim = qdim if kdim is None else kdim
    vdim = qdim if vdim is None else vdim

    q = torch.randn(bs, tgt_len, qdim)
    if same_qkv:
        k = v = q
    elif same_kv:
        k = torch.randn(bs, src_len, kdim)
        v = k
    else:
        k = torch.randn(bs, src_len, kdim)
        v = torch.randn(bs, src_len, vdim)

    attn_mask = torch.randn(q.shape[1], k.shape[1])
    attn_mask[0, 0] = -torch.inf
    key_padding_mask = torch.zeros(bs, k.shape[1], dtype=torch.bool)
    key_padding_mask[0, 0] = 1

    with torch.no_grad():
        outp, attn = attention(
            q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, return_weights=True
        )
    assert outp.shape == (bs, q.shape[1], qdim)
    assert attn.shape == (bs, q.shape[1], k.shape[1])
    assert attn.min() >= 0.0 and attn.max() <= 1.0
    assert torch.allclose(attn[:, 0, 0], torch.zeros_like(attn[:, 0, 0]))
    assert torch.allclose(attn[0, :, 0], torch.zeros_like(attn[0, :, 0]))


@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("initial_residual_scale", [None, 0.0])
def test_transformer_encoder(memory, initial_residual_scale):
    bs, seq_len, dim = 2, 4, 8
    memory_dim = 5 if memory else None

    encoder = networks.TransformerEncoder(
        dim,
        n_blocks=2,
        n_heads=4,
        memory_dim=memory_dim,
        initial_residual_scale=initial_residual_scale,
    )

    inp = torch.randn(bs, seq_len, dim)
    memory = torch.randn(bs, seq_len, memory_dim) if memory else None
    with torch.no_grad():
        outp = encoder(inp, memory=memory)
    assert outp.shape == (bs, seq_len, dim)

    if initial_residual_scale == 0.0:
        assert torch.allclose(inp, outp)


@pytest.mark.parametrize("initial_residual_scale", [None, 0.0])
def test_transformer_decoder(initial_residual_scale):
    bs, seq_len, mem_len, dim = 2, 3, 4, 8
    memory_dim = 5

    decoder = networks.TransformerDecoder(
        dim,
        n_blocks=2,
        n_heads=4,
        memory_dim=memory_dim,
        initial_residual_scale=initial_residual_scale,
    )

    inp = torch.randn(bs, seq_len, dim)
    memory = torch.randn(bs, mem_len, memory_dim)
    with torch.no_grad():
        outp = decoder(inp, memory=memory)
    assert outp.shape == (bs, seq_len, dim)

    if initial_residual_scale == 0.0:
        assert torch.allclose(inp, outp)
