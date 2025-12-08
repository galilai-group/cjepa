import pytest
import torch

from videosaur.videosaur.modules import encoders


@pytest.mark.parametrize("use_spatial_flatten", [False, True])
@pytest.mark.parametrize("use_pos_embed", [False, True])
@pytest.mark.parametrize("use_output_transform", [False, True])
def test_frame_encoder(use_spatial_flatten, use_pos_embed, use_output_transform):
    def spatial_flatten(x):
        return x.flatten(-2, -1).transpose(-2, -1)

    if use_spatial_flatten:
        backbone = lambda x: x + 1
    else:
        backbone = lambda x: spatial_flatten(x + 1)

    pos_embed = (lambda x: x + 2) if use_pos_embed else None
    output_transform = (lambda x: x + 4) if use_output_transform else None

    encoder = encoders.FrameEncoder(
        backbone,
        pos_embed=pos_embed,
        output_transform=output_transform,
        spatial_flatten=use_spatial_flatten,
    )

    inp = torch.zeros(1, 3, 4, 4)
    outputs = encoder(inp)

    expected_offset = 1 + 2 * int(use_pos_embed) + 4 * int(use_output_transform)
    assert torch.allclose(outputs["features"], spatial_flatten(inp + expected_offset))
    assert torch.allclose(outputs["backbone_features"], spatial_flatten(inp + 1))


@pytest.mark.parametrize(
    "model,features,expected_shape",
    [
        ("vit_tiny_patch16_224", ["vit_block1", "vit_block12", "vit_output"], (14 * 14, 192)),
        ("resnet34_savi", ["resnet_block4"], (512, 28, 28)),
    ],
)
@pytest.mark.parametrize("frozen", [False, True])
def test_timm_encoder(model, features, expected_shape, frozen):
    encoder = encoders.TimmExtractor(model, frozen=frozen, features=features)
    if frozen:
        for param in encoder.parameters():
            assert not param.requires_grad

    inp = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        outputs = encoder(inp)

    if len(features) > 1:
        for output in outputs.values():
            assert output.shape[1:] == expected_shape
    else:
        # Single features are automatically unpacked
        assert outputs.shape[1:] == expected_shape


def test_timm_encoder_dynamic_img_size():
    encoder = encoders.TimmExtractor(
        model="vit_tiny_patch16_224",
        features="vit_block1",
        model_kwargs={"dynamic_img_size": True, "dynamic_img_pad": True},
    )

    # After padding, should result in 128x128 image size -> 8 * 8 tokens
    inp = torch.randn(1, 3, 127, 127)

    with torch.no_grad():
        outputs = encoder(inp)

    assert outputs.shape == (1, 8 * 8, 192)
