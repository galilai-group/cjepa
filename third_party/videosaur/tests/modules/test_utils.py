import pytest
import torch

from third_party.videosaur.videosaur.modules import utils


@pytest.mark.parametrize("last_output", [False, True])
def test_chain(last_output):
    class SingleModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inp):
            return inp

    class DictModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inp1, inp2=None):
            outputs = {"out1": inp1}
            if inp2 is not None:
                outputs["out2"] = inp2

            return outputs

    inp = 123

    chain = utils.Chain({"mod1": SingleModule()}, last_output=last_output)
    if last_output:
        assert chain(inp) == inp
    else:
        assert chain(inp) == {"mod1": inp}

    chain = utils.Chain(
        {"mod1": SingleModule(), "mod2": DictModule()},
        {"mod2": ["mod1", "mod1"]},
        last_output=last_output,
    )
    if last_output:
        assert chain(inp) == {"out1": inp, "out2": inp}
    else:
        assert chain(inp) == {"mod1": inp, "out1": inp, "out2": inp}

    chain = utils.Chain(
        {"mod1": DictModule(), "mod2": SingleModule()}, {"mod2": ["out1"]}, last_output=last_output
    )
    if last_output:
        assert chain(inp) == inp
    else:
        assert chain(inp) == {"out1": inp, "mod2": inp}


def test_patchify():
    bs, channels, h, w, patch_size = 2, 3, 4, 4, 2
    n_patches = (h / patch_size) * (w / patch_size)
    patchify = utils.Patchify(patch_size)

    inp = torch.stack([torch.stack([torch.eye(h)] * 3)] * bs)
    patches = patchify(inp)

    assert patches.shape == (bs, n_patches, channels * patch_size * patch_size)
    assert torch.allclose(
        patches[0, 0], torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.float)
    )


@pytest.mark.parametrize("input_size", [(5, 5), (4, 3)])
@pytest.mark.parametrize("size_tensor", [False, True])
@pytest.mark.parametrize("patch_inputs", [False, True])
@pytest.mark.parametrize("patch_outputs", [False, True])
@pytest.mark.parametrize("video_inputs", [False, True])
@pytest.mark.parametrize("channels_last", [False, True])
def test_mask_resizer(
    input_size, size_tensor, patch_inputs, patch_outputs, video_inputs, channels_last
):
    target_size = (8, 6)
    if size_tensor:
        size = None
        size_tensor = torch.ones(target_size)
    else:
        size = target_size
        size_tensor = None

    resizer = utils.Resizer(
        size=size,
        patch_inputs=patch_inputs,
        patch_outputs=patch_outputs,
        video_inputs=video_inputs,
        channels_last=channels_last,
    )

    batch_size, n_frames, n_channels = 2, 3, 4
    height, width = input_size

    def build_shape(height, width, flatten_patches=False):
        shape = [batch_size]
        if video_inputs:
            shape.append(n_frames)

        if not channels_last:
            shape.append(n_channels)

        if flatten_patches:
            shape.append(height * width)
        else:
            shape.extend([height, width])

        if channels_last:
            shape.append(n_channels)
        return shape

    inputs = torch.rand(*build_shape(height, width, flatten_patches=patch_inputs))

    if patch_inputs and height / width != target_size[0] / target_size[1]:
        # Can not resize from patches if aspect ratio does not match target
        with pytest.raises(ValueError):
            resized_inputs = resizer(inputs, size_tensor)
    else:
        resized_inputs = resizer(inputs, size_tensor)
        assert list(resized_inputs.shape) == build_shape(*target_size, flatten_patches=patch_outputs)


def test_coordinate_position_embed():
    dims, size = 5, (3, 3)
    embed = utils.CoordinatePositionEmbed(dim=dims, size=size)

    grid = embed.build_grid(size, bounds=(0.0, 1.0))
    assert torch.all((0.0 <= grid) & (grid <= 1.0))
    assert list(grid.shape) == [2, *size]

    grid = embed.build_grid(size, bounds=(0.0, 1.0), add_inverse=True)
    assert torch.all((0.0 <= grid) & (grid <= 1.0))
    assert list(grid.shape) == [4, *size]

    assert list(embed(torch.randn(2, dims, *size)).shape) == [2, dims, *size]
