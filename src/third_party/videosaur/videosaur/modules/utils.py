import functools
import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import einops
import torch
from einops.layers.torch import Rearrange
from torch import nn

from src.third_party.videosaur.videosaur.utils import config_as_kwargs, get_class_by_name, make_build_fn


@make_build_fn(__name__, "utils module")
def build(config, name: str):
    if name == "Chain":
        if isinstance(config.models, Mapping):
            models = {name: build_module(conf) for name, conf in config.models.items()}
        elif isinstance(config.models, Iterable):
            models = {f"model{i}": build_module(conf) for i, conf in enumerate(config.models)}
        else:
            raise ValueError(f"Inappropriate type for utils.Chain.models: {type(config.models)}")
        return Chain(
            models,
            **config_as_kwargs(config, ("models",)),
        )
    else:
        return None


def build_module(
    config, default_name: Optional[str] = None, default_group: Optional[str] = None, **kwargs
):
    """Generic module build function."""
    if config is None:
        raise ValueError("No config specified while building module")
    name = config.get("name") or default_name
    if name is None:
        raise ValueError("No name specified while building module")

    group = None
    if "." in name:
        group, _, name = name.rpartition(".")

    if group is None or group == "":
        group = default_group

    from src.third_party.videosaur.videosaur.modules import BUILD_FNS_BY_MODULE_GROUP

    if group in BUILD_FNS_BY_MODULE_GROUP:
        build_fn = BUILD_FNS_BY_MODULE_GROUP[group]
        return build_fn(config, name, **kwargs)
    elif group is None:
        raise ValueError(
            f"No module group specified. Available groups: {list(BUILD_FNS_BY_MODULE_GROUP)}"
        )
    else:
        raise ValueError(f"Unknown module group {group}")


def build_torch_module(config, name: str):
    """Build function for torch.nn modules."""
    # import torch.nn  # noqa

    cls = get_class_by_name("torch.nn", name)
    if cls is not None:
        return cls(
            **config_as_kwargs(config),
        )
    else:
        raise ValueError(f"Unknown torch.nn module `{name}`")


def build_torch_function(config, name: str):
    """Build function for torch functions."""
    # import torch  # noqa

    fn = get_class_by_name("torch", name)
    if fn is not None:
        return functools.partial(
            fn,
            **config_as_kwargs(config),
        )
    else:
        raise ValueError(f"Unknown torch function `{name}`")


def init_parameters(layers: Union[nn.Module, Iterable[nn.Module]], weight_init: str = "default"):
    assert weight_init in ("default", "he_uniform", "he_normal", "xavier_uniform", "xavier_normal")
    if isinstance(layers, nn.Module):
        layers = [layers]

    for idx, layer in enumerate(layers):
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.zeros_(layer.bias)

        if hasattr(layer, "weight") and layer.weight is not None:
            gain = 1.0
            if isinstance(layers, nn.Sequential):
                if idx < len(layers) - 1:
                    next = layers[idx + 1]
                    if isinstance(next, nn.ReLU):
                        gain = 2**0.5

            if weight_init == "he_uniform":
                torch.nn.init.kaiming_uniform_(layer.weight, gain)
            elif weight_init == "he_normal":
                torch.nn.init.kaiming_normal_(layer.weight, gain)
            elif weight_init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(layer.weight, gain)
            elif weight_init == "xavier_normal":
                torch.nn.init.xavier_normal_(layer.weight, gain)


class Chain(nn.Module):
    """Chain several modules."""

    def __init__(
        self,
        models: Dict[str, nn.Module],
        mapping: Dict[str, List[str]] = None,
        last_output: bool = True,
    ):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.mapping = mapping if mapping else {}
        self.last_output = last_output

    def forward(self, *args):
        outputs = {}

        last_output = args
        for name, model in self.models.items():
            input_keys = self.mapping.get(name)
            if input_keys is None:
                inputs = last_output
            else:
                inputs = [outputs[key] for key in input_keys]

            if isinstance(inputs, Mapping):
                last_output = model(**inputs)
            elif isinstance(inputs, (list, tuple)):
                last_output = model(*inputs)
            else:
                last_output = model(inputs)

            if isinstance(last_output, Mapping):
                outputs.update(last_output)
            else:
                outputs[name] = last_output

        if self.last_output:
            return last_output
        else:
            return outputs


class Patchify:
    """Module that reshapes spatial inputs to patches."""

    def __init__(self, patch_size: int, video_inputs: bool = False):
        if video_inputs:
            self.to_patches = Rearrange(
                "b f c (h s1) (w s2) -> b f (h w) (s1 s2 c)", s1=patch_size, s2=patch_size
            )
        else:
            self.to_patches = Rearrange(
                "b c (h s1) (w s2) -> b (h w) (s1 s2 c)", s1=patch_size, s2=patch_size
            )

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """Patchify input.

        Args:
            input: tensor with shape (batch, [frames], channels, height, width).

        Returns:
            Patchified tensor of shape (batch, n_patches, n_dims).
        """
        return self.to_patches(input)


class Resizer:
    """Module that takes image-based tensor and resizes it to an appropriate size.

    Args:
        size: Tuple of (height, width) to resize to. If unspecified, assume an additional
            input used to infer the size. The last two dimensions of this input are taken
            as height and width.
        patch_inputs: If true, assumes tensor to resize has format `(batch, [frames],
            channels, n_points)` instead of separate height, width dimensions.
        patch_outputs: If true, flatten spatial dimensions after resizing.
        video_inputs: If true, assume inputs have an additional video dimension
        channels_last: If true, assume channel dimension comes after spatial dimensions. Output will
            be in same format as input.
        resize_mode: Mode to use for resizing. For nearest neighbor resizing, specify
            nearest-exact instead of nearest.
    """

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        patch_inputs: bool = False,
        patch_outputs: bool = False,
        video_inputs: bool = False,
        channels_last: bool = False,
        resize_mode: str = "bilinear",
    ):
        if resize_mode not in ("linear", "bilinear", "bicubic", "nearest-exact"):
            if resize_mode == "nearest":
                raise ValueError("Use resize mode `nearest-exact` instead of `nearest`")
            else:
                raise ValueError(f"Unsupported resize mode {resize_mode}")

        self.size = size
        self.patch_inputs = patch_inputs
        self.patch_outputs = patch_outputs
        self.video_inputs = video_inputs
        self.channels_last = channels_last
        self.n_expected_dims = 4 + (1 if video_inputs else 0) - (1 if patch_inputs else 0)
        self.resize_mode = resize_mode

    def __call__(
        self, inputs: torch.Tensor, size_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if inputs.ndim != self.n_expected_dims:
            raise ValueError(
                f"Mask has {inputs.ndim} dimensions, but expected it to "
                f"have {self.n_expected_dims} dimensions."
            )

        if self.size is None:
            if size_tensor is None:
                raise ValueError("If size is unspecified, need to pass a tensor to take size from")
            size = size_tensor.shape[-2:]
        else:
            size = list(self.size)

        if self.video_inputs:
            batch, n_frames = inputs.shape[:2]
            inputs = inputs.flatten(0, 1)

        if self.channels_last:
            if self.patch_inputs:
                inputs = inputs.transpose(-1, -2)
            else:
                inputs = inputs.transpose(-1, -2).transpose(-2, -3)

        if self.patch_inputs:
            n_patches = inputs.shape[-1]
            ratio = size[1] / size[0]
            height = int(math.sqrt(n_patches / ratio))
            width = int(math.sqrt(n_patches * ratio))
            if height * width != n_patches:
                if height == width:
                    raise ValueError(
                        f"Can not reshape {n_patches} patches to square aspect ratio as it's not a "
                        "perfect square."
                    )
                raise ValueError(f"Can not reshape {n_patches} patches to aspect ratio {ratio}.")

            inputs = inputs.unflatten(-1, (height, width))

        dtype = inputs.dtype
        if inputs.dtype == torch.bool:
            inputs = inputs.to(torch.uint8)

        outputs = torch.nn.functional.interpolate(inputs, size=size, mode=self.resize_mode)

        if inputs.dtype != dtype:
            inputs = inputs.to(dtype)

        if self.resize_mode == "bicubic":
            outputs.clamp_(0.0, 1.0)  # Bicubic interpolation can get out of range

        if self.patch_outputs:
            outputs = outputs.flatten(-2, -1)

        if self.channels_last:
            if self.patch_outputs:
                outputs = outputs.transpose(-2, -1)
            else:
                outputs = outputs.transpose(-3, -2).transpose(-2, -1)

        if self.video_inputs:
            outputs = outputs.unflatten(0, (batch, n_frames))

        return outputs


class SoftToHardMask:
    """Module that converts masks from soft to hard."""

    def __init__(
        self, convert_one_hot: bool = True, use_threshold: bool = False, threshold: float = 0.5
    ):
        self.convert_one_hot = convert_one_hot
        self.use_threshold = use_threshold
        self.threshold = threshold

    def __call__(self, masks: torch.Tensor) -> torch.Tensor:
        return soft_to_hard_mask(masks, self.convert_one_hot, self.use_threshold, self.threshold)


def soft_to_hard_mask(
    masks: torch.Tensor,
    convert_one_hot: bool = True,
    use_threshold: bool = False,
    threshold: float = 0.5,
):
    """Convert soft to hard masks."""
    # masks: batch [x n_frames] x n_channels x height x width
    assert masks.ndim == 4 or masks.ndim == 5
    min = torch.min(masks)
    max = torch.max(masks)
    if min < 0:
        raise ValueError(f"Minimum mask value should be >=0, but found {min.cpu().numpy()}")
    if max > 1:
        raise ValueError(f"Maximum mask value should be <=1, but found {max.cpu().numpy()}")

    if use_threshold:
        masks = masks > threshold

    if convert_one_hot:
        mask_argmax = torch.argmax(masks, dim=-3)
        masks = nn.functional.one_hot(mask_argmax, masks.shape[-3]).to(torch.float32)
        masks = masks.transpose(-1, -2).transpose(-2, -3)  # B, [F,] H, W, C -> B, [F], C, H, W

    return masks


class LayerScale(nn.Module):
    """Module scaling input by learned scalar.

    Adapted from timm library.
    """

    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def get_activation_fn(name_or_instance: Union[str, nn.Module]) -> nn.Module:
    if isinstance(name_or_instance, nn.Module):
        return name_or_instance
    elif isinstance(name_or_instance, str):
        if name_or_instance.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif name_or_instance.lower() == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation function {name_or_instance}")
    else:
        raise ValueError(
            f"Unsupported type for activation function: {type(name_or_instance)}. "
            "Can be `str` or `torch.nn.Module`."
        )


class CoordinatePositionEmbed(nn.Module):
    """Coordinate positional embedding as in Slot Attention."""

    def __init__(self, dim: int, size: Tuple[int, int]):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.register_buffer("grid", self.build_grid(size))
        self.proj = nn.Conv2d(self.grid.shape[0], dim, kernel_size=1, bias=True)
        init_parameters(self.proj, "xavier_uniform")

    @staticmethod
    def build_grid(
        size: Tuple[int, int],
        bounds: Tuple[float, float] = (-1.0, 1.0),
        add_inverse: bool = False,
    ) -> torch.Tensor:
        ranges = [torch.linspace(*bounds, steps=res) for res in size]
        grid = torch.meshgrid(*ranges, indexing="ij")

        if add_inverse:
            grid = torch.stack((grid[0], grid[1], 1.0 - grid[0], 1.0 - grid[1]), axis=0)
        else:
            grid = torch.stack((grid[0], grid[1]), axis=0)

        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.ndim == 4
        ), f"Expected input shape (batch, channels, height, width), but got {x.shape}"
        return x + self.proj(self.grid)


class LearnedPositionEmbed(nn.Module):
    """Learned positional embedding as in Vision Transformers."""

    def __init__(
        self,
        dim: int,
        *,
        n_patches: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
        initial_scale: Optional[float] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_patches is None and size is None:
            raise ValueError("Need to specify either `n_patches` (for 1D) or `size` (for 2D)")
        elif n_patches is not None and size is not None:
            raise ValueError("Can not to specify both `n_patches` (for 1D) or `size` (for 2D)")

        if initial_scale is None:
            initial_scale = dim**-0.5

        if n_patches is not None:
            self.expected_dims = 3
            self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, dim))
        else:
            self.expected_dims = 4
            if isinstance(size, int):
                size = (size, size)
            self.pos_emb = nn.Parameter(torch.zeros(1, dim, *size))

        nn.init.trunc_normal_(
            self.pos_emb, std=initial_scale, a=-2 * initial_scale, b=2 * initial_scale
        )

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.expected_dims == 3:
            assert (
                x.ndim == 3
            ), f"Expected input shape (batch, patches, dimensions), but got {x.shape}"
        elif self.expected_dims == 4:
            assert (
                x.ndim == 4
            ), f"Expected input shape (batch, channels, height, width), but got {x.shape}"

        x = x + self.pos_emb

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class FeatureSimilarity:
    """Compute dot-product based similarity between two sets of features.

    Args:
        normalize: Apply L2 normalization to features before computing dot-product, i.e. compute
            cosine similarity.
        temperature: Divide similarities by this value after computing dot-product.
        threshold: Set values below this threshold to maximum dissimilarity before temperature
            scaling.
        mask_diagonal: Whether to set the diagonal of the similarity matrix to maximum
            dissimilarity after applying temperature scaling.
        softmax: Apply softmax over last dimension after computing similarity.
        sigmoid: Apply sigmoid after computing similarity.
        relative: Whether to transform similarities such that resulting similarity matrix only
            contains similarities spatially around position.
        relative_window_size: Size of relative window.
    """

    def __init__(
        self,
        normalize: bool = True,
        temperature: float = 1.0,
        threshold: Optional[float] = None,
        mask_diagonal: bool = False,
        softmax: bool = False,
    ):
        self.normalize = normalize
        self.temperature = temperature
        self.threshold = threshold
        self.mask_diagonal = mask_diagonal
        self.softmax = softmax

        # Choose padding value such that it indicates maximum dissimilarity
        self.padding_value = -torch.inf if self.softmax else -1.0 / self.temperature

    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        similarity = torch.einsum("bpd, bkd -> bpk", features1, features2)

        if self.threshold is not None:
            similarity[similarity < self.threshold] = self.padding_value

        similarity /= self.temperature

        if self.mask_diagonal:
            diag = torch.diagonal(similarity, dim1=-2, dim2=-1)
            diag[:, :] = self.padding_value

        if self.softmax:
            # if all the values in a row are padding, softmax will return nan.
            # To avoid this, we set the padding values to 0.
            similarity[
                (similarity == self.padding_value)
                .all(dim=-1, keepdim=True)
                .expand(-1, -1, similarity.shape[-1])
            ] = 0.0
            similarity = torch.softmax(similarity, dim=-1)

        return similarity

    def __call__(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            features1 = nn.functional.normalize(features1, p=2.0, dim=-1)
            features2 = nn.functional.normalize(features2, p=2.0, dim=-1)

        return self.compute_similarity(features1, features2)


class FeatureSelfSimilarity(FeatureSimilarity):
    """Compute self-similarity between features."""

    def __init__(
        self,
        video_inputs: bool = False,
        normalize: bool = True,
        temperature: float = 1.0,
        threshold: Optional[float] = None,
        mask_diagonal: bool = False,
        softmax: bool = False,
    ):
        super().__init__(
            normalize,
            temperature,
            threshold,
            mask_diagonal,
            softmax,
        )
        self.video_inputs = video_inputs

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if self.video_inputs:
            bs = len(features)
            features = einops.rearrange(features, "b t p d -> (b t) p d")

        if self.normalize:
            features = nn.functional.normalize(features, p=2.0, dim=-1)

        similarity = self.compute_similarity(features, features)

        if self.video_inputs:
            similarity = einops.rearrange(similarity, "(b t) p k -> b t p k", b=bs)

        return similarity


class FeatureTimeSimilarity(FeatureSimilarity):
    """Compute similaritiy between features over time."""

    def __init__(
        self,
        time_shift: int = 1,
        normalize: bool = True,
        temperature: float = 1.0,
        threshold: Optional[float] = None,
        mask_diagonal: bool = False,
        softmax: bool = False,
    ):
        super().__init__(
            normalize,
            temperature,
            threshold,
            mask_diagonal,
            softmax,
        )
        self.time_shift = time_shift

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        assert features.ndim == 4, "`features` should have shape (batch, frames, positions, dims)"

        if self.normalize:
            features = nn.functional.normalize(features, p=2.0, dim=-1)

        source_features = features[:, : -self.time_shift]
        dest_features = features[:, self.time_shift :]

        source_features = einops.rearrange(source_features, "b t p d -> (b t) p d")
        dest_features = einops.rearrange(dest_features, "b t p d -> (b t) p d")

        similarity = self.compute_similarity(source_features, dest_features)

        similarity = einops.rearrange(similarity, "(b t) p k -> b t p k", b=len(features))

        return similarity
