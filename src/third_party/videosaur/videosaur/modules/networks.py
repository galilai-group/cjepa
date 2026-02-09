import math
from typing import Callable, List, Optional, Tuple, Union

import einops
import torch
from torch import nn
from torch.nn import functional as F

from src.third_party.videosaur.videosaur.modules import utils
from src.third_party.videosaur.videosaur.utils import make_build_fn

# Default weight init for MLP, CNNEncoder, CNNDecoder
DEFAULT_WEIGHT_INIT = "default"


@make_build_fn(__name__, "network module")
def build(config, name: str):
    if name == "two_layer_mlp":
        inp_dim = None
        outp_dim = None
        if "dim" in config:
            inp_dim = config["dim"]
            outp_dim = config["dim"]
        if "inp_dim" in config:
            inp_dim = config["inp_dim"]
        if "outp_dim" in config:
            outp_dim = config["outp_dim"]

        if inp_dim is None:
            raise ValueError("Specify input dimensions with `inp_dim` or `dim`")
        if outp_dim is None:
            raise ValueError("Specify output dimension with `outp_dim` or `dim`")

        hidden_dims = [config.get("hidden_dim", 4 * inp_dim)]
        layer_norm = config.get("layer_norm") or config.get("initial_layer_norm", False)
        residual = config.get("residual", False)
        activation = config.get("activation", "relu")
        final_activation = config.get("final_activation", False)
        weight_init = config.get("weight_init", DEFAULT_WEIGHT_INIT)

        return MLP(
            inp_dim,
            outp_dim,
            hidden_dims,
            layer_norm,
            activation,
            final_activation,
            residual,
            weight_init,
        )
    elif name == "slot_attention_encoder" or name.startswith("savi_cnn_encoder"):
        inp_dim = config.get("inp_dim", 3)

        if name == "slot_attention_encoder":
            feature_multiplier = 1
            downsamplings = 0
        elif name == "savi_cnn_encoder":
            feature_multiplier = 1
            downsamplings = 1
        elif name == "savi_cnn_encoder_64":
            feature_multiplier = 0.5
            downsamplings = 0

        feature_multiplier = config.get("feature_multiplier", feature_multiplier)
        downsamplings = config.get("downsamplings", downsamplings)
        weight_init = config.get("weight_init", DEFAULT_WEIGHT_INIT)

        return make_slot_attention_encoder(inp_dim, feature_multiplier, downsamplings, weight_init)
    elif name.startswith("savi_decoder"):
        inp_dim = config.get("inp_dim")
        if inp_dim is None:
            raise ValueError("Need to specify input dimensions with `inp_dim`")

        if name == "savi_decoder":
            upsamplings = 4
        elif name == "savi_decoder_64":
            upsamplings = 3

        upsamplings = config.get("upsamplings", upsamplings)
        weight_init = config.get("weight_init", DEFAULT_WEIGHT_INIT)

        return make_savi_decoder(
            inp_dim, config.get("feature_multiplier", 1), upsamplings, weight_init
        )
    else:
        return None


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims: List[int],
        initial_layer_norm: bool = False,
        activation: Union[str, nn.Module] = "relu",
        final_activation: Union[bool, str] = False,
        residual: bool = False,
        weight_init: str = DEFAULT_WEIGHT_INIT,
    ):
        super().__init__()
        self.residual = residual
        if residual:
            assert inp_dim == outp_dim

        layers = []
        if initial_layer_norm:
            layers.append(nn.LayerNorm(inp_dim))

        cur_dim = inp_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(cur_dim, dim))
            layers.append(utils.get_activation_fn(activation))
            cur_dim = dim

        layers.append(nn.Linear(cur_dim, outp_dim))
        if final_activation:
            if isinstance(final_activation, bool):
                final_activation = "relu"
            layers.append(utils.get_activation_fn(final_activation))

        self.layers = nn.Sequential(*layers)
        utils.init_parameters(self.layers, weight_init)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        outp = self.layers(inp)

        if self.residual:
            return inp + outp
        else:
            return outp


def _infer_common_length(fail_on_missing_length=True, **kwargs) -> int:
    """Given kwargs of scalars and lists, checks that all lists have the same length and returns it.

    Optionally fails if no length was provided.
    """
    length = None
    name = None
    for cur_name, arg in kwargs.items():
        if isinstance(arg, (tuple, list)):
            cur_length = len(arg)
            if length is None:
                length = cur_length
                name = cur_name
            elif cur_length != length:
                raise ValueError(
                    f"Inconsistent lengths: {cur_name} has length {cur_length}, "
                    f"but {name} has length {length}"
                )

    if fail_on_missing_length and length is None:
        names = ", ".join(f"`{key}`" for key in kwargs.keys())
        raise ValueError(f"Need to specify a list for at least one of {names}.")

    return length


def _maybe_expand_list(arg: Union[int, List[int]], length: int) -> list:
    if not isinstance(arg, (tuple, list)):
        return [arg] * length

    return list(arg)


class CNNEncoder(nn.Sequential):
    """Simple convolutional encoder.

    For `features`, `kernel_sizes`, `strides`, scalars can be used to avoid repeating arguments,
    but at least one list needs to be provided to specify the number of layers.
    """

    def __init__(
        self,
        inp_dim: int,
        features: Union[int, List[int]],
        kernel_sizes: Union[int, List[int]],
        strides: Union[int, List[int]] = 1,
        outp_dim: Optional[int] = None,
        weight_init: str = "default",
    ):
        length = _infer_common_length(features=features, kernel_sizes=kernel_sizes, strides=strides)
        features = _maybe_expand_list(features, length)
        kernel_sizes = _maybe_expand_list(kernel_sizes, length)
        strides = _maybe_expand_list(strides, length)

        layers = []
        cur_dim = inp_dim
        for dim, kernel_size, stride in zip(features, kernel_sizes, strides):
            layers.append(
                nn.Conv2d(
                    cur_dim,
                    dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=self.get_same_padding(kernel_size, stride),
                )
            )
            layers.append(nn.ReLU(inplace=True))
            cur_dim = dim

        if outp_dim is not None:
            layers.append(nn.Conv1d(cur_dim, outp_dim, kernel_size=1, stride=1))

        super().__init__(*layers)
        utils.init_parameters(self, weight_init)

    @staticmethod
    def get_same_padding(kernel_size: int, stride: int) -> Union[str, int]:
        """Try to infer same padding for convolutions."""
        # This method is very lazily implemented, but oh well..
        if stride == 1:
            return "same"
        if kernel_size == 3:
            if stride == 2:
                return 1
        elif kernel_size == 5:
            if stride == 2:
                return 2

        raise ValueError(f"Don't know 'same' padding for kernel {kernel_size}, stride {stride}")


def make_slot_attention_encoder(
    inp_dim: int,
    feature_multiplier: float = 1,
    downsamplings: int = 0,
    weight_init: str = DEFAULT_WEIGHT_INIT,
) -> CNNEncoder:
    """CNN encoder as used in Slot Attention paper.

    By default, 4 layers with 64 channels each, keeping the spatial input resolution the same.

    This encoder is also used by SAVi, in the following configurations:

    - for image resolution 64: feature_multiplier=0.5, downsamplings=0
    - for image resolution 128: feature_multiplier=1, downsamplings=1

    and STEVE, in the following configurations:

    - for image resolution 64: feature_multiplier=1, downsamplings=0
    - for image resolution 128: feature_multiplier=1, downsamplings=1
    """
    assert 0 <= downsamplings <= 4
    channels = int(64 * feature_multiplier)
    strides = [2] * downsamplings + [1] * (4 - downsamplings)
    return CNNEncoder(
        inp_dim,
        features=[channels, channels, channels, channels],
        kernel_sizes=[5, 5, 5, 5],
        strides=strides,
        weight_init=weight_init,
    )


class CNNDecoder(nn.Sequential):
    """Simple convolutional decoder.

    For `features`, `kernel_sizes`, `strides`, scalars can be used to avoid repeating arguments,
    but at least one list needs to be provided to specify the number of layers.
    """

    def __init__(
        self,
        inp_dim: int,
        features: Union[int, List[int]],
        kernel_sizes: Union[int, List[int]],
        strides: Union[int, List[int]] = 1,
        outp_dim: Optional[int] = None,
        weight_init: str = DEFAULT_WEIGHT_INIT,
    ):
        length = _infer_common_length(features=features, kernel_sizes=kernel_sizes, strides=strides)
        features = _maybe_expand_list(features, length)
        kernel_sizes = _maybe_expand_list(kernel_sizes, length)
        strides = _maybe_expand_list(strides, length)

        layers = []
        cur_dim = inp_dim
        for dim, kernel_size, stride in zip(features, kernel_sizes, strides):
            padding, output_padding = self.get_same_padding(kernel_size, stride)
            layers.append(
                nn.ConvTranspose2d(
                    cur_dim,
                    dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            cur_dim = dim

        if outp_dim is not None:
            layers.append(nn.Conv1d(cur_dim, outp_dim, kernel_size=1, stride=1))

        super().__init__(*layers)
        utils.init_parameters(self, weight_init)

    @staticmethod
    def get_same_padding(kernel_size: int, stride: int) -> Tuple[int, int]:
        """Try to infer same padding for transposed convolutions."""
        # This method is very lazily implemented, but oh well..
        if kernel_size == 3:
            if stride == 1:
                return 1, 0
            if stride == 2:
                return 1, 1
        elif kernel_size == 5:
            if stride == 1:
                return 2, 0
            if stride == 2:
                return 2, 1

        raise ValueError(f"Don't know 'same' padding for kernel {kernel_size}, stride {stride}")


def make_savi_decoder(
    inp_dim: int,
    feature_multiplier: float = 1,
    upsamplings: int = 4,
    weight_init: str = DEFAULT_WEIGHT_INIT,
) -> CNNDecoder:
    """CNN encoder as used in SAVi paper.

    By default, 4 layers with 64 channels each, upscaling from a 8x8 feature map to 128x128.
    """
    assert 0 <= upsamplings <= 4
    channels = int(64 * feature_multiplier)
    strides = [2] * upsamplings + [1] * (4 - upsamplings)
    return CNNDecoder(
        inp_dim,
        features=[channels, channels, channels, channels],
        kernel_sizes=[5, 5, 5, 5],
        strides=strides,
        weight_init=weight_init,
    )


class Attention(nn.Module):
    """Multihead attention.

    Adapted from timm's ViT implementation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        inner_dim: Optional[int] = None,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        kdim = dim if kdim is None else kdim
        vdim = dim if vdim is None else vdim
        inner_dim = dim if inner_dim is None else inner_dim
        if inner_dim % num_heads != 0:
            raise ValueError("`inner_dim` must be divisible by `num_heads`")

        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.head_dim = inner_dim // num_heads
        self.scale = self.head_dim**-0.5

        self._same_qkv_dim = dim == kdim and dim == vdim
        self._same_kv_dim = kdim == vdim

        if self._same_qkv_dim:
            self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        elif self._same_kv_dim:
            self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
            self.kv = nn.Linear(kdim, inner_dim * 2, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
            self.k = nn.Linear(kdim, inner_dim, bias=qkv_bias)
            self.v = nn.Linear(vdim, inner_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.out_proj_drop = nn.Dropout(proj_drop)

        self.init_parameters()

    def init_parameters(self):
        if self._same_qkv_dim:
            bound = math.sqrt(6.0 / (self.qkv.weight.shape[0] // 3 + self.qkv.weight.shape[1]))
            nn.init.uniform_(self.qkv.weight, -bound, bound)  # Xavier init for separate Q, K, V
            if self.qkv.bias is not None:
                nn.init.constant_(self.qkv.bias, 0.0)
        elif self._same_kv_dim:
            utils.init_parameters(self.q, "xavier_uniform")
            bound = math.sqrt(6.0 / (self.kv.weight.shape[0] // 2 + self.kv.weight.shape[1]))
            nn.init.uniform_(self.kv.weight, -bound, bound)  # Xavier init for separate K, V
            if self.kv.bias is not None:
                nn.init.constant_(self.kv.bias, 0.0)
        else:
            utils.init_parameters((self.q, self.k, self.v), "xavier_uniform")

        utils.init_parameters(self.out_proj, "xavier_uniform")

    def _in_proj(self, q, k, v):
        """Efficiently compute in-projection.

        Adapted from torch.nn.functional.multi_head_attention.
        """
        if self._same_qkv_dim:
            w_kv = b_kv = b_q = b_k = b_v = None
            w = self.qkv.weight
            b = self.qkv.bias if hasattr(self.qkv, "bias") else None
        elif self._same_kv_dim:
            w = b = b_k = b_v = None
            w_q = self.q.weight
            w_kv = self.kv.weight
            b_q = self.q.bias if hasattr(self.q, "bias") else None
            b_kv = self.kv.bias if hasattr(self.kv, "bias") else None
        else:
            w = w_kv = b = b_kv = None
            w_q = self.q.weight
            w_k = self.k.weight
            w_v = self.v.weight
            b_q = self.q.bias if hasattr(self.q, "bias") else None
            b_k = self.k.bias if hasattr(self.k, "bias") else None
            b_v = self.v.bias if hasattr(self.v, "bias") else None

        if k is v:
            if q is k:
                # Self-attention
                return F.linear(q, w, b).chunk(3, dim=-1)
            else:
                # Encoder-decoder attention
                if w is not None:
                    dim = w.shape[0] // 3
                    w_q, w_kv = w.split([dim, dim * 2])
                    if b is not None:
                        b_q, b_kv = b.split([dim, dim * 2])
                return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
        else:
            if w is not None:
                w_q, w_k, w_v = w.chunk(3)
                if b is not None:
                    b_q, b_k, b_v = b.chunk(3)
            elif w_kv is not None:
                w_k, w_v = w_kv.chunk(2)
                if b_kv is not None:
                    b_k, b_v = b_kv.chunk(2)

            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = key if key is not None else query
        value = value if value is not None else query

        bs, n_queries, _ = query.shape
        n_keys = key.shape[1]

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                expected = (n_queries, n_keys)
                if attn_mask.shape != expected:
                    raise ValueError(
                        f"2D `attn_mask` should have shape {expected}, but has "
                        f"shape {attn_mask.shape}"
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.ndim == 3:
                expected = (bs * self.num_heads, n_queries, n_keys)
                if attn_mask.shape != expected:
                    raise ValueError(
                        f"3D `attn_mask` should have shape {expected}, but has "
                        f"shape {attn_mask.shape}"
                    )
        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool
            expected = (bs, n_keys)
            if key_padding_mask.shape != expected:
                raise ValueError(
                    f"`key_padding_mask` should have shape {expected}, but has shape "
                    f"{key_padding_mask.shape}"
                )
            key_padding_mask = einops.repeat(
                key_padding_mask, "b n -> (b h) 1 n", b=bs, h=self.num_heads, n=n_keys
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        q, k, v = self._in_proj(query, key, value)

        q = einops.rearrange(q, "b n (h d) -> (b h) n d", h=self.num_heads, d=self.head_dim)
        k = einops.rearrange(k, "b n (h d) -> (b h) n d", h=self.num_heads, d=self.head_dim)
        v = einops.rearrange(v, "b n (h d) -> (b h) n d", h=self.num_heads, d=self.head_dim)

        q_scaled = q / self.scale
        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn = torch.bmm(q_scaled, k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)  # (B x H) x N x M
        pre_dropout_attn = attn
        attn = self.attn_drop(attn)

        weighted_v = attn @ v
        x = einops.rearrange(weighted_v, "(b h) n d -> b n (h d)", h=self.num_heads, d=self.head_dim)
        x = self.out_proj(x)
        x = self.out_proj_drop(x)

        if return_weights:
            weights = einops.rearrange(pre_dropout_attn, "(b h) n m -> b h n m", h=self.num_heads)
            return x, weights.mean(dim=1)
        else:
            return x, None


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Like torch.nn.TransformerEncoderLayer, but with customizations."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dim_attn: Optional[int] = None,
        dim_kv: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.functional.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        initial_residual_scale: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device=device,
            dtype=dtype,
        )
        self.self_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            kdim=dim_kv,
            vdim=dim_kv,
            inner_dim=dim_attn,
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        if initial_residual_scale is not None:
            self.scale1 = utils.LayerScale(d_model, init_values=initial_residual_scale)
            self.scale2 = utils.LayerScale(d_model, init_values=initial_residual_scale)
        else:
            self.scale1 = nn.Identity()
            self.scale2 = nn.Identity()

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        keys = keys if keys is not None else x
        values = values if values is not None else x
        x, attn = self.self_attn(
            x,
            keys,
            values,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_weights=return_weights,
        )
        x = self.dropout1(x)

        if return_weights:
            return x, attn
        else:
            return x

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = x + self.scale1(
                self._sa_block(
                    self.norm1(x), src_mask, src_key_padding_mask, keys=memory, values=memory
                )
            )
            x = x + self.scale2(self._ff_block(self.norm2(x)))
        else:
            x = self.norm1(
                x
                + self.scale1(
                    self._sa_block(x, src_mask, src_key_padding_mask, keys=memory, values=memory)
                )
            )
            x = self.norm2(x + self.scale2(self._ff_block(x)))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_blocks: int,
        n_heads: int,
        qkv_dim: Optional[int] = None,
        memory_dim: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "relu",
        hidden_dim: Optional[int] = None,
        initial_residual_scale: Optional[float] = None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim,
                    n_heads,
                    dim_feedforward=hidden_dim,
                    dim_attn=qkv_dim,
                    dim_kv=memory_dim,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=1e-05,
                    batch_first=True,
                    norm_first=True,
                    initial_residual_scale=initial_residual_scale,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        inp: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = inp

        for block in self.blocks:
            x = block(x, mask, key_padding_mask, memory)

        return x


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    """Like torch.nn.TransformerDecoderLayer, but with customizations."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dim_attn: Optional[int] = None,
        dim_kv: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.functional.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        initial_residual_scale: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device=device,
            dtype=dtype,
        )
        self.self_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            inner_dim=dim_attn,
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.multihead_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            kdim=dim_kv,
            vdim=dim_kv,
            inner_dim=dim_attn,
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        if initial_residual_scale is not None:
            self.scale1 = utils.LayerScale(d_model, init_values=initial_residual_scale)
            self.scale2 = utils.LayerScale(d_model, init_values=initial_residual_scale)
            self.scale3 = utils.LayerScale(d_model, init_values=initial_residual_scale)
        else:
            self.scale1 = nn.Identity()
            self.scale2 = nn.Identity()
            self.scale3 = nn.Identity()

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        x, attn = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_weights=return_weights,
        )
        x = self.dropout1(x)

        if return_weights:
            return x, attn
        else:
            return x, None

    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        return_weights: bool = False,
    ) -> torch.Tensor:
        x, attn = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_weights=return_weights,
        )
        x = self.dropout2(x)

        if return_weights:
            return x, attn
        else:
            return x, None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        x = tgt
        if self.norm_first:
            residual, attn1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self.scale1(residual)
            residual, attn2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask, return_weights
            )
            x = x + self.scale2(residual)
            residual = self._ff_block(self.norm3(x))
            x = x + self.scale3(residual)
        else:
            residual, attn1 = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            x = self.norm1(x + self.scale1(residual))
            residual, attn2 = self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask, return_weights
            )
            x = self.norm2(x + self.scale2(residual))
            residual = self._ff_block(x)
            x = self.norm3(x + self.scale3(residual))

        if return_weights:
            return x, attn1, attn2
        else:
            return x, None, None


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_blocks: int,
        n_heads: int,
        qkv_dim: Optional[int] = None,
        memory_dim: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "relu",
        hidden_dim: Optional[int] = None,
        initial_residual_scale: Optional[float] = None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.blocks = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    dim,
                    n_heads,
                    dim_feedforward=hidden_dim,
                    dim_attn=qkv_dim,
                    dim_kv=memory_dim,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=1e-05,
                    batch_first=True,
                    norm_first=True,
                    initial_residual_scale=initial_residual_scale,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        output = tgt

        for idx, block in enumerate(self.blocks):
            output, _, attn = block(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                return_weights=return_weights and idx == len(self.blocks) - 1,
            )

        if return_weights:
            return output, attn
        else:
            return output
