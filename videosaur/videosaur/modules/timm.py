"""Implementations of custom timm models.

Models are registered with timm and can be instantiated with `timm.create_model`.

This module provides the following models:

- resnet18_savi (no pre-trained weights)
- resnet34_savi (no pre-trained weights)
- resnet50_savi (no pre-trained weights)
- resnet50_dino
- vit_base_patch16_224_mae
- vit_large_patch16_224_mae
- vit_huge_patch16_224_mae
- resnet50_mocov3
- vit_small_patch16_224_mocov3
- vit_base_patch16_224_mocov3
- vit_small_patch16_224_msn
- vit_base_patch16_224_msn
- vit_base_patch4_224_msn
- vit_large_patch16_224_msn
- vit_large_patch7_224_msn
"""
import math
from functools import partial
from typing import List, Optional

import timm
import torch
from timm.models import layers, resnet, vision_transformer
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from torch import nn


def _create_savi_resnet(block, stages, **kwargs) -> resnet.ResNet:
    """ResNet as used by SAVi and SAVi++, adapted from SAVi codebase.

    The differences to the normal timm ResNet implementation are to use group norm instead of batch
    norm, and to use 3x3 filters, 1x1 strides and no max pooling in the stem.

    Returns 16x16 feature maps for input size of 128x128, and 28x28 features maps for inputs of
    size 224x224.
    """
    model_args = dict(block=block, layers=stages, norm_layer=layers.GroupNorm, **kwargs)
    model = resnet._create_resnet("resnet34", pretrained=False, **model_args)
    model.conv1 = nn.Conv2d(
        model.conv1.in_channels,
        model.conv1.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.init_weights(zero_init_last=True)  # Re-init weights because we added a new conv layer
    return model


@timm.models.register_model
def resnet18_savi(pretrained=False, **kwargs):
    """ResNet18 as implemented in SAVi codebase.

    Final features have 512 channels.
    """
    if pretrained:
        raise ValueError("No pretrained weights available for `resnet18_savi`.")
    return _create_savi_resnet(resnet.BasicBlock, stages=[2, 2, 2, 2], **kwargs)


@timm.models.register_model
def resnet34_savi(pretrained=False, **kwargs):
    """ResNet34 as used in SAVi and SAVi++ papers.

    Final features have 512 channels.
    """
    if pretrained:
        raise ValueError("No pretrained weights available for `resnet34_savi`.")
    return _create_savi_resnet(resnet.BasicBlock, stages=[3, 4, 6, 3], **kwargs)


@timm.models.register_model
def resnet50_savi(pretrained=False, **kwargs):
    """ResNet50 as implemented in SAVi codebase.

    Final features have 2048 channels.
    """
    if pretrained:
        raise ValueError("No pretrained weights available for `resnet50_savi`.")
    return _create_savi_resnet(resnet.Bottleneck, stages=[3, 4, 6, 3], **kwargs)


def _resnet50_dino_pretrained_filter(state_dict, model):
    del model.fc
    return state_dict


@timm.models.register_model
def resnet50_dino(pretrained=False, **kwargs):
    """ResNet50 pre-trained with DINO, without classification head.

    Weights from https://github.com/facebookresearch/dino
    """
    kwargs["pretrained_cfg"] = resnet._cfg(
        url=(
            "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/"
            "dino_resnet50_pretrain.pth"
        )
    )
    model_args = dict(block=resnet.Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    model = build_model_with_cfg(
        resnet.ResNet,
        "resnet50_dino",
        pretrained=pretrained,
        pretrained_filter_fn=_resnet50_dino_pretrained_filter,
        **model_args,
    )
    return model


@timm.models.register_model
def vit_base_patch16_224_mae(pretrained=False, checkpoint_path=None, **kwargs):
    """ViT-B/16 pre-trained with MAE.

    Weights from https://github.com/facebookresearch/mae
    """
    if checkpoint_path is None:
        kwargs["pretrained_cfg"] = vision_transformer._cfg(
            url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
        )
    else:
        kwargs["pretrained_cfg"] = vision_transformer._cfg(file=checkpoint_path)
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    return vision_transformer._create_vision_transformer(
        "vit_base_patch16_224_mae", pretrained=pretrained, **model_kwargs
    )


@timm.models.register_model
def vit_large_patch16_224_mae(pretrained=False, **kwargs):
    """ViT-L/16 pre-trained with MAE.

    Weights from https://github.com/facebookresearch/mae
    """
    kwargs["pretrained_cfg"] = vision_transformer._cfg(
        url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth"
    )
    model_kwargs = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    return vision_transformer._create_vision_transformer(
        "vit_large_patch16_224_mae", pretrained=pretrained, **model_kwargs
    )


@timm.models.register_model
def vit_huge_patch14_224_mae(pretrained=False, **kwargs):
    """ViT-H/14 pre-trained with MAE.

    Weights from https://github.com/facebookresearch/mae
    """
    kwargs["pretrained_cfg"] = vision_transformer._cfg(
        url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth"
    )
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    return vision_transformer._create_vision_transformer(
        "vit_huge_patch14_224_mae", pretrained=pretrained, **model_kwargs
    )


def _add_moco_v3_positional_embedding(model, temperature=10000.0):
    """MoCo-v3 ViT uses 2D sincos embedding instead of learned positional encoding.

    Adapted from https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    """
    h, w = model.patch_embed.grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
    assert (
        model.embed_dim % 4 == 0
    ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"

    pos_dim = model.embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
    )[None, :, :]
    if hasattr(model, "num_tokens"):  # Old timm versions
        assert model.num_tokens == 1, "Assuming one and only one token, [cls]"
    else:
        assert model.num_prefix_tokens == 1, "Assuming one and only one token, [cls]"

    pe_token = torch.zeros([1, 1, model.embed_dim], dtype=torch.float32)
    model.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    model.pos_embed.requires_grad = False


def _moco_v3_pretrained_filter(state_dict, model, linear_name):
    state_dict = state_dict["state_dict"]

    for k in list(state_dict.keys()):
        # Retain only base_encoder up to before the embedding layer
        if k.startswith("module.base_encoder") and not k.startswith(
            f"module.base_encoder.{linear_name}"
        ):
            # Remove prefix
            state_dict[k[len("module.base_encoder.") :]] = state_dict[k]
        # Delete renamed or unused k
        del state_dict[k]

    if hasattr(model, "fc"):
        del model.fc

    return state_dict


def _create_moco_v3_vit(variant, pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(
        variant, pretrained_cfg=kwargs.pop("pretrained_cfg", None)
    )
    model = build_model_with_cfg(
        vision_transformer.VisionTransformer,
        variant,
        pretrained=pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=partial(_moco_v3_pretrained_filter, linear_name="head"),
        **kwargs,
    )
    _add_moco_v3_positional_embedding(model)
    return model


@timm.models.register_model
def vit_small_patch16_224_mocov3(pretrained=False, **kwargs):
    """ViT-S/16 pre-trained with MoCo-v3.

    Weights from https://github.com/facebookresearch/moco-v3
    """
    kwargs["pretrained_cfg"] = vision_transformer._cfg(
        url="https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
    )
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    return _create_moco_v3_vit("vit_small_patch16_224_mocov3", pretrained=pretrained, **model_kwargs)


@timm.models.register_model
def vit_base_patch16_224_mocov3(pretrained=False, **kwargs):
    """ViT-B/16 pre-trained with MoCo-v3.

    Weights from https://github.com/facebookresearch/moco-v3
    """
    kwargs["pretrained_cfg"] = vision_transformer._cfg(
        url="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
    )
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    return _create_moco_v3_vit("vit_base_patch16_224_mocov3", pretrained=pretrained, **model_kwargs)


@timm.models.register_model
def resnet50_mocov3(pretrained=False, **kwargs):
    """ResNet-50 pre-trained with MoCo-v3.

    Weights from https://github.com/facebookresearch/moco-v3
    """
    kwargs["pretrained_cfg"] = resnet._cfg(
        url="https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar"
    )
    model_args = dict(block=resnet.Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return build_model_with_cfg(
        resnet.ResNet,
        "resnet50_mocov3",
        pretrained=pretrained,
        pretrained_filter_fn=partial(_moco_v3_pretrained_filter, linear_name="fc"),
        **model_args,
    )


def _msn_vit_pretrained_filter(state_dict, model):
    state_dict = state_dict["target_encoder"]

    for k in list(state_dict.keys()):
        if not k.startswith("module.fc."):
            # remove prefix
            state_dict[k[len("module.") :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    return state_dict


def _create_msn_vit(variant, pretrained=False, **kwargs):
    pretrained_cfg = resolve_pretrained_cfg(
        variant, pretrained_cfg=kwargs.pop("pretrained_cfg", None)
    )
    model = build_model_with_cfg(
        vision_transformer.VisionTransformer,
        variant,
        pretrained=pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=_msn_vit_pretrained_filter,
        **kwargs,
    )
    return model


@timm.models.register_model
def vit_small_patch16_224_msn(pretrained=False, **kwargs):
    """ViT-S/16 pre-trained with MSN.

    Weights from https://github.com/facebookresearch/msn
    """
    kwargs["pretrained_cfg"] = vision_transformer._cfg(
        url="https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar"
    )
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    return _create_msn_vit("vit_small_patch16_224_msn", pretrained=pretrained, **model_kwargs)


@timm.models.register_model
def vit_base_patch16_224_msn(pretrained=False, **kwargs):
    """ViT-B/16 pre-trained with MSN.

    Weights from https://github.com/facebookresearch/msn
    """
    kwargs["pretrained_cfg"] = vision_transformer._cfg(
        url="https://dl.fbaipublicfiles.com/msn/vitb16_600ep.pth.tar"
    )
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    return _create_msn_vit("vit_base_patch16_224_msn", pretrained=pretrained, **model_kwargs)


@timm.models.register_model
def vit_base_patch4_224_msn(pretrained=False, **kwargs):
    """ViT-B/4 pre-trained with MSN.

    Weights from https://github.com/facebookresearch/msn
    """
    kwargs["pretrained_cfg"] = vision_transformer._cfg(
        url="https://dl.fbaipublicfiles.com/msn/vitb4_300ep.pth.tar"
    )
    model_kwargs = dict(
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    return _create_msn_vit("vit_base_patch4_224_msn", pretrained=pretrained, **model_kwargs)


@timm.models.register_model
def vit_large_patch16_224_msn(pretrained=False, **kwargs):
    """ViT-L/16 pre-trained with MSN.

    Weights from https://github.com/facebookresearch/msn
    """
    kwargs["pretrained_cfg"] = vision_transformer._cfg(
        url="https://dl.fbaipublicfiles.com/msn/vitl16_600ep.pth.tar"
    )
    model_kwargs = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    return _create_msn_vit("vit_large_patch16_224_msn", pretrained=pretrained, **model_kwargs)


@timm.models.register_model
def vit_large_patch7_224_msn(pretrained=False, **kwargs):
    """ViT-L/7 pre-trained with MSN.

    Weights from https://github.com/facebookresearch/msn
    """
    kwargs["pretrained_cfg"] = vision_transformer._cfg(
        url="https://dl.fbaipublicfiles.com/msn/vitl7_200ep.pth.tar"
    )
    model_kwargs = dict(
        patch_size=7,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    return _create_msn_vit("vit_large_patch7_224_msn", pretrained=pretrained, **model_kwargs)


def patch_timm_for_fx_tracing():
    """Patch timm to allow torch.fx tracing."""

    def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = "bicubic",
        antialias: bool = True,
        verbose: bool = False,
    ):
        """From timm.layers.pos_embed.resample_abs_pose_embed.

        To avoid control flow using dynamic variables, the check returning early for same size
        is not executed.
        """
        # sort out sizes, assume square if old size not provided
        num_pos_tokens = posemb.shape[1]

        # REMOVED because this relies on dynamic variables:
        # num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
        # if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        #    return posemb

        if old_size is None:
            hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
            old_size = hw, hw

        if num_prefix_tokens:
            posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
        else:
            posemb_prefix, posemb = None, posemb

        # do the interpolation
        embed_dim = posemb.shape[-1]
        orig_dtype = posemb.dtype
        posemb = posemb.float()  # interpolate needs float32
        posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
        posemb = nn.functional.interpolate(
            posemb, size=new_size, mode=interpolation, antialias=antialias
        )
        posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        posemb = posemb.to(orig_dtype)

        # add back extra (class, etc) prefix tokens
        if posemb_prefix is not None:
            posemb = torch.cat([posemb_prefix, posemb], dim=1)

        return posemb

    # Monkey patch method in vision transformer
    timm.models.vision_transformer.resample_abs_pos_embed = resample_abs_pos_embed


torch.fx.wrap("int")  # Needed to allow tracing with int()
patch_timm_for_fx_tracing()
