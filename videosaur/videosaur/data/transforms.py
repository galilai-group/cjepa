import math
from collections.abc import Sequence
from functools import partial
from typing import Dict, Optional

import einops
import numpy as np
import torch
from einops.layers.torch import Rearrange
from torchvision import transforms as tvt
from torchvision.transforms import functional as tvt_functional

from videosaur.videosaur.data import transforms_video

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

MOVI_DEFAULT_MEAN = [0.5, 0.5, 0.5]
MOVI_DEFAULT_STD = [0.5, 0.5, 0.5]

DATASET_TYPES = {
    "coco": "image",
    "davis": "video",
    "ytvis": "video",
    "movi": "video",
    "clevrer": "video",
    "pusht": "video",
    "dummy": "video",
    "dummyimage": "image",
}


def build(config):
    dataset_split = config.name.split("_")
    assert len(dataset_split) == 2, "name should be in 'dataset_split' format"
    dataset, split = dataset_split
    assert dataset in DATASET_TYPES.keys(), f"{dataset} is not supported"
    assert split in ["train", "val", "test"]
    dataset_type = DATASET_TYPES[dataset]
    transform_type = config.get("type", "video")
    crop_type = config.get("crop_type", None)
    h_flip_prob = config.get("h_flip_prob", None)
    size = _to_2tuple(config.input_size)
    mask_size = _to_2tuple(config.mask_size) if config.get("mask_size") else size
    use_movi_normalization = config.get("use_movi_normalization", False)
    use_segmentations = config.get("use_segmentations", True)

    if dataset_type not in ("image", "video"):
        raise ValueError(f"Unsupported dataset type {transform_type}")
    if transform_type not in ("image", "video"):
        raise ValueError(f"Unsupported transform type {transform_type}")
    if transform_type == "video":
        assert dataset_type == "video"
    if dataset_type == "image":
        assert transform_type == "image"

    if crop_type is not None:
        if split == "train":
            assert crop_type in [
                "central",
                "random",
                "short_side_resize_random",
                "short_side_resize_central",
            ]

        resize_input = CropResize(
            dataset_type=dataset_type,
            crop_type=crop_type,
            size=size,
            resize_mode="bicubic",
            clamp_zero_one=True,
        )
        if split in ["val", "test"]:
            assert crop_type in [
                "central",
                "short_side_resize_central",
            ], f"Only central crops are supported for {split}."
            assert h_flip_prob is None, f"Horizontal flips are not supported for {split}."
            resize_segmentation = CropResize(
                dataset_type=dataset_type,
                crop_type=crop_type,
                size=mask_size,
                resize_mode="nearest-exact",
            )
    else:
        resize_input = Resize(size=size, mode="bicubic", clamp_zero_one=True)
        resize_segmentation = Resize(size=mask_size, mode="nearest-exact")

    if use_movi_normalization:
        normalize = Normalize(
            dataset_type=dataset_type, mean=MOVI_DEFAULT_MEAN, std=MOVI_DEFAULT_STD
        )
    else:
        normalize = Normalize(
            dataset_type=dataset_type, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )
    input_transform = tvt.Compose(
        [
            ToTensorInput(dataset_type=dataset_type),
            resize_input,
            normalize,
        ]
    )
    if h_flip_prob is not None:
        input_transform.transforms.append(
            RandomHorizontalFlip(dataset_type=dataset_type, p=h_flip_prob)
        )
    segmentation_transformation = None
    if split == "val" and use_segmentations:
        segmentation_transformation = tvt.Compose(
            [
                ToTensorMask(),
                DenseToOneHotMask(num_classes=config.num_classes),
                resize_segmentation,
            ]
        )
    transforms = {transform_type: input_transform}

    if dataset == "movi":
        if "target_size" in config:
            target_size = _to_2tuple(config.target_size)
            transforms[f"target_{transform_type}"] = tvt.Compose(
                [
                    ToTensorInput(dataset_type=dataset_type),
                    Resize(size=target_size, mode="bicubic", clamp_zero_one=True),
                    normalize,
                ]
            )
        if split == "val" and segmentation_transformation is not None:
            transforms["segmentations"] = segmentation_transformation

    # just copy from movi for now
    elif dataset == "clevrer" or dataset == "pusht":
        if "target_size" in config:
            target_size = _to_2tuple(config.target_size)
            transforms[f"target_{transform_type}"] = tvt.Compose(
                [
                    ToTensorInput(dataset_type=dataset_type),
                    Resize(size=target_size, mode="bicubic", clamp_zero_one=True),
                    normalize,
                ]
            )
        if split == "val" and segmentation_transformation is not None:
            transforms["segmentations"] = segmentation_transformation
    elif dataset == "davis":
        if "target_size" in config:
            raise NotImplementedError("Separate targets not implemented for transform `davis`")
        if split == "val" and use_segmentations:
            transforms["segmentations"] = tvt.Compose(
                [
                    ToTensorMask(),
                    DenseToOneHotMask(num_classes=config.num_classes, remove_zero_masks=True),
                    resize_segmentation,
                ]
            )
    elif dataset == "coco":
        if "target_size" in config:
            raise NotImplementedError("Separate targets not implemented for transform `coco`")
        if split == "val" and use_segmentations:
            transforms["segmentations"] = tvt.Compose(
                [COCOToBinary(num_classes=config.num_classes), resize_segmentation]
            )
    elif dataset == "ytvis":
        if "target_size" in config:
            raise NotImplementedError("Separate targets not implemented for transform `ytvis`")

        if split == "val" and use_segmentations:
            transforms["segmentations"] = tvt.Compose(
                [YTVISToBinary(num_classes=config.num_classes), resize_segmentation]
            )


    elif dataset == "dummy" or dataset == "dummyimage":
        if transform_type == "image":
            transforms["image"] = tvt.Compose(
                [
                    tvt.ToTensor(),
                    normalize,
                ]
            )
            transforms["masks"] = tvt.Compose(
                [
                    ToTensorMask(),
                    DenseToOneHotMask(num_classes=config.num_classes),
                ]
            )
        elif transform_type == "video":
            transforms["video"] = tvt.Compose(
                [
                    ToTensorInput(dataset_type=dataset_type),
                    normalize,
                ]
            )
            transforms["masks"] = tvt.Compose(
                [
                    ToTensorMask(),
                    DenseToOneHotMask(num_classes=config.num_classes),
                ]
            )
    else:
        raise ValueError(f"Unknown dataset transforms module `{dataset}`")
    if dataset != "dummy":
        # At this point in the transforms, videos are in CFHW format.
        # Now reorder to FCHW format.
        if dataset_type == "video":
            transforms[transform_type].transforms.append(CFHWToFCHWFormat())
            if f"target_{transform_type}" in transforms:
                transforms[f"target_{transform_type}"].transforms.append(CFHWToFCHWFormat())

        # We transform image-video datasets as one frame video
        # So need to remove first dimension in the end
        if transform_type == "image" and dataset_type == "video":
            squeeze_video_dim = partial(torch.squeeze, dim=0)
            for tf in transforms.values():
                tf.transforms.append(squeeze_video_dim)

    return transforms

def build_inference_transform(config):
    """Builds the transform for inference.
    
    Modity if needed to match the preprocessing needed for your video.
    """
    use_movi_normalization = config.get("use_movi_normalization", True)
    size = config.get("input_size", 224)
    dataset_type = config.get("dataset_type", "video")
    
    resize_input = CropResize(
            dataset_type=dataset_type,
            crop_type="central",
            size=size,
            resize_mode="bilinear",
            clamp_zero_one=False,
        )

    if use_movi_normalization:
        normalize = Normalize(
            dataset_type=dataset_type, mean=MOVI_DEFAULT_MEAN, std=MOVI_DEFAULT_STD
        )
    else:
        normalize = Normalize(
            dataset_type=dataset_type, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )
    input_transform = tvt.Compose(
        [
            resize_input,
            normalize,
        ]
    )
    return  input_transform


def _to_2tuple(val):
    if val is None:
        return None
    elif isinstance(val, int):
        return (val, val)
    else:
        return val


class CFHWToFCHWFormat:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.permute(tensor, dims=(1, 0, 2, 3))


class ToTensorInput:
    def __init__(self, dataset_type) -> None:
        self.dataset_type = dataset_type
        if dataset_type == "video":
            self.to_cfhw_tensor = transforms_video.ToTensorVideo()

    def __call__(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array)
        if self.dataset_type == "video":
            tensor = self.to_cfhw_tensor(tensor)
        elif self.dataset_type == "image":
            tensor = tvt_functional.convert_image_dtype(tensor, dtype=torch.float)
            tensor = einops.rearrange(tensor, "h w c -> c h w")
        return tensor

# Note : not recommended. Use temporal_subsample_keys from pipelines.py instead
# class TemporalSubsample:
#     """Temporal subsampling for raw video arrays.

#     Operates on numpy arrays with shape (F, H, W, C) and returns a subsampled
#     numpy array using the stride `num_frameskip`: arr[0::stride]
#     """

#     def __init__(self, stride: int = 1):
#         assert isinstance(stride, int) and stride >= 1
#         self.stride = stride

#     def __call__(self, array: np.ndarray) -> np.ndarray:
#         if array is None:
#             return None
#         # If array has fewer than 1 dim, return as-is
#         if not hasattr(array, "shape") or len(array.shape) == 0:
#             return array
#         # Expecting shape (F, H, W, C) — slice first axis
#         return array[0:: self.stride]


class Normalize:
    def __init__(self, dataset_type: str, mean, std):
        print(dataset_type)
        if dataset_type == "image":
            self.norm = tvt.Normalize(mean=mean, std=std)
        elif dataset_type == "video":
            self.norm = transforms_video.NormalizeVideo(mean=mean, std=std)
        else:
            ValueError(f"Not valid dataset type: {dataset_type}")

    def __call__(self, tensor) -> torch.Tensor:
        return self.norm(tensor)


class RandomHorizontalFlip:
    """
    Flip the video or image clip along the horizonal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, dataset_type: str, p: float = 0.5):
        if dataset_type == "image":
            self.flip = tvt.RandomHorizontalFlip(p)
        elif dataset_type == "video":
            self.flip = transforms_video.RandomHorizontalFlipVideo(p)
        else:
            ValueError(f"Not valid dataset type: {dataset_type}")

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.flip(tensor)


class CropResize:
    def __init__(
        self,
        dataset_type: str,
        crop_type: str,
        size: int,
        resize_mode: str,
        clamp_zero_one: bool = False,
        crop_params: Optional[Dict] = None,
    ):
        assert crop_type in [
            "random",
            "central",
            "short_side_resize_central",
            "short_side_resize_random",
        ]
        if crop_type == "random" and crop_params is None:
            crop_params = {}

        if dataset_type == "video":
            self.crop_resize = get_video_crop_resize(
                crop_type, crop_params, size, resize_mode, clamp_zero_one
            )

        elif dataset_type == "image":
            self.crop_resize = get_image_crop_resize(
                crop_type, crop_params, size, resize_mode, clamp_zero_one
            )
        else:
            raise ValueError(f"Unknown dataset_type dataset_type={dataset_type}")
        if clamp_zero_one and crop_type == "random" and resize_mode == "bicubic":
            # we clamp here only random crops
            # because central are already croped in Resize
            self.crop_resize = tvt.Compose([self.crop_resize, partial(torch.clamp, min=0, max=1)])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.crop_resize(tensor)


def get_image_crop_resize(crop_type, crop_params, size, resize_mode, clamp_zero_one):
    if crop_type == "short_side_resize_random":
        short_side_resize = Resize(
            size,
            resize_mode,
            clamp_zero_one=clamp_zero_one,
            short_side_scale=True,
        )
        crop = tvt.RandomCrop(size=size)
        return tvt.Compose([short_side_resize, crop])
    elif crop_type == "central":
        central_crop = CenterFullCrop()
        resize = Resize(size, resize_mode, clamp_zero_one=clamp_zero_one)
        return tvt.Compose([central_crop, resize])
    elif crop_type == "short_side_resize_central":
        short_side_resize = Resize(
            size, resize_mode, clamp_zero_one=clamp_zero_one, short_side_scale=True
        )
        central_crop = CenterFullCrop()
        return tvt.Compose([short_side_resize, central_crop])
    elif crop_type == "random":
        return tvt.RandomResizedCrop(
            size=size,
            interpolation=tvt_functional.InterpolationMode[resize_mode.upper()],
            **crop_params,
        )
    else:
        ValueError(f"Not valid crop_type {crop_type}")


def get_video_crop_resize(crop_type, crop_params, size, resize_mode, clamp_zero_one):
    if crop_type == "central":
        central_crop = transforms_video.CenterFullCropVideo()
        resize = Resize(size, resize_mode, clamp_zero_one=clamp_zero_one)
        return tvt.Compose([central_crop, resize])
    elif crop_type == "short_side_resize_central":
        short_side_resize = Resize(
            size,
            resize_mode,
            clamp_zero_one=clamp_zero_one,
            short_side_scale=True,
        )
        central_crop = transforms_video.CenterFullCropVideo()
        return tvt.Compose([short_side_resize, central_crop])
    elif crop_type == "random":
        return transforms_video.RandomResizedCropVideo(
            size=size,
            interpolation_mode=resize_mode,
            **crop_params,
        )
    elif crop_type == "short_side_resize_random":
        short_side_resize = Resize(
            size,
            resize_mode,
            clamp_zero_one=clamp_zero_one,
            short_side_scale=True,
        )
        crop = transforms_video.RandomCropVideo(size)
        return tvt.Compose(
            [
                short_side_resize,
                crop,
            ]
        )
    else:
        ValueError(f"Not valid crop_type {crop_type}")


class Resize:
    def __init__(
        self,
        size: int,
        mode: str,
        clamp_zero_one: bool = False,
        short_side_scale: bool = False,
    ):

        self.mode = mode
        self.clamp_zero_one = clamp_zero_one
        self.short_side_scale = short_side_scale
        if short_side_scale:
            if isinstance(size, Sequence):
                assert size[0] == size[1]
                self.size = size[0]
            elif isinstance(size, int):
                self.size = size
            else:
                raise ValueError(f"size should be int or tuple but got {size}")
        else:
            self.size = size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        is_image = len(tensor.shape) == 3
        if is_image:
            tensor = tensor[None]
        is_bool = False
        if tensor.dtype == torch.bool:
            is_bool = True
            tensor = tensor.to(torch.uint8)
        if self.short_side_scale:
            tensor = self.scale_short_side(tensor)
        else:
            tensor = torch.nn.functional.interpolate(tensor, size=self.size, mode=self.mode)
        if self.clamp_zero_one and self.mode == "bicubic":
            tensor = torch.clamp(tensor, min=0, max=1)
        if is_bool:
            tensor = tensor.to(torch.bool)
        if is_image:
            tensor = tensor[0]
        return tensor

    def scale_short_side(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scales the shorter spatial dim to the given size.

        To maintain aspect ratio, the longer side is then scaled accordingly.

        Args:
            tensor: A 4D tensor of shape (F, C, H, W) or (B, C, H, W)

        Returns:
            tensor: Tensor with scaled spatial dims.
        """
        assert len(tensor.shape) == 4
        _, _, h, w = tensor.shape
        if w < h:
            new_h = int(math.floor((float(h) / w) * self.size))
            new_w = self.size
        else:
            new_h = self.size
            new_w = int(math.floor((float(w) / h) * self.size))

        return torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode=self.mode)


class CenterFullCrop:
    def __call__(self, img):
        """
        Args:
            image (torch.tensor): Image to be cropped. Size is (C, H, W)
        Returns:
            torch.tensor: central cropping of image. Size is
            (C, crop_size, crop_size)
        """
        min_size = int(min(img.shape[1:]))
        crop_size = (min_size, min_size)
        return tvt.functional.center_crop(img, crop_size)


class ToTensorMask:
    """Transform masks from numpy array uint8 array to float tensor."""

    def __call__(self, mask: np.ndarray):
        assert mask.shape[-1] == 1
        return torch.from_numpy(mask).squeeze(-1).to(torch.float32)


class YTVISToBinary:
    """Transform YTVIS masks to stardart binary form with shape (I, H, W)."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, mask: torch.Tensor):
        f, h, w, num_obj = mask.shape
        mask_binary = torch.zeros(f, h, w, self.num_classes, dtype=torch.bool)
        mask = torch.from_numpy(mask != 0).to(torch.bool)
        mask_binary[..., :num_obj] = mask
        mask_binary = einops.rearrange(mask_binary, "f h w i -> f i h w")
        return mask_binary


class COCOToBinary:
    """Transform COCO masks to stardart binary form with shape (I, H, W)."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, mask: torch.Tensor):
        num_obj, h, w = mask.shape
        mask_binary = torch.zeros(self.num_classes, h, w, dtype=torch.bool)
        mask = torch.from_numpy(mask != 0).to(torch.bool)
        mask_binary[:num_obj] = mask
        return mask_binary


class DenseToOneHotMask:
    """Transform dense mask of shape (..., H, W) to (..., I, H, W).

    Potentially removes one-hot dim that corresponds to zeros.
    It is useful if all the instances are non-zero encoded,
    whereas zeros correspond to unlabeled pixels (e.g. in DAVIS dataset).
    """

    def __init__(self, num_classes: int, remove_zero_masks: bool = False):
        self.num_classes = num_classes
        self.remove_zero_masks = remove_zero_masks

    def __call__(self, mask: torch.Tensor):
        if self.remove_zero_masks:
            mask_oh = torch.nn.functional.one_hot(mask.to(torch.long), self.num_classes + 1)
            mask_oh = mask_oh[..., 1:]
        else:
            mask_oh = torch.nn.functional.one_hot(mask.to(torch.long), self.num_classes)
        if mask_oh.dim() == 3:
            mask_oh = einops.rearrange(mask_oh, "h w i -> i h w")
        elif mask_oh.dim() == 4:
            mask_oh = einops.rearrange(mask_oh, "f h w i ->f i h w")
        return mask_oh.to(torch.bool)


class Denormalize:
    """
    Denormalization transform for both image and video inputs.

    In case of videos, expected format is FCHW
    as we apply Denormalize after switch from CFHW to FCHW.
    """

    def __init__(self, input_type, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD) -> None:
        if input_type == "video":
            denormalize = transforms_video.DenormalizeVideo(mean=mean, std=std)
            self.denormalize = tvt.Compose(
                [
                    Rearrange("F C H W -> C F H W"),
                    denormalize,
                    Rearrange("C F H W -> F C H W"),
                ]
            )

        elif input_type == "image":
            self.denormalize = DenormalizeImage(mean=mean, std=std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.denormalize(tensor)


class DenormalizeImage(tvt.Normalize):
    def __init__(self, mean, std):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std)
