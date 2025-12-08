# code from torchvision:
# https://github.com/pytorch/vision/blob/main/torchvision/transforms/_transforms_video.py

import numbers
import random

import einops
import torch
from torchvision.transforms import RandomCrop, RandomResizedCrop

__all__ = [
    "RandomCropVideo",
    "RandomResizedCropVideo",
    "CenterCropVideo",
    "NormalizeVideo",
    "ToTensorVideo",
    "RandomHorizontalFlipVideo",
]


class RandomCropVideo(RandomCrop):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, F, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, F, OH, OW)
        """
        i, j, h, w = self.get_params(clip, self.size)
        return crop(clip, i, j, h, w)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class RandomResizedCropVideo(RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
        self.scale = scale
        self.ratio = ratio

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, F, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, F, H, W)
        """
        i, j, h, w = self.get_params(clip, self.scale, self.ratio)
        return resized_crop(clip, i, j, h, w, self.size, self.interpolation_mode)

    def __repr__(self) -> str:
        repr_ = (
            f"{self.__class__.__name__}(size={self.size},"
            + f"interpolation_mode={self.interpolation_mode},"
            + f"scale={self.scale}, ratio={self.ratio})"
        )
        return repr_


class CenterCropVideo:
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, F, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, F, crop_size, crop_size)
        """
        return center_crop(clip, self.crop_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_size={self.crop_size})"


class CenterFullCropVideo:
    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video to be cropped. Size is (C, F, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, F, crop_size, crop_size)
        """
        min_size = int(min(clip.shape[2:]))
        crop_size = (min_size, min_size)
        return center_crop(clip, crop_size)


class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, F, H, W)
        """
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"


class DenormalizeVideo(NormalizeVideo):
    def __init__(self, mean, std):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std)


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (F, H, W, C)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (C, F, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class FromTensorVideo:
    """
    Permute the dimensions of clip tensor,  multiply value by 255.0 and
    convert tensor data type from float to unit8.
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.float): Size is (C, F, H, W)
        Return:
            clip (torch.tensor, dtype=torch.uint8): Size is (F, H, W, C)
        """
        return from_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class RandomHorizontalFlipVideo:
    """
    Flip the video clip along the horizonal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, F, H, W)
        Return:
            clip (torch.tensor): Size is (C, F, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, F, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i : i + h, j : j + w]


def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode, align_corners=False
    )


def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the video clip
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (C, F, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized clip
    Returns:
        clip (torch.tensor): Resized and cropped clip. Size is (C, F, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    clip = crop(clip, i, j, h, w)
    clip = resize(clip, size, interpolation_mode)
    return clip


def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (F, H, W, C)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (C, F, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    return clip.float().permute(3, 0, 1, 2) / 255.0


def from_tensor(clip):
    """
    Permute the dimensions of clip tensor  multiply value by 255.0 and
    convert tensor data type from float to uint8.

    Args:
        clip (torch.tensor, dtype=torch.float): Size is (C, F, H, W)
    Return:
        clip (torch.tensor, dtype=torch.uint8): Size is (F, H, W, C)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.float:
        raise TypeError("clip tensor should have data type float. Got %s" % str(clip.dtype))
    clip = einops.rearrange(clip, "C F H W -> F H W C")
    return (clip * 255.0).byte()


def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, F, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (C, F, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (C, F, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (C, F, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    return clip.flip(-1)
