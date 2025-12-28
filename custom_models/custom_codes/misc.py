import os
import cv2
import time
import random
import numpy as np

import torch
from torchmetrics import MeanMetric as TorchMeanMetric

from nerv.utils.tensor import to_numpy


def datetime2str(form='%Y-%m-%d_%H-%M-%S'):
    datetime = time.strftime(form, time.localtime())
    return datetime


def assert_array_shape(xyz, shapes=()):
    """Check array shape.

    Args:
        xyz (np.ndarray): array
        shape (tuple of tuple of ints, optional): possible target shapes,
            -1 means arbitrary. Defaults to ((-1, 3)).
    """
    if not shapes:
        raise ValueError('"shapes" cannot be empty')

    if isinstance(shapes[0], int):
        shapes = (shapes, )

    flags = {x: True for x in range(len(shapes))}
    for idx, shape in enumerate(shapes):
        if len(xyz.shape) != len(shape):
            flags[idx] = False

        for dim, num in enumerate(shape):
            if num == -1:
                continue
            elif xyz.shape[dim] != num:
                flags[idx] = False
    if sum(flags.values()) == 0:  # None of the possible shape works
        raise ValueError(
            f'Input array {xyz.shape} is not in target shapes {shapes}!')


def array_equal(a, b):
    """Compare if two arrays are the same.

    Args:
        a/b: can be np.ndarray or torch.Tensor.
    """
    if a.shape != b.shape:
        return False
    try:
        assert (a == b).all()
        return True
    except:
        return False


def array_in_list(array, lst):
    """Judge whether an array is in a list."""
    for v in lst:
        if array_equal(array, v):
            return True
    return False


def set_seed(seed=1, deterministic=False):
    """Set the random seed of the environment for reproducibility.

    Args:
        seed (int): the random seed to set.
        deterministic (bool, optional): whether to use deterministic torch
            backend. Default: False.
    """
    print('Using random seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        print('Using deterministic pytorch backends')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, device=torch.device('cpu')):
        self.reset()
        self.device = device

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def compute(self):
        return torch.tensor(self.avg).to(self.device)

    def to(self, device):
        self.device = device
        return self


class MeanMetric:
    """Compatibility wrapper around `torchmetrics.MeanMetric`.

    Provides a small API expected by the codebase: `update(value, weight=1.0)`,
    `compute()` and `to(device)`. Works with torchmetrics 1.x and 2.x.
    """

    def __init__(self, device, nan_strategy="warn", compute_on_step=True, dist_sync_on_step=False, process_group=None, dist_sync_fn=None):
        self.device = device
        # create underlying torchmetrics MeanMetric and move to device
        self._metric = TorchMeanMetric()
        try:
            self._metric.to(device)
        except Exception:
            pass

    def update(self, value, weight=1.0):
        # torchmetrics MeanMetric.update accepts (value, weight) in recent versions,
        # older variations may accept only value — try both.
        try:
            return self._metric.update(value, weight)
        except TypeError:
            return self._metric.update(value)

    def compute(self):
        out = self._metric.compute()
        if isinstance(out, torch.Tensor):
            return out.to(self.device)
        try:
            return torch.tensor(out).to(self.device)
        except Exception:
            return out

    def to(self, device):
        try:
            self._metric.to(device)
        except Exception:
            pass
        self.device = device
        return self


def _convert4save_img(array):
    """Convert a image array to be saveable."""
    # extend channel axis
    if len(array.shape) == 2:
        array = np.stack([array] * 3, axis=-1)
    # [C, H, W] --> [H, W, C], where C == 3
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    return np.ascontiguousarray(array)


def _convert4save_video(array):
    """Convert a video array to be saveable."""
    # extend channel axis
    if len(array.shape) == 3:
        array = np.stack([array] * 3, axis=-1)
    # [T, C, H, W] --> [T, H, W, C], where C == 3
    if array.shape[1] == 3:
        array = array.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(array)


def convert4save(array, is_video=False):
    """Check the dtype and value range of input array for save.

    Need to convert to [(T), H, W, C] with np.uint8 value range [0, 255].

    Args:
        array (np.ndarray or torch.Tensor): array to be converted.
        is_video (bool, optional): whether the array is a video or image.
            Default: False (means array is an image).

    Returns:
        np.ndarray: the converted array ready for save (image or video).
    """
    array = to_numpy(array)
    if 'int' in str(array.dtype):
        assert 0 <= array.min() <= array.max() <= 255
    elif 'float' in str(array.dtype):
        assert 0. <= array.min() <= array.max() <= 1.
        array = np.round(array * 255.).astype(np.uint8)
    if is_video:
        return _convert4save_video(array)
    else:
        return _convert4save_img(array)


def save_video(video, save_path, fps=30, codec='mp4v'):
    """video: np.ndarray of shape [M, H, W, 3]"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    video = convert4save(video, is_video=True)
    video = video[..., [2, 1, 0]]
    H, W = video.shape[-3:-1]
    assert save_path.split('.')[-1] == 'mp4'  # save as mp4 file
    # opencv has opposite dimension definition as numpy
    size = [W, H]
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*codec), fps, size)
    for i in range(video.shape[0]):
        out.write(video[i])
    out.release()
