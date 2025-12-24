import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch


def to_vis(img, scale=True):
    """Convert img to np.uint8 array that can be used to visualization."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    elif isinstance(img, Image.Image):
        img = np.array(img)
    elif not isinstance(img, np.ndarray):
        raise TypeError(f'Unrecognized type: {type(img)}')
    # we assume pixel values are either [0., 1.], or [-1., 1.] or [0, 255]
    if scale:
        if img.min() >= 0. and img.max() <= 1.:
            img = np.round(img * 255.).astype(np.uint8)
        elif img.min() >= -1. and img.max() <= 1.:
            img = np.round((img + 1.) / 2. * 255.).astype(np.uint8)
        else:
            img = np.round(img).astype(np.uint8)
    return img


def plot_imgs_seq(imgs, figsize=None, show=True):
    """Plot images in different figures."""
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    imgs = [to_vis(img) for img in imgs]
    for i, img in enumerate(imgs):
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.title(f'{i}')

    if show:
        plt.show()


def plot_imgs_grid(imgs, grids=None, figsize=None, title=None, show=True):
    """Plot images in one figure using subplots."""
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    imgs = [to_vis(img) for img in imgs]
    if grids is None:
        num_imgs = len(imgs)
        x = int(np.sqrt(num_imgs))
        y = int(np.ceil(num_imgs / x))
        grids = (x, y)
    plt.figure(figsize=figsize)
    for i, img in enumerate(imgs):
        plt.subplot(grids[0], grids[1], i + 1)
        plt.imshow(img)

    if title:
        plt.title(title)

    if show:
        plt.show()
