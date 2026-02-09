import functools
import os
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import ImageColor, Image
from third_party.videosaur.videosaur.data.transforms import Resize
import seaborn as sns
import cv2

CMAP_STYLE = "tab"


def get_cmap_style() -> str:
    cmap = os.environ.get("VIDEOSAUR_CMAP")
    if cmap is None:
        cmap = CMAP_STYLE

    if cmap not in ("tab", "generated"):
        raise ValueError(f"Invalid color map {cmap}")

    return cmap


def mix_videos_with_masks(
    video: torch.Tensor, masks: torch.Tensor, alpha: float = 0.4
) -> torch.Tensor:
    input_shape = video.shape
    cmap = color_map(masks.shape[2])
    video = (255 * video).flatten(0, 1).to(torch.uint8)
    masks = masks.flatten(0, 1).to(bool)
    input_with_masks = torch.stack(
        [
            draw_segmentation_masks_on_image(frame, mask, colors=cmap, alpha=alpha)
            for frame, mask in zip(video, masks)
        ]
    )
    return input_with_masks.reshape(*input_shape)


def mix_images_with_masks(
    image: torch.Tensor, masks: torch.Tensor, alpha: float = 0.4
) -> torch.Tensor:
    input_shape = image.shape
    cmap = color_map(masks.shape[1])
    image = (255 * image).to(torch.uint8)
    masks = masks.to(bool)
    input_with_masks = torch.stack(
        [
            draw_segmentation_masks_on_image(frame, mask, colors=cmap, alpha=alpha)
            for frame, mask in zip(image, masks)
        ]
    )
    return input_with_masks.reshape(*input_shape)


def draw_segmentation_masks_on_image(
    image: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.8,
    colors: Optional[
        Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]
    ] = None,
) -> torch.Tensor:
    """
    Draws segmentation masks on given RGB image.

    The values of the input image should be uint8 between 0 and 255.

    Adapted from torchvision.utils.draw_segmentation_masks to run on GPUs if needed.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (color or list of colors, optional): List containing the colors
            of the masks or single color for all masks. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (num_masks, H, W)")
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
    if masks.shape[-2:] != image.shape[-2:]:
        image_resized = np.transpose(image.numpy(), (1,2,0))
        image_resized = cv2.resize(image_resized, (224, 224), interpolation=cv2.INTER_AREA)
        trans = np.transpose(image_resized, (2,0,1))
        image = torch.tensor(trans, dtype=torch.uint8)
        # raise ValueError(
        #     (
        #         "The image and the masks must have the same height and width,"
        #         + f"but got {masks.shape[-2:]} and {image.shape[-2:]}"
        #     )
        # )

    num_masks = masks.size()[0]
    if colors is not None and num_masks > len(colors):
        raise ValueError(f"There are more masks ({num_masks}) than colors ({len(colors)})")

    if num_masks == 0:
        warnings.warn("masks doesn't contain any mask. No mask was drawn", stacklevel=0)
        return image

    if colors is None:

        def generate_color_palette(num_objects: int):
            palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
            return [tuple((i * palette) % 255) for i in range(num_objects)]

        colors = generate_color_palette(num_masks)

    if not isinstance(colors, list):
        colors = [colors]
    if not isinstance(colors[0], (tuple, str)):
        raise ValueError("colors must be a tuple or a string, or a list thereof")
    if isinstance(colors[0], tuple) and len(colors[0]) != 3:
        raise ValueError("It seems that you passed a tuple of colors instead of a list of colors")

    out_dtype = torch.uint8

    colors_ = []
    for color in colors:
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        colors_.append(torch.tensor(color, dtype=out_dtype, device=image.device))

    img_to_draw = image.detach().clone()
    for mask, color in zip(masks, colors_):
        img_to_draw[:, mask] = color[:, None]

    out = image * (1 - alpha) + img_to_draw * alpha
    return out.to(out_dtype)


_TAB10_DATA = (
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # 1f77b4
    (1.0, 0.4980392156862745, 0.054901960784313725),  # ff7f0e
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # 2ca02c
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # d62728
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # 9467bd
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),  # 8c564b
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  # e377c2
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),  # 7f7f7f
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # bcbd22
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  # 17becf
)

_TAB20_DATA = (
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # 1f77b4
    (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),  # aec7e8
    (1.0, 0.4980392156862745, 0.054901960784313725),  # ff7f0e
    (1.0, 0.7333333333333333, 0.47058823529411764),  # ffbb78
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # 2ca02c
    (0.596078431372549, 0.8745098039215686, 0.5411764705882353),  # 98df8a
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # d62728
    (1.0, 0.596078431372549, 0.5882352941176471),  # ff9896
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # 9467bd
    (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),  # c5b0d5
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),  # 8c564b
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451),  # c49c94
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  # e377c2
    (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),  # f7b6d2
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),  # 7f7f7f
    (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),  # c7c7c7
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # bcbd22
    (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),  # dbdb8d
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  # 17becf
    (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),  # 9edae5
)

_TAB20B_DATA = (
    (0.2235294117647059, 0.23137254901960785, 0.4745098039215686),  # 393b79
    (0.3215686274509804, 0.32941176470588235, 0.6392156862745098),  # 5254a3
    (0.4196078431372549, 0.43137254901960786, 0.8117647058823529),  # 6b6ecf
    (0.611764705882353, 0.6196078431372549, 0.8705882352941177),  # 9c9ede
    (0.38823529411764707, 0.4745098039215686, 0.2235294117647059),  # 637939
    (0.5490196078431373, 0.6352941176470588, 0.3215686274509804),  # 8ca252
    (0.7098039215686275, 0.8117647058823529, 0.4196078431372549),  # b5cf6b
    (0.807843137254902, 0.8588235294117647, 0.611764705882353),  # cedb9c
    (0.5490196078431373, 0.42745098039215684, 0.19215686274509805),  # 8c6d31
    (0.7411764705882353, 0.6196078431372549, 0.2235294117647059),  # bd9e39
    (0.9058823529411765, 0.7294117647058823, 0.3215686274509804),  # e7ba52
    (0.9058823529411765, 0.796078431372549, 0.5803921568627451),  # e7cb94
    (0.5176470588235295, 0.23529411764705882, 0.2235294117647059),  # 843c39
    (0.6784313725490196, 0.28627450980392155, 0.2901960784313726),  # ad494a
    (0.8392156862745098, 0.3803921568627451, 0.4196078431372549),  # d6616b
    (0.9058823529411765, 0.5882352941176471, 0.611764705882353),  # e7969c
    (0.4823529411764706, 0.2549019607843137, 0.45098039215686275),  # 7b4173
    (0.6470588235294118, 0.3176470588235294, 0.5803921568627451),  # a55194
    (0.807843137254902, 0.42745098039215684, 0.7411764705882353),  # ce6dbd
    (0.8705882352941177, 0.6196078431372549, 0.8392156862745098),  # de9ed6
)

# This colormap first contains the tab10 colors, then every second color of the tab20 colors, and
# then the colors of tab20b
_OUR_TAB_DATA = (
    _TAB10_DATA
    + _TAB20_DATA[1::2]
    + _TAB20B_DATA[::4]
    + _TAB20B_DATA[1::4]
    + _TAB20B_DATA[2::4]
    + _TAB20B_DATA[3::4]
)


@functools.lru_cache
def color_map(N, normalized=False):
    cmap_style = get_cmap_style()
    if cmap_style == "tab" and N <= len(_OUR_TAB_DATA):
        cmap = np.array(_OUR_TAB_DATA[:N], dtype=np.float32)
        if N >= 8:
            # Replace dark gray with a darkish pink, namely the 6th color of Accent
            cmap[7] = (0.94117647058823528, 0.00784313725490196, 0.49803921568627452)
        if N >= 18:
            # Replace light gray with a red-brown, namely the 12th color of Paired
            cmap[17] = (0.69411764705882351, 0.34901960784313724, 0.15686274509803921)
        if not normalized:
            cmap = (cmap * 255).astype(np.uint8)
    else:
        cmap = generate_color_map(N, normalized)

    return [tuple(c) for c in cmap]


def generate_color_map(N, normalized=False):
    dtype = np.float32 if normalized else np.uint8

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = (r, g, b)

    cmap = cmap / 255 if normalized else cmap

    return cmap

def create_grid_frame(frames, grid_size=(2, 6), image_size=(224, 224), padding=2):
    # Initialize an empty frame with padding
    grid_frame = np.zeros((grid_size[0] * (image_size[0] + padding) - padding, 
                           grid_size[1] * (image_size[1] + padding) - padding), dtype=np.float64)
    
    for index, frame in enumerate(frames):
        row = index // grid_size[1]
        col = index % grid_size[1]
        start_row = row * (image_size[0] + padding)
        start_col = col * (image_size[1] + padding)
        grid_frame[start_row:start_row+image_size[0], start_col:start_col+image_size[1]] = frame
    
    return grid_frame

def create_grid_frame_rgb(frames, grid_size=(2, 6), image_size=(224, 224), padding=2):
    """
    Create a grid frame from individual RGB frames.

    Args:
        frames (list of np.ndarray): List of frames, each frame should be of shape (height, width, 3).
        grid_size (tuple): The grid size as (rows, columns).
        image_size (tuple): The size of each image in the grid as (height, width).
        padding (int): The padding size between images in the grid.

    Returns:
        np.ndarray: An image of the grid.
    """
    # Initialize an empty frame with padding for RGB channels
    grid_frame = np.zeros((
        grid_size[0] * (image_size[0] + padding) - padding, 
        grid_size[1] * (image_size[1] + padding) - padding, 
        3),  # Depth of 3 for RGB
        dtype=np.float32)
    
    for index, frame in enumerate(frames):
        if frame.ndim < 3:
            raise ValueError("All frames must have 3 dimensions (height, width, channels)")
        if frame.shape[2] != 3:
            raise ValueError("All frames must be RGB with 3 channels")
        
        row = index // grid_size[1]
        col = index % grid_size[1]
        start_row = row * (image_size[0] + padding)
        start_col = col * (image_size[1] + padding)
        end_row = start_row + image_size[0]
        end_col = start_col + image_size[1]
        
        # Check if frame resizing is needed
        # assert frame.shape[:2] == image_size
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        
        grid_frame[start_row:end_row, start_col:end_col, :] = frame
    
    return grid_frame

def _prepare_masks_and_video(inputs, outputs, softmasks=False):
    b, f, n_slots, hw = outputs["decoder"]["masks"].shape
    h = int(np.sqrt(hw))
    w = h
    masks_video = outputs["decoder"]["masks"].reshape(b, f, n_slots, h, w)
    assert b == 1, "Batch size must be 1 for visualization"
    masks_video = masks_video.squeeze(0)
    
    #resize masks to 224x224
    resizer = Resize(224, mode='bilinear')
    masks_video = resizer(masks_video)
    
    if not softmasks:
        ind = torch.argmax(masks_video, dim=1, keepdim=True)
        masks_video = torch.zeros_like(masks_video)
        masks_video.scatter_(1, ind, 1)

    video = inputs["video_visualization"]  # (B, C, T, H, W)
    return masks_video, video


def mix_inputs_with_masks(inputs, outputs, softmasks=False):
    """
    Create the original grid video showing the input and each slot-masked frame separately.
    """
    masks_video, video = _prepare_masks_and_video(inputs, outputs, softmasks)
    n_slots = masks_video.shape[1]

    masked_video_frames = []
    for t in range(masks_video.shape[0]):  # Iterate through each time step
        frames = [masks_video[t, i] for i in range(n_slots)]  # Get all frames for this time step
        frames = [np.ones_like(frames[0])] + frames  # First slot is the original (all ones mask)
        grid_frame = create_grid_frame(frames)
        grid_frame_rgb = np.repeat(grid_frame[:, :, np.newaxis], 3, axis=2)

        video_frames = [video[0, :, t].permute(1, 2, 0).numpy() for _ in range(n_slots + 1)]
        grid_video = create_grid_frame_rgb(video_frames)
        masked_video = (grid_video * grid_frame_rgb * 255).astype(np.uint8)
        masked_video_frames.append(masked_video)
    return masked_video_frames


def overlay_inputs_with_masks(inputs, outputs, softmasks=False, alpha=0.4):
    """
    Overlay slot masks on the original video, producing a single video with colored attention regions.
    """
    masks_video, video = _prepare_masks_and_video(inputs, outputs, softmasks)
    n_slots = masks_video.shape[1]
    palette = sns.color_palette("deep", n_slots)
    colors = [tuple(int(c * 255) for c in color) for color in palette]

    masked_video_frames = []
    for t in range(masks_video.shape[0]):
        frame = (video[0, :, t] * 255).to(torch.uint8)
        masks = masks_video[t]
        masks_bool = masks > 0.5 if softmasks else masks.bool()
        overlay = draw_segmentation_masks_on_image(frame, masks_bool, colors=colors, alpha=alpha)
        masked_video_frames.append(overlay.permute(1, 2, 0).cpu().numpy())

    return masked_video_frames
