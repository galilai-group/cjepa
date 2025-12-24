from os import path

import cv2
import six
import numpy as np

from nerv.utils.misc import convert4save
from nerv.utils.io import check_file_exist, mkdir_or_exist


def read_img(img_or_path, flag=cv2.IMREAD_COLOR):
    """Read an image.

    Args:
        img_or_path (np.ndarray or str): either an image or path of an image.
        flag (int, optional): flags specifying the color type of loaded image.
            Default: cv2.IMREAD_COLOR.

    Returns:
        np.ndarray: loaded image array.
    """
    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif isinstance(img_or_path, six.string_types):
        check_file_exist(img_or_path,
                         'img file does not exist: {}'.format(img_or_path))
        return cv2.imread(img_or_path, flag)
    else:
        raise TypeError('"img" must be a numpy array or a filename')


def img_from_bytes(content, flag=cv2.IMREAD_COLOR):
    """Read an image from bytes.

    Args:
        content (bytes): images bytes got from files or other streams.
        flag (int, optional): same as :func:`read_img`.
            Default: cv2.IMREAD_COLOR.

    Returns:
        np.ndarray: image array.
    """
    img_np = np.fromstring(content, np.uint8)
    img = cv2.imdecode(img_np, flag)
    return img


def write_img(img, file_path, rgb2bgr=True):
    """Write image to file.

    Args:
        img (np.ndarray): image to be written to file.
        file_path (str): file path.
        rgb2bgr (bool, optional): whether convert the color channel.
            Default: True.

    Returns:
        bool: successful or not.
    """
    mkdir_or_exist(path.dirname(file_path))
    img = convert4save(img)
    if rgb2bgr:
        img = img[..., [2, 1, 0]]
    return cv2.imwrite(file_path, img)


def bgr2gray(img, keepdim=False):
    """Convert a BGR image to grayscale image

    Args:
        img (np.ndarray or str): either an image or path of an image.
        keepdim (bool): if set to False, return the gray image with 2 dims,
            otherwise 3 dims. Default: False

    Returns:
        np.ndarray: the converted grayscale image.
    """
    in_img = read_img(img)
    out_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    if keepdim:
        out_img = out_img[..., np.newaxis]
    return out_img


def gray2bgr(img):
    """Convert a grayscale image to BGR image.

    Args:
        img (np.ndarray or str): either an image or path of an image.

    Returns:
        np.ndarray: the converted BGR image.
    """
    in_img = read_img(img)
    if in_img.ndim == 2:
        out_img = cv2.cvtColor(in_img[..., np.newaxis], cv2.COLOR_GRAY2BGR)
    else:
        out_img = cv2.cvtColor(in_img, cv2.COLOR_GRAY2BGR)
    return out_img


def bgr2rgb(img):
    """Convert a BGR image to RGB image.

    Args:
        img (np.ndarray or str): either an image or path of an image.

    Returns:
        np.ndarray: the converted RGB image.
    """
    in_img = read_img(img)
    out_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
    return out_img


def rgb2bgr(img):
    """Convert a RGB image to BGR image.

    Args:
        img (np.ndarray or str): either an image or path of an image.

    Returns:
        np.ndarray: the converted BGR image.
    """
    in_img = read_img(img)
    out_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2BGR)
    return out_img


def bgr2hsv(img):
    """Convert a BGR image to HSV image.

    Args:
        img (np.ndarray or str): either an image or path of an image.

    Returns:
        np.ndarray: the converted HSV image.
    """
    in_img = read_img(img)
    out_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)
    return out_img


def hsv2bgr(img):
    """Convert a HSV image to BGR image.

    Args:
        img (np.ndarray or str): either an image or path of an image.

    Returns:
        np.ndarray: the converted BGR image.
    """
    in_img = read_img(img)
    out_img = cv2.cvtColor(in_img, cv2.COLOR_HSV2BGR)
    return out_img


def scale_size(size, scale):
    """Scale a size.

    Args:
        size (Tuple[int]): (w, h)
        scale (float): scaling factor.

    Returns:
        Tuple[int]: scaled size.

    """
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)


def resize(img, size, return_scale=False, interpolation=cv2.INTER_LINEAR):
    """Resize image by expected size

    Args:
        img (np.ndarray): image or image path.
        size (Tuple[int]): (w, h).
        return_scale (bool, optional): whether to return w_scale and h_scale.
            Default: False.
        interpolation (enum, optional): interpolation method.
            Default: cv2.INTER_LINEAR.

    Returns:
        np.ndarray: resized image.
    """
    img = read_img(img)
    h, w = img.shape[:2]
    resized_img = cv2.resize(img, size, interpolation=interpolation)
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / float(w)
        h_scale = size[1] / float(h)
        return resized_img, w_scale, h_scale


def resize_like(
    img,
    dst_img,
    return_scale=False,
    interpolation=cv2.INTER_LINEAR,
):
    """Resize image to the same size of a given image

    Args:
        img (np.ndarray): image or image path.
        dst_img (np.ndarray): the given image with expected size.
        return_scale (bool, optional): whether to return w_scale and h_scale.
            Default: False.
        interpolation (enum, optional): interpolation method.
            Default: cv2.INTER_LINEAR.

    Returns:
        np.ndarray: resized image.
    """
    h, w = dst_img.shape[:2]
    return resize(img, (w, h), return_scale, interpolation)


def resize_by_ratio(img, ratio, interpolation=cv2.INTER_LINEAR):
    """Resize image by a ratio.

    Args:
        img (np.ndarray): image or image path.
        ratio (float): scaling factor.
        interpolation (enum, optional): interpolation method.
            Default: cv2.INTER_LINEAR.

    Returns:
        np.ndarray: resized image.
    """
    assert isinstance(ratio, (float, int)) and ratio > 0
    img = read_img(img)
    h, w = img.shape[:2]
    new_size = scale_size((w, h), ratio)
    return cv2.resize(img, new_size, interpolation=interpolation)


def resize_keep_ar(
    img,
    max_long_edge,
    max_short_edge,
    return_scale=False,
    interpolation=cv2.INTER_LINEAR,
):
    """Resize image with aspect ratio unchanged

    The long edge of resized image is no greater than max_long_edge, the short
    edge of resized image is no greater than max_short_edge.

    Args:
        img (np.ndarray): image or image path.
        max_long_edge (int): max value of the long edge of resized image.
        max_short_edge (int): max value of the short edge of resized image.
        return_scale (bool): whether to return scale besides the resized image.
            Default: False.
        interpolation (enum, optional): interpolation method.
            Default: cv2.INTER_LINEAR.

    Returns:
        Tuple: (resized image, scale factor)
    """
    if max_long_edge < max_short_edge:
        raise ValueError(
            '"max_long_edge" should not be less than "max_short_edge"')
    img = read_img(img)
    h, w = img.shape[:2]
    scale = min(
        float(max_long_edge) / max(h, w),
        float(max_short_edge) / min(h, w))
    new_size = scale_size((w, h), scale)
    resized_img = cv2.resize(img, new_size, interpolation=interpolation)
    if return_scale:
        return resized_img, scale
    else:
        return resized_img
