import numpy as np

from torch.nn import LayerNorm, GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from nerv.utils.misc import array_in_list


def conv_out_shape(in_size, stride, padding, kernel_size, dilation=1):
    """Calculate the output shape of a Conv layer."""
    return np.floor((in_size + 2 * padding - dilation *
                     (kernel_size - 1) - 1) / float(stride) + 1)


def deconv_out_shape(
    in_size,
    stride,
    padding,
    kernel_size,
    out_padding,
    dilation=1,
):
    """Calculate the output shape of a ConvTranspose layer."""
    return (in_size - 1) * stride - 2 * padding + dilation * (
        kernel_size - 1) + out_padding + 1


def filter_wd_parameters(model, skip_list=()):
    """Create parameter groups for optimizer.

    We do two things:
        - filter out params that do not require grad
        - exclude bias and Norm layers
    """
    # we need to sort the names so that we can save/load ckps
    w_name, b_name, no_decay_name = [], [], []
    for name, m in model.named_modules():
        # exclude norm weight
        if isinstance(m, (LayerNorm, GroupNorm, _BatchNorm, _InstanceNorm)):
            w_name.append(name)
        # exclude bias
        if hasattr(m, 'bias') and m.bias is not None:
            b_name.append(name)
        if name in skip_list:
            no_decay_name.append(name)
    w_name.sort()
    b_name.sort()
    no_decay_name.sort()
    no_decay = [model.get_submodule(m).weight for m in w_name] + \
        [model.get_submodule(m).bias for m in b_name]
    for name in no_decay_name:
        no_decay += [
            p for p in model.get_submodule(m).parameters()
            if p.requires_grad and not array_in_list(p, no_decay)
        ]

    decay_name = []
    for name, param in model.named_parameters():
        if param.requires_grad and not array_in_list(param, no_decay):
            decay_name.append(name)
    decay_name.sort()
    decay = [model.get_parameter(name) for name in decay_name]
    return {'decay': list(decay), 'no_decay': list(no_decay)}
