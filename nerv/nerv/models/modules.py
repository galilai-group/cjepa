import torch.nn as nn


def get_conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    dim='2d',
):
    """Get Conv layer."""
    return eval(f'nn.Conv{dim}')(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        dilation=dilation,
        groups=groups,
        bias=bias)


def get_deconv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    dim='2d',
):
    """Get Conv layer."""
    return eval(f'nn.ConvTranspose{dim}')(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        output_padding=stride - 1,
        dilation=dilation,
        groups=groups,
        bias=bias)


def get_normalizer(norm, channels, groups=16, dim='2d'):
    """Get normalization layer."""
    if norm == '':
        return nn.Identity()
    elif norm == 'bn':
        return eval(f'nn.BatchNorm{dim}')(channels)
    elif norm == 'gn':
        # 16 is taken from Table 3 of the GN paper
        return nn.GroupNorm(groups, channels)
    elif norm == 'in':
        return eval(f'nn.InstanceNorm{dim}')(channels)
    elif norm == 'ln':
        return nn.LayerNorm(channels)
    else:
        raise ValueError(f'Normalizer {norm} not supported!')


def get_act_func(act):
    """Get activation function."""
    if act == '':
        return nn.Identity()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'leakyrelu':
        return nn.LeakyReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'swish':
        return nn.SiLU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'softplus':
        return nn.Softplus()
    elif act == 'mish':
        return nn.Mish()
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Activation function {act} not supported!')


def conv_norm_act(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    norm='bn',
    act='relu',
    dim='2d',
):
    """Conv - Norm - Act."""
    conv = get_conv(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=norm not in ['bn', 'in'],
        dim=dim)
    normalizer = get_normalizer(norm, out_channels, dim=dim)
    act_func = get_act_func(act)
    return nn.Sequential(conv, normalizer, act_func)


def deconv_norm_act(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    norm='bn',
    act='relu',
    dim='2d',
):
    """ConvTranspose - Norm - Act."""
    deconv = get_deconv(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=norm not in ['bn', 'in'],
        dim=dim)
    normalizer = get_normalizer(norm, out_channels, dim=dim)
    act_func = get_act_func(act)
    return nn.Sequential(deconv, normalizer, act_func)


def fc_norm_act(in_features, out_features, norm='bn', act='relu'):
    """FC - Norm - Act."""
    fc = nn.Linear(in_features, out_features, bias=norm not in ['bn', 'in'])
    normalizer = get_normalizer(norm, out_features, dim='1d')
    act_func = get_act_func(act)
    return nn.Sequential(fc, normalizer, act_func)


def build_mlps(in_channels, hidden_sizes, out_channels, norm='bn', act='relu'):
    """Construct MLP with norm and act."""
    if not hidden_sizes:  # None or empty list
        return nn.Linear(in_channels, out_channels)
    modules = [fc_norm_act(in_channels, hidden_sizes[0], norm=norm, act=act)]
    for i in range(0, len(hidden_sizes) - 1):
        modules.append(
            fc_norm_act(
                hidden_sizes[i], hidden_sizes[i + 1], norm=norm, act=act))
    modules.append(nn.Linear(hidden_sizes[-1], out_channels))
    return nn.Sequential(*modules)
