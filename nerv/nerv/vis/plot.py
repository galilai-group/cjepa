import numpy as np
import matplotlib.pyplot as plt

from nerv.utils.misc import to_numpy


def get_lim(x):
    """Calculate the axis limit from data."""
    assert len(x.shape) == 1
    x_min, x_max = np.min(x), np.max(x)
    x_min = 1.2 * x_min if x_min < 0. else 0.8 * x_min
    x_max = 1.2 * x_max if x_max > 0. else 0.8 * x_max
    return (x_min, x_max)


def set_axis(axis,
             grid=True,
             lim=None,
             label=('x', 'y', 'z'),
             equal_axis=False):
    """Set up the axis of plt figure."""
    is_3d = (len(lim) == 3)
    axis.grid(grid)
    axis.set_xlabel(label[0])
    axis.set_ylabel(label[1])
    if is_3d:
        axis.set_zlabel(label[2])
    # make axis have the same unit length
    if equal_axis:
        if is_3d:
            set_3daxes_equal(axis)
        else:
            axis.axis('equal')
    # set lim manually
    else:
        axis.set_xlim(*lim[0])
        axis.set_ylim(*lim[1])
        if is_3d:
            axis.set_zlim(*lim[2])


def set_3daxes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    From: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_2d(
        x,
        y,
        line_label=None,
        show=True,
        color=None,
        linewidth=1.5,
        linestyle='-',
        figsize=(5, 5),
        axis=None,
        title=None,
        lim=None,
        label=('x', 'y'),
        equal_axis=False,
):
    """Plot a 2D curve."""
    x, y = to_numpy(x), to_numpy(y)

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    plot_configs = dict(
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=line_label,
    )
    ax.plot(x, y, **plot_configs)

    if lim is None:
        lim = (get_lim(x), get_lim(y))

    set_axis(ax, lim=lim, label=label, equal_axis=equal_axis)
    plt.tight_layout()

    if show:
        if line_label is not None:
            plt.legend()
        plt.show()

    return fig


def scatter_2d(
        x,
        y,
        scatter_label=None,
        show=True,
        color=None,
        marker='.',
        marker_size=8,
        figsize=(5, 5),
        axis=None,
        title=None,
        lim=None,
        label=('x', 'y'),
        equal_axis=False,
):
    """Draw a 2D scatter."""
    x, y = to_numpy(x), to_numpy(y)

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    scatter_configs = dict(
        c=color,
        marker=marker,
        s=marker_size,
        label=scatter_label,
    )
    ax.scatter(x, y, **scatter_configs)

    if lim is None:
        lim = (get_lim(x), get_lim(y))

    set_axis(ax, lim=lim, label=label, equal_axis=equal_axis)
    plt.tight_layout()

    if show:
        if scatter_label is not None:
            plt.legend()
        plt.show()

    return fig


def plot_3d(
        x,
        y,
        z,
        line_label=None,
        show=True,
        color=None,
        linewidth=1.5,
        linestyle='-',
        figsize=(5, 5),
        axis=None,
        title=None,
        lim=None,
        label=('x', 'y', 'z'),
        equal_axis=False,
):
    """Plot a 3D curve."""
    x, y, z = to_numpy(x), to_numpy(y), to_numpy(z)

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    plot_configs = dict(
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=line_label,
    )
    ax.plot(x, y, z, **plot_configs)

    if lim is None:
        lim = (get_lim(x), get_lim(y), get_lim(z))

    set_axis(ax, lim=lim, label=label, equal_axis=equal_axis)
    plt.tight_layout()

    if show:
        if line_label is not None:
            plt.legend()
        plt.show()

    return fig
