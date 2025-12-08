import numpy as np
import pytest
import torch
from torch.nn import functional as F

from videosaur import metrics


@pytest.mark.parametrize("video_input", [False, True])
def test_image_ari(video_input):
    metric = metrics.ImageARI(video_input=video_input)

    a = _one_hot([[[0, 0], [1, 1]]], axis=1)
    b = _one_hot([[[0, 1], [2, 3]]], axis=1)
    if video_input:
        a = torch.stack((a, a), axis=1)
        b = torch.stack((b, b), axis=1)

    metric.update(a, a)
    assert np.allclose(metric.compute().numpy(), 1.0)

    metric.update(a, b)
    assert np.allclose(metric.compute().numpy(), 0.5)

    # Mask not binary should raise error
    a[:, :, :, 0] = 2
    with pytest.raises(ValueError):
        metric.update(a, b)
    a[:, :, :, 0] = 1
    b[:, :, :, 1] = 2
    with pytest.raises(ValueError):
        metric.update(a, b)


@pytest.mark.parametrize("ignore_overlaps", [False, True])
@pytest.mark.parametrize("video_input", [False, True])
def test_image_ari_ignore_overlaps(ignore_overlaps, video_input):
    metric = metrics.ImageARI(ignore_overlaps=ignore_overlaps, video_input=video_input)

    a = _one_hot([[[0, 0], [1, 1], [2, 2]]], axis=1)
    a_pred = a.clone()
    a[0, :, 2, :] = 1  # Set some overlaps, ignoring last row
    b = _one_hot([[[0, 0], [1, 1], [2, 3]]], axis=1)
    c = _one_hot([[[0, 1], [2, 3], [4, 4]]], axis=1)
    if video_input:
        a = torch.stack((a, a), axis=1)
        a_pred = torch.stack((a_pred, a_pred), axis=1)
        b = torch.stack((b, b), axis=1)
        c = torch.stack((c, c), axis=1)

    if ignore_overlaps:
        metric.update(a, a_pred)
        assert np.allclose(metric.compute().numpy(), 1.0)
        metric.update(a, b)
        assert np.allclose(metric.compute().numpy(), 1.0)
        metric.reset()
        metric.update(a, c)
        assert np.allclose(metric.compute().numpy(), 0.0)

        # Fully empty ground truth masks should result in no update
        metric.reset()
        metric.update(torch.zeros_like(a), b)
        assert np.isnan(metric.compute().numpy())
    else:
        with pytest.raises(ValueError):
            metric.update(a, a_pred)  # Not one-hot should raise error


def test_video_ari():
    metric = metrics.VideoARI()

    a = _one_hot([[[0, 0], [1, 1]]], axis=1, n_classes=8)
    b = _one_hot([[[0, 1], [2, 3]]], axis=1, n_classes=8)
    c = _one_hot([[[4, 5], [6, 7]]], axis=1, n_classes=8)
    a = torch.stack((a, a), axis=1)
    b = torch.stack((b, c), axis=1)

    metric.update(a, a)
    assert np.allclose(metric.compute().numpy(), 1.0)

    metric.update(a, b)
    assert np.allclose(metric.compute().numpy(), 0.5)

    # Mask not binary should raise error
    a[:, :, :, 0] = 2
    with pytest.raises(ValueError):
        metric.update(a, b)
    a[:, :, :, 0] = 1
    b[:, :, :, 1] = 2
    with pytest.raises(ValueError):
        metric.update(a, b)


@pytest.mark.parametrize("ignore_overlaps", [False, True])
def test_video_ari_ignore_overlaps(ignore_overlaps):
    metric = metrics.VideoARI(ignore_overlaps=ignore_overlaps)

    a = _one_hot([[[0, 0], [1, 1], [2, 2]]], axis=1)
    a_pred = a.clone()
    a[0, :, 2, :] = 1  # Set some overlaps, ignoring last row
    b = _one_hot([[[0, 0], [1, 1], [2, 3]]], axis=1)
    c1 = _one_hot([[[0, 1], [2, 3], [4, 4]]], axis=1, n_classes=8)
    c2 = _one_hot([[[4, 5], [6, 7], [4, 4]]], axis=1, n_classes=8)

    a = torch.stack((a, a), axis=1)
    a_pred = torch.stack((a_pred, a_pred), axis=1)
    b = torch.stack((b, b), axis=1)
    c = torch.stack((c1, c2), axis=1)

    if ignore_overlaps:
        metric.update(a, a_pred)
        assert np.allclose(metric.compute().numpy(), 1.0)
        metric.update(a, b)
        assert np.allclose(metric.compute().numpy(), 1.0)
        metric.reset()
        metric.update(a, c)
        assert np.allclose(metric.compute().numpy(), 0.0)

        # Fully empty ground truth masks should result in no update
        metric.reset()
        metric.update(torch.zeros_like(a), b)
        assert np.isnan(metric.compute().numpy())
    else:
        with pytest.raises(ValueError):
            metric.update(a, a_pred)  # Not one-hot should raise error


def test_adjusted_rand_index():
    # All zeros ground truth mask
    assert np.allclose(
        metrics.adjusted_rand_index(torch.zeros(1, 1, 4), _one_hot([[0, 0, 1, 1]])).numpy(), 1.0
    )

    # Examples from scikit-learn
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    assert np.allclose(
        metrics.adjusted_rand_index(_one_hot([[0, 0, 1, 1]]), _one_hot([[0, 0, 1, 1]])).numpy(), 1.0
    )
    assert np.allclose(
        metrics.adjusted_rand_index(_one_hot([[0, 0, 1, 1]]), _one_hot([[1, 1, 0, 0]])).numpy(), 1.0
    )
    assert np.allclose(
        metrics.adjusted_rand_index(_one_hot([[0, 0, 1, 1]]), _one_hot([[0, 0, 1, 2]])).numpy(),
        0.571428571,
    )
    assert np.allclose(
        metrics.adjusted_rand_index(_one_hot([[0, 0, 1, 1]]), _one_hot([[0, 1, 2, 3]])).numpy(), 0.0
    )
    assert np.allclose(
        metrics.adjusted_rand_index(_one_hot([[0, 0, 1, 1]]), _one_hot([[0, 1, 0, 1]])).numpy(), -0.5
    )


@pytest.mark.parametrize("video_input", [False, True])
def test_image_iou(video_input):
    metric = metrics.ImageIoU(video_input=video_input, matching="overlap")

    a = _one_hot([[[0, 0], [1, 1]]], axis=1)
    b = _one_hot([[[0, 1], [2, 3]]], axis=1)
    if video_input:
        a = torch.stack((a, a), axis=1)
        b = torch.stack((b, b), axis=1)

    metric.update(a, a)
    assert np.allclose(metric.compute().numpy(), 1.0)

    metric.update(a, b)
    assert np.allclose(metric.compute().numpy(), 0.75)

    # Mask not binary should raise error
    a[:, :, :, 0] = 2
    with pytest.raises(ValueError):
        metric.update(a, b)
    a[:, :, :, 0] = 1
    b[:, :, :, 1] = 2
    with pytest.raises(ValueError):
        metric.update(a, b)


@pytest.mark.parametrize("ignore_overlaps", [False, True])
@pytest.mark.parametrize("video_input", [False, True])
def test_image_iou_ignore_overlaps(ignore_overlaps, video_input):
    metric = metrics.ImageIoU(
        ignore_overlaps=ignore_overlaps, video_input=video_input, matching="overlap"
    )

    a = _one_hot([[[0, 0], [1, 1], [2, 2]]], axis=1)
    a_pred = a.clone()
    a[0, :, 2, :] = 1  # Set some overlaps, ignoring last row
    b = _one_hot([[[0, 0], [1, 1], [2, 3]]], axis=1)
    c = _one_hot([[[0, 1], [2, 3], [4, 4]]], axis=1)
    if video_input:
        a = torch.stack((a, a), axis=1)
        a_pred = torch.stack((a_pred, a_pred), axis=1)
        b = torch.stack((b, b), axis=1)
        c = torch.stack((c, c), axis=1)

    if ignore_overlaps:
        metric.update(a, a_pred)
        assert np.allclose(metric.compute().numpy(), 1.0)
        metric.update(a, b)
        assert np.allclose(metric.compute().numpy(), 1.0)
        metric.reset()
        metric.update(a, c)
        assert np.allclose(metric.compute().numpy(), 0.5)

        # Fully empty ground truth masks should result in no update
        metric.reset()
        metric.update(torch.zeros_like(a), b)
        assert np.isnan(metric.compute().numpy())
    else:
        with pytest.raises(ValueError):
            metric.update(a, a_pred)  # Not one-hot should raise error


def test_video_iou():
    metric = metrics.VideoIoU(matching="overlap")

    a = _one_hot([[[0, 0], [3, 3]]], axis=1, n_classes=8)
    b = _one_hot([[[0, 0], [2, 2]]], axis=1, n_classes=8)
    c = _one_hot([[[4, 5], [6, 7]]], axis=1, n_classes=8)
    a = torch.stack((a, a), axis=1)
    b = torch.stack((b, c), axis=1)

    metric.update(a, a)
    assert np.allclose(metric.compute().numpy(), 1.0)

    metric.update(a, b)
    assert np.allclose(metric.compute().numpy(), 0.75)

    # Mask not binary should raise error
    a[:, :, :, 0] = 2
    with pytest.raises(ValueError):
        metric.update(a, b)
    a[:, :, :, 0] = 1
    b[:, :, :, 1] = 2
    with pytest.raises(ValueError):
        metric.update(a, b)


@pytest.mark.parametrize("ignore_overlaps", [False, True])
def test_video_iou_ignore_overlaps(ignore_overlaps):
    metric = metrics.VideoIoU(ignore_overlaps=ignore_overlaps, matching="overlap")

    a = _one_hot([[[0, 0], [1, 1], [2, 2]]], axis=1)
    a_pred = a.clone()
    a[0, :, 2, :] = 1  # Set some overlaps, ignoring last row
    b = _one_hot([[[0, 0], [1, 1], [2, 3]]], axis=1)
    c1 = _one_hot([[[0, 0], [2, 2], [4, 4]]], axis=1, n_classes=8)
    c2 = _one_hot([[[4, 5], [6, 7], [4, 4]]], axis=1, n_classes=8)

    a = torch.stack((a, a), axis=1)
    a_pred = torch.stack((a_pred, a_pred), axis=1)
    b = torch.stack((b, b), axis=1)
    c = torch.stack((c1, c2), axis=1)

    if ignore_overlaps:
        metric.update(a, a_pred)
        assert np.allclose(metric.compute().numpy(), 1.0)
        metric.update(a, b)
        assert np.allclose(metric.compute().numpy(), 1.0)
        metric.reset()
        metric.update(a, c)
        assert np.allclose(metric.compute().numpy(), 0.5)

        # Fully empty ground truth masks should result in no update
        metric.reset()
        metric.update(torch.zeros_like(a), b)
        assert np.isnan(metric.compute().numpy())
    else:
        with pytest.raises(ValueError):
            metric.update(a, a_pred)  # Not one-hot should raise error


def test_intersection_over_union_with_matching():
    def call_metric(true, pred, matching, **kwargs):
        if not isinstance(true, torch.Tensor):
            true = _one_hot(true)
        if not isinstance(pred, torch.Tensor):
            pred = _one_hot(pred)
        values = metrics.intersection_over_union_with_matching(
            true, pred, matching=matching, **kwargs
        )
        return values.numpy()

    # All zeros ground truth and pred mask
    values = call_metric(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), "none", empty_value=1.0)
    assert np.allclose(values, np.array([1.0, 1.0]))

    # All zeros ground truth mask
    values = call_metric(torch.zeros(1, 1, 2), [[0, 0, 1, 1]], "none")
    assert np.allclose(values, np.array([0.0, 0.0]))
    values = call_metric(torch.zeros(1, 1, 4), [[0, 0, 1, 1]], "overlap")
    assert np.allclose(values, np.array([0.0, 0.0, 0.0, 0.0]))
    values = call_metric(torch.zeros(1, 1, 4), [[0, 0, 1, 1]], "hungarian")
    assert np.allclose(values, np.array([0.0, 0.0, 0.0, 0.0]))

    assert np.allclose(call_metric([[0, 0, 1, 1]], [[0, 0, 1, 1]], "none")[0], np.array([1.0, 1.0]))
    assert np.allclose(
        call_metric([[0, 0, 1, 1]], [[0, 0, 1, 1]], "overlap")[0], np.array([1.0, 1.0])
    )
    assert np.allclose(
        call_metric([[0, 0, 1, 1]], [[0, 0, 1, 1]], "hungarian")[0], np.array([1.0, 1.0])
    )

    assert np.allclose(call_metric([[1, 1, 0, 0]], [[0, 0, 1, 1]], "none")[0], np.array([0.0, 0.0]))
    assert np.allclose(
        call_metric([[1, 1, 0, 0]], [[0, 0, 1, 1]], "overlap")[0], np.array([1.0, 1.0])
    )
    assert np.allclose(
        call_metric([[1, 1, 0, 0]], [[0, 0, 1, 1]], "hungarian")[0], np.array([1.0, 1.0])
    )

    assert np.allclose(
        call_metric([[0, 0, 1, 1]], [[0, 0, 1, 2]], "overlap")[0], np.array([1.0, 0.5])
    )
    assert np.allclose(
        call_metric([[0, 0, 1, 1]], [[0, 0, 1, 2]], "hungarian")[0], np.array([1.0, 0.5])
    )
    assert np.allclose(
        call_metric([[0, 0, 1, 2]], [[0, 0, 1, 1]], "overlap")[0], np.array([1.0, 0.5, 0.5])
    )
    assert np.allclose(
        call_metric([[0, 0, 1, 2]], [[0, 0, 1, 1]], "hungarian")[0], np.array([1.0, 0.5, 0.0])
    )
    assert np.allclose(
        call_metric([[0, 0, 1, 1, 1]], [[1, 1, 1, 3, 3]], "overlap")[0], np.array([2 / 3, 2 / 3])
    )
    assert np.allclose(
        call_metric([[0, 0, 1, 1, 1, 1]], [[1, 1, 1, 1, 1, 3]], "overlap")[0], np.array([0.4, 0.5])
    )
    assert np.allclose(
        call_metric([[0, 0, 1, 1, 1, 1]], [[1, 1, 1, 1, 1, 3]], "hungarian")[0],
        np.array([0.4, 0.25]),
    )


def test_j_and_f_metric():
    def unpack_result(result):
        return {k: v.numpy() for k, v in result.items()}

    metric = metrics.ImageJandF()

    a = _one_hot([[[0, 0], [1, 1]]], axis=1, n_classes=3)
    b = _one_hot([[[0, 1], [2, 3]]], axis=1)

    metric.update(a, a)
    result = unpack_result(metric.compute())
    assert result == {"j_and_f": 1.0, "jaccard": 1.0, "boundary_f_measure": 1.0}

    metric.update(a, b)
    result = unpack_result(metric.compute())
    assert np.allclose(result["jaccard"], (1.0 + 0.5) / 2)
    assert np.allclose(result["boundary_f_measure"], (1.0 + 2 / 3) / 2)
    assert np.allclose(result["j_and_f"], (1.0 + (0.5 + 2 / 3) / 2) / 2)


def test_masks_to_boundaries():
    masks = torch.zeros(2, 32, 32, dtype=torch.bool)
    masks[0, 0:9, 0:9] = True
    masks[1, 12:20, 12:20] = True

    expected_boundaries = torch.zeros(2, 32, 32, dtype=torch.bool)
    expected_boundaries[0, 0, :9] = True
    expected_boundaries[0, :9, 0] = True
    expected_boundaries[0, 8, :9] = True
    expected_boundaries[0, :9, 8] = True
    expected_boundaries[1, 12, 12:20] = True
    expected_boundaries[1, 12:20, 12] = True
    expected_boundaries[1, 19, 12:20] = True
    expected_boundaries[1, 12:20, 19] = True

    boundaries = metrics.masks_to_boundaries(masks)
    assert np.allclose(boundaries, expected_boundaries)


def _one_hot(ids, axis=-1, n_classes=None):
    n_classes = n_classes if n_classes else np.max(ids) + 1
    one_hot = F.one_hot(torch.tensor(ids), n_classes)
    axis = axis if axis >= 0 else one_hot.ndim + 1 - axis
    dims = list(range(one_hot.ndim - 1))
    dims.insert(axis, one_hot.ndim - 1)
    one_hot = one_hot.permute(*dims)
    return one_hot
