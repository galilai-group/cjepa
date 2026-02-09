import numpy as np
import pytest

from src.third_party.videosaur.videosaur import schedulers


@pytest.mark.parametrize(
    "step, warmup_steps, expected",
    [
        (0, 10, 0),
        (5, 10, 0.5),
        (10, 10, 1.0),
        (11, 10, 1.0),
    ],
)
def test_linear_warmup(step, warmup_steps, expected):
    factor = schedulers.linear_warmup(step, warmup_steps)
    assert np.allclose(factor, expected)


@pytest.mark.parametrize(
    "step, warmup_steps, decay_steps, expected",
    [
        (0, 10, 30, 0),
        (5, 10, 30, 0.5),
        (10, 10, 30, 1.0),
        (20, 10, 30, 0.5**0.5),
        (30, 10, 30, 0.5),
        (40, 10, 30, 0.5**1.5),
    ],
)
def test_exp_decay_with_warmup(step, warmup_steps, decay_steps, expected):
    factor = schedulers.exp_decay_with_warmup(step, warmup_steps, decay_steps, decay_rate=0.5)
    assert np.allclose(factor, expected)


@pytest.mark.parametrize(
    "step, warmup_steps, decay_steps, expected",
    [
        (0, 10, 30, 0),
        (5, 10, 30, 0.5),
        (10, 10, 30, 1.0),
        (20, 10, 30, 0.5),
        (30, 10, 30, 0.0),
        (31, 10, 30, 0.0),
    ],
)
def test_cosine_decay_with_warmup(step, warmup_steps, decay_steps, expected):
    factor = schedulers.cosine_decay_with_warmup(step, warmup_steps, decay_steps)
    assert np.allclose(factor, expected)
