import functools
import math
from typing import Callable, List, Union

import torch
from torch.optim.lr_scheduler import LambdaLR

from videosaur.videosaur import utils


@utils.make_build_fn(__name__, "scheduler")
def build(
    config,
    name: str,
):
    if name == "constant":
        scheduler_fn = constant
    elif name == "linear_warmup":
        scheduler_fn = functools.partial(
            linear_warmup, warmup_steps=config.get("warmup_steps", 1000)
        )
    if name == "exp_decay_with_warmup":
        scheduler_fn = functools.partial(
            exp_decay_with_warmup,
            warmup_steps=config.get("warmup_steps", 1000),
            decay_steps=config.get("decay_steps", 100000),
            decay_rate=config.get("decay_rate", 0.5),
        )
    elif name == "cosine_decay_with_warmup":
        scheduler_fn = functools.partial(
            cosine_decay_with_warmup,
            warmup_steps=config.get("warmup_steps", 1000),
            decay_steps=config.get("decay_steps", 100000),
        )
    else:
        raise ValueError(f"Unknown scheduler {name}")

    return scheduler_fn


def apply_schedule_fn_to_optimizer(
    optimizer: torch.optim.Optimizer,
    decay_fn: Union[Callable[[int], float], List[Callable[[int], float]]],
) -> LambdaLR:
    return LambdaLR(optimizer, decay_fn)


def constant(step: int) -> float:
    """Constant schedule.

    Function maps current step or epoch to factor of learning rate schedules.
    """
    return 1.0


def linear_warmup(step: int, warmup_steps: int) -> float:
    """Linear warmup.

    Function maps current step or epoch to factor of learning rate schedules.
    """
    if warmup_steps > 0:
        return min(1.0, step / warmup_steps)
    else:
        return 1.0


def exp_decay_with_warmup(
    step: int,
    warmup_steps: int,
    decay_steps: int,
    decay_rate: float,
) -> float:
    """Exponential decay with linear learning rate warmup.

    Function maps current step or epoch to factor of learning rate schedules. After `decay_steps`,
    factor equals `decay_rate`.
    """
    if step < warmup_steps:
        return linear_warmup(step, warmup_steps)
    else:
        step = step - warmup_steps
        decay_steps = decay_steps - warmup_steps
        return decay_rate ** (step / decay_steps)


def cosine_decay_with_warmup(
    step: int,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Cosine decay to zero with linear learning rate warmup.

    Function maps current step or epoch to factor of learning rate schedules.
    """
    if step < warmup_steps:
        return linear_warmup(step, warmup_steps)
    else:
        step = step - warmup_steps
        decay_steps = decay_steps - warmup_steps
        step = min(step, decay_steps)
        return 0.5 * (1 + math.cos(math.pi * (step / decay_steps)))
