import itertools
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import nn

from videosaur.videosaur import schedulers


class OptimizerBuilder:
    def __init__(
        self,
        name: str,
        lr: float,
        lr_scheduler: Optional[Dict[str, Any]] = None,
        param_groups: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        if name.lower() not in ("adam"):
            raise ValueError(f"Unknown optimizer {name}")

        self.name = name.lower()
        self.lr = lr
        self.param_groups = param_groups
        self.optim_kwargs = kwargs

        groups_have_lr_sched = self.param_groups is not None and any(
            key == "lr_scheduler" for key in self.param_groups
        )
        if groups_have_lr_sched and lr_scheduler is not None:
            raise ValueError(
                "Can either define global `lr_scheduler` or schedulers in "
                "`param_groups`, but not both"
            )

        if groups_have_lr_sched:
            self.schedule_fn = []
            for idx, param_group in enumerate(self.param_groups):
                if "lr_scheduler" not in param_group:
                    raise ValueError(f"Missing `lr_scheduler` in param_group {idx}")
                self.schedule_fn.append(schedulers.build(param_group["lr_scheduler"]))
                del param_group["lr_scheduler"]
        elif lr_scheduler is not None:
            self.schedule_fn = schedulers.build(lr_scheduler)
        else:
            self.schedule_fn = None

    def __call__(self, modules: Dict[str, nn.Module]):
        if self.param_groups is None:
            parameters = itertools.chain.from_iterable(m.parameters() for m in modules.values())
        else:
            parameters = []
            for idx, param_group in enumerate(self.param_groups):
                assert isinstance(param_group, Mapping)
                group_modules = param_group["modules"]
                if isinstance(group_modules, str):
                    group_modules = [group_modules]
                for name in group_modules:
                    if name not in modules:
                        raise ValueError(
                            f"Requested module {name} in param group {idx}, but "
                            f"this module is not available. Available modules: {list(modules)}."
                        )

                params = itertools.chain.from_iterable(
                    modules[name].parameters() for name in group_modules
                )
                parameters.append(
                    {"params": params, **{k: v for k, v in param_group.items() if k != "modules"}}
                )

        if self.name == "adam":
            optimizer = torch.optim.Adam(parameters, lr=self.lr, **self.optim_kwargs)
        elif self.name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=self.lr, **self.optim_kwargs)
        else:
            raise ValueError(f"Optimizer {self.name} is not known")

        if self.schedule_fn is not None:
            scheduler = schedulers.apply_schedule_fn_to_optimizer(optimizer, self.schedule_fn)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        else:
            return optimizer
