"""Plan in PushT with a trained C-JEPA checkpoint."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2

from checkpoint import ensure_planning_checkpoint


def image_transform(size):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(**spt.data.dataset_stats.ImageNet),
            v2.Resize((size, size)),
        ]
    )


def choose_starts(dataset, num_eval, goal_offset, seed):
    episode_column = (
        "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    )
    episode = dataset.get_col_data(episode_column)
    step = dataset.get_col_data("step_idx")
    _, episode_inverse = np.unique(episode, return_inverse=True)
    episode_end = np.full(episode_inverse.max() + 1, np.iinfo(step.dtype).min)
    np.maximum.at(episode_end, episode_inverse, step)
    valid_rows = np.flatnonzero(step <= episode_end[episode_inverse] - goal_offset)

    # Preserve the legacy evaluator exactly, including its exclusive upper
    # bound of len(dataset_start) - 1 (which leaves the last valid row out).
    sampling_population = len(valid_rows) - 1
    if sampling_population < num_eval:
        raise ValueError(
            f"Only {sampling_population} legacy-sampleable starts are available; "
            f"need {num_eval}"
        )
    selected_rows = valid_rows[
        np.random.default_rng(seed).choice(
            sampling_population, size=num_eval, replace=False
        )
    ]
    return episode[selected_rows].astype(int).tolist(), step[selected_rows].astype(
        int
    ).tolist()


def make_processors(dataset, columns):
    processors = {}
    for column in columns:
        values = dataset.get_col_data(column)
        values = values[np.isfinite(values).all(axis=-1)]
        processors[column] = StandardScaler().fit(values)
        if column != "action":
            processors[f"goal_{column}"] = processors[column]
    return processors


@hydra.main(version_base=None, config_path="config", config_name="eval")
def run(cfg: DictConfig):
    checkpoint_root = swm.data.utils.get_cache_dir(
        cfg.dataset.cache_dir, sub_folder="checkpoints"
    )
    if cfg.download:
        ensure_planning_checkpoint(
            cfg.policy,
            checkpoint_root,
            **OmegaConf.to_container(cfg.checkpoint, resolve=True),
        )

    dataset = swm.data.load_dataset(
        cfg.dataset.name,
        cache_dir=cfg.dataset.cache_dir,
        keys_to_cache=list(cfg.dataset.normalize),
    )
    episodes, starts = choose_starts(
        dataset,
        cfg.eval.num_eval,
        cfg.eval.goal_offset,
        cfg.seed,
    )
    processors = make_processors(dataset, cfg.dataset.normalize)

    model = swm.policy.AutoCostModel(
        cfg.policy,
        cache_dir=checkpoint_root,
    )
    device = torch.device(cfg.device)
    model = model.to(device).eval().requires_grad_(False)
    solver = hydra.utils.instantiate(cfg.solver, model=model, device=device)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(**cfg.plan),
        process=processors,
        transform={
            "pixels": image_transform(cfg.eval.image_size),
            "goal": image_transform(cfg.eval.image_size),
        },
    )

    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["num_envs"] = cfg.eval.num_eval
    world_cfg["max_episode_steps"] = cfg.eval.eval_budget * 2
    world = swm.World(**world_cfg)
    world.set_policy(policy)

    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=episodes,
        start_steps=starts,
        goal_offset=cfg.eval.goal_offset,
        eval_budget=cfg.eval.eval_budget,
        callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
        video=output if cfg.eval.video else None,
    )
    metrics["evaluation_seconds"] = time.time() - started
    print(metrics)
    with (output / "results.txt").open("a") as file:
        file.write(OmegaConf.to_yaml(cfg, resolve=True))
        file.write(f"metrics: {metrics}\n\n")
    world.close()


def main():
    """Accept the convenient bare --download flag before Hydra parses argv."""
    if "--download" in sys.argv:
        index = sys.argv.index("--download")
        if not any(argument.startswith("download=") for argument in sys.argv[1:]):
            sys.argv[index] = "download=true"
        else:
            sys.argv.pop(index)
    run()


if __name__ == "__main__":
    main()
