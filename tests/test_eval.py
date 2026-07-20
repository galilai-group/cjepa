import numpy as np
import torch
from gymnasium.spaces import Box
from stable_worldmodel import PlanConfig

from eval import LegacyWorldModelPolicy, choose_starts


class Dataset:
    column_names = ("episode_idx", "step_idx")

    def __init__(self, episode, step):
        self.columns = {
            "episode_idx": np.asarray(episode),
            "step_idx": np.asarray(step),
        }

    def get_col_data(self, name):
        return self.columns[name]


def test_choose_starts_matches_legacy_sampling_exactly():
    dataset = Dataset(
        episode=[0, 0, 0, 0, 1, 1, 1, 1],
        step=[0, 1, 2, 3, 0, 1, 2, 3],
    )
    valid_rows = np.array([0, 1, 2, 4, 5, 6])
    expected_rows = valid_rows[
        np.random.default_rng(42).choice(len(valid_rows) - 1, size=5, replace=False)
    ]

    episodes, steps = choose_starts(dataset, 5, goal_offset=1, seed=42)

    np.testing.assert_array_equal(
        episodes, dataset.columns["episode_idx"][expected_rows]
    )
    np.testing.assert_array_equal(steps, dataset.columns["step_idx"][expected_rows])
    assert (1, 2) not in zip(episodes, steps, strict=True)


class Solver:
    def __init__(self):
        self.calls = []

    def configure(self, **kwargs):
        self.configuration = kwargs

    def __call__(self, info, init_action=None):
        self.calls.append((info, init_action))
        return {"actions": torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)}


class Environment:
    num_envs = 2
    action_space = Box(-1, 1, shape=(2, 2), dtype=np.float32)


def test_legacy_policy_replans_all_environments_synchronously():
    solver = Solver()
    policy = LegacyWorldModelPolicy(
        solver=solver,
        config=PlanConfig(
            horizon=2,
            receding_horizon=1,
            action_block=2,
        ),
    )
    policy.set_env(Environment())
    info = {
        "id": np.array([[1], [2]]),
        "terminated": np.array([True, False]),
    }

    first = policy.get_action(info)
    second = policy.get_action(info)

    assert len(solver.calls) == 1
    assert len(solver.calls[0][0]["id"]) == 2
    np.testing.assert_array_equal(first, [[0, 1], [8, 9]])
    np.testing.assert_array_equal(second, [[2, 3], [10, 11]])
