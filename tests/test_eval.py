import numpy as np

from eval import choose_starts


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
