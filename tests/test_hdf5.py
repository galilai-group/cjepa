import numpy as np
import stable_worldmodel as swm


def test_stable_worldmodel_loads_preextracted_slots(tmp_path):
    path = tmp_path / "slots.h5"
    with swm.data.HDF5Writer(path, mode="overwrite") as writer:
        writer.write_episode(
            {
                "slots": np.random.randn(6, 3, 8).astype(np.float32),
                "episode_idx": np.zeros(6, dtype=np.int64),
                "step_idx": np.arange(6, dtype=np.int64),
            }
        )

    dataset = swm.data.load_dataset(str(path), num_steps=3, frameskip=1)
    sample = dataset[0]
    assert sample["slots"].shape == (3, 3, 8)
    assert set(("slots", "episode_idx", "step_idx")).issubset(dataset.column_names)
