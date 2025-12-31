import numpy as np
from stable_worldmodel.data.dataset import VideoDataset, Dataset
import torch

class ClevrerVideoDataset(VideoDataset):
    """
    Custom VideoDataset for CLEVRER with episode index offset support.

    This class extends VideoDataset to add an idx_offset parameters  that shifts
    all episode indices by a constant value. 
    """

    def __init__(self, name, *args, idx_offset=0, **kwargs):
        # Call parent VideoDataset.__init__
        super().__init__(name, *args, **kwargs)

        # Store the offset
        self.idx_offset =idx_offset


    def __repr__(self):
        return (
            f"ClevrerVideoDataset(name='{self.dataset}', "
            f"num_episodes={len(self.episodes)}, "
            f"idx_offset={self.idx_offset}, "
            f"frameskip={self.frameskip}, "
            f"num_steps={self.num_steps})"
        )

    def __getitem__(self, index):
        episode = self.idx_to_episode[index]
        episode_indices = self.episode_indices[episode+self.idx_offset]
        offset = index - self.episode_starts[episode]

        # determine clip bounds
        start = offset if not self.complete_traj else 0
        stop = start + self.clip_len if not self.complete_traj else len(self.episode_indices[episode+self.idx_offset])
        step_slice = episode_indices[start:stop]
        steps = self.dataset[step_slice]

        for col, data in steps.items():
            if col == "action":
                continue

            data = data[:: self.frameskip]
            steps[col] = data

            if col in self.decode_columns:
                steps[col] = self.decode(steps["data_dir"], steps[col], start=start, end=stop)

        if self.transform:
            steps = self.transform(steps)

        # stack frames
        for col in self.decode_columns:
            if col not in steps:
                continue
            steps[col] = torch.stack(steps[col])

        # reshape action
        if "action" in steps:
            act_shape = self.num_steps if not self.complete_traj else len(self.episode_indices[episode+self.idx_offset])
            steps["action"] = steps["action"].reshape(act_shape, -1)

        return steps

