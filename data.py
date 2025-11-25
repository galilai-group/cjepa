import torch
import os
import json
import numpy as np
import torchvision

from torchvision.io import read_video


class VideoStepsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        *args,
        num_steps=2,
        frameskip=1,
        cache_dir=None,
        split=None,
        transform=None,
        **kwargs,
    ):
        self.frameskip = frameskip
        self.n_steps = num_steps
        self.path = os.path.join(path, split + ".json") if split else path
        self.transform = transform

        assert os.path.exists(self.path), f"Dataset path {self.path} does not exist"
        with open(self.path, "r") as f:
            data = json.load(f)
        self.base_dir = data["root_dir"]
        self.recs = data["records"]  # list of dicts: {'video_path', 'num_frames', ...}

        # compute valid samples per video (episode)
        self.valid_per_video = []
        for rec in self.recs:
            L = int(rec.get("num_frames", 0))
            valid = max(0, L - self.n_steps * self.frameskip + 1)
            self.valid_per_video.append(valid)

        # cumulative slice starts for each episode in sample-space
        self.cum_slices = np.cumsum([0] + self.valid_per_video).astype(int)
        self.total_samples = int(self.cum_slices[-1])

        # map from sample index 
        if self.total_samples > 0:
            self.idx_to_ep = np.searchsorted(self.cum_slices, np.arange(self.total_samples), side="right") - 1 # starting from 0
        else:
            self.idx_to_ep = np.array([], dtype=int)

        # cache
        self._video_cache = {}
        self._max_cache_items = 16
    

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return int(self.total_samples)
    
    def __getitem__(self, idx):
        if idx < 0:
            idx = self.total_samples + idx

        if idx < 0 or idx >= self.total_samples:
            raise IndexError("index out of range")

        ep = int(self.idx_to_ep[idx]) # ep starting from 0
        offset = int(idx - self.cum_slices[ep])
        rec = self.recs[ep]
        start = offset
        idxs = list(range(start, start + self.n_steps * self.frameskip, self.frameskip))

        # load video frames with simple caching.. todo : lru?
        if ep in self._video_cache:
            video = self._video_cache[ep]
        else:
            video, _, _ = read_video(os.path.join(self.base_dir,  rec["video_path"]), output_format = "TCHW") # TCHW, 0-255 torch.unit8

            # cache
            if len(self._video_cache) >= self._max_cache_items:
                k = next(iter(self._video_cache))
                del self._video_cache[k]
            self._video_cache[ep] = video

        frames = video[idxs] / 255.0  # normalize to 0-1 float

        if self.transform is not None:
            frames = torch.stack([self.transform(frames[i]) for i in range(frames.shape[0])], dim=0)

        sample = {
            "pixels": frames,
            "goal": frames.clone(),
            "action": torch.zeros(self.n_steps, 1),
            "episode_idx": ep,
            "step_idx": torch.tensor(idxs, dtype=torch.long), # but this is a local index.. do i need global? 
            "episode_len": int(rec.get("num_frames", frames.shape[0])),
        }

        return sample
