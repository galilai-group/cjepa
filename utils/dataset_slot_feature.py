import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

class SlotFeatureDataset(Dataset):
    """
    Dataset for loading pre-extracted slot features from .npz files.
    Each sample is a window of slot features and corresponding labels.
    """
    def __init__(self, feature_dir, window_size=8, stride=2, pattern=None):
        self.feature_dir = Path(feature_dir)
        self.window_size = window_size
        self.stride = stride
        self.samples = []
        self.pattern = pattern if pattern is not None else 'slot_features_*.npz'
        self._load_all(self.pattern)

    def _load_all(self, pattern):
        npz_files = sorted(self.feature_dir.glob(pattern))
        for npz_file in tqdm(npz_files):
            data = np.load(npz_file)
            slots = data['slots']  # (T, num_slots, dim)
            # subtract mean on each frame
            slots = slots - slots.mean(axis=1, keepdims=True)
            labels = data['labels']  # (T,)
            video_indices = data['video_indices']
            # Sliding window
            for start_idx in range(0, len(slots) - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                self.samples.append({
                    'slots': slots[start_idx:end_idx],
                    'labels': labels[start_idx:end_idx],
                    'video_indices': video_indices[start_idx:end_idx],
                    'video_num': data['video_num'].item() if 'video_num' in data else npz_file.stem.split('_')[-1],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Return as torch tensors
        return (
            torch.tensor(sample['slots'], dtype=torch.float32),
            torch.tensor(sample['labels'], dtype=torch.float32),
        )
