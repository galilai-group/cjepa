import os
import os.path as osp
import numpy as np
from PIL import Image, ImageFile
import glob
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm
from nerv.utils import glob_all, load_obj, strip_suffix, read_img
import torchvision.transforms.v2 as T


from .utils import BaseTransforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PushTSynDataset(Dataset):
    """Pusht dataset from"""

    def __init__(
        self,
        data_root,
        split,
        pusht_transform,
        resolution=(64, 64),
        n_sample_frames=6,
        frame_offset=None,
        video_len=50
    ):

        assert split in ['train', 'val', 'test']
        self.data_root = os.path.join(data_root, split)
        self.split = split
        self.pusht_transform = pusht_transform
        self.n_sample_frames = n_sample_frames
        self.video_len=video_len
        self.frame_offset = video_len // n_sample_frames if \
            frame_offset is None else frame_offset        
        self.resolution = resolution

        mean = (0.5, 0.5, 0.5)
        std  = (0.5, 0.5, 0.5)
        self.video_transform = T.Compose([
            T.ToImage(),                           # Tensor → ImageTensor
            T.ToDtype(torch.float32, scale=True), # uint8 → float [0,1]
            T.Resize(resolution),
            T.Normalize(mean, std),  # [-1, 1]

        ])
        # Get all numbers
        self.files = self._get_files()
        self.num_videos = len(self.files)
        self.valid_idx = self._get_sample_idx()

        # by default, we load small video clips
        self.load_video = False

    def _get_video_start_idx(self, idx):
        return self.valid_idx[idx]

    def _read_frames(self, idx):
        """Read video frames. Directly read from jpg images if possible."""  
        video_idx, start_idx = self._get_video_start_idx(idx)
        video_path = self.files[video_idx]
        # if videos are not converted to frames, read from mp4 file
        frame_list = [start_idx + n * self.frame_offset for n in range(self.n_sample_frames)]
        video = VideoDecoder(video_path)[:][frame_list] #[T, C, H, W]
        # video = video.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        video = self.video_transform(video)  # [T, C, H, W]

        return video.float()


    def _read_bboxes(self, idx):
        """Load empty bbox and pres mask for compatibility."""
        bboxes = np.zeros((self.n_sample_frames, 5, 4))
        pres_mask = np.zeros((self.n_sample_frames, 5))
        return bboxes, pres_mask

    def _get_files(self):
        """Get path for all videos."""
        # test set doesn't have annotation json files
        video_paths = glob.glob(os.path.join(self.data_root, '*.mp4'))
        return sorted(video_paths)

    def get_video(self, video_idx):
        valid=False
        while not valid:
            try: 
                video_path = self.files[video_idx]
                video = VideoDecoder(video_path)[:][:self.video_len] #[T, C, H, W]
                video = self.video_transform(video)  # [T, C, H, W]
                video = video[::self.frame_offset]
                valid=True
                # if video.shape[0] >= self.n_sample_frames:
                #     video = video[:self.n_sample_frames]
                #     valid=True
                # else:
                #     video_idx = (video_idx +1) % len(self.files)  
            except Exception as e:
                video_idx = (video_idx +1) % len(self.files)  

        # [T, H, W, C] >> [T, C, H, W] 
        # video = video.permute(0, 3, 1, 2).contiguous()


        return {
            'video': video,
            'error_flag': False,
            'data_idx': video_idx,
        }

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - bbox: [T, max_num_obj, 4], empty, for compatibility
            - pres_mask: [T, max_num_obj], empty, for compatibility
        """
        if self.load_video:
            return self.get_video(idx)

        frames = self._read_frames(idx)
        data_dict = {
            'data_idx': idx,
            'img': frames,
        }
        if self.split != 'train':
            bboxes, pres_mask = self._read_bboxes(idx)
            data_dict['bbox'] = torch.from_numpy(bboxes).float()
            data_dict['pres_mask'] = torch.from_numpy(pres_mask).bool()
        return data_dict

    def _get_sample_idx(self):
        """
            Get (video_idx, start_frame_idx) pairs as a list.
        """
        valid_idx = []
        for video_idx in tqdm(
            range(len(self.files)),
            desc=f'Indexing {self.split} videos',
        ):
            video_path = self.files[video_idx]
            video_len = self._get_video_num_frames(video_path)
            # simply use random uniform sampling
            max_start_idx = video_len - \
                (self.n_sample_frames - 1) * self.frame_offset
            if max_start_idx <= 0:
                continue
            if self.split == 'train':
                valid_idx += [(video_idx, idx) for idx in range(max_start_idx)]
            # in val/test we only sample each frame once
            else:
                size = self.n_sample_frames * self.frame_offset
                if video_len - size + 1 <= 0:
                    continue
                start_idx = []
                for idx in range(0, video_len - size + 1, size):
                    start_idx += [i + idx for i in range(self.frame_offset)]
                valid_idx += [(video_idx, idx) for idx in start_idx]
        return valid_idx

    @staticmethod
    def _get_video_num_frames(video_path):
        """Read frame count from metadata when possible to avoid full decode."""
        decoder = VideoDecoder(video_path)
        num_frames = None
        if hasattr(decoder, 'num_frames'):
            num_frames = decoder.num_frames
        elif hasattr(decoder, 'metadata'):
            try:
                meta = decoder.metadata() if callable(decoder.metadata) else decoder.metadata
                if isinstance(meta, dict):
                    num_frames = meta.get('num_frames') or meta.get('video', {}).get('num_frames')
            except Exception:
                num_frames = None
        if num_frames is None:
            try:
                num_frames = len(decoder)
            except TypeError:
                num_frames = decoder[:].shape[0]
        return int(num_frames)

    def __len__(self):
        if self.load_video:
            return len(self.files)
        return len(self.valid_idx)


class PushTSynSlotsDataset(PushTSynDataset):
    """pusht dataset from G-SWM with pre-computed slots."""

    def __init__(
        self,
        data_root,
        video_slots,
        split,
        pusht_transform,
        n_sample_frames=16,
        frame_offset=None,
        video_len=70
    ):
        super().__init__(
            data_root=data_root,
            split=split,
            pusht_transform=pusht_transform,
            n_sample_frames=n_sample_frames,
            frame_offset=frame_offset,
        )

        # pre-computed slots
        self.video_slots = video_slots
        self.video_len = video_len

    def _read_slots(self, idx):
        """Read video frames slots."""
        folder, start_idx = self.valid_idx[idx]
        slots = self.video_slots[os.path.basename(folder)]  # [T, N, C]
        slots = [
            slots[start_idx + n * self.frame_offset]
            for n in range(self.n_sample_frames)
        ]
        return np.stack(slots, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - bbox: [T, max_num_obj, 4], empty, for compatibility
            - pres_mask: [T, max_num_obj], empty, for compatibility
            - slots: [T, N, C] slots extracted from PushT video frames
        """
        slots = self._read_slots(idx)
        frames = self._read_frames(idx)
        data_dict = {
            'data_idx': idx,
            'slots': slots,
            'img': frames,
        }
        if self.split != 'train':
            bboxes, pres_mask = self._read_bboxes(idx)
            data_dict['bbox'] = torch.from_numpy(bboxes).float()
            data_dict['pres_mask'] = torch.from_numpy(pres_mask).bool()
        return data_dict


def build_pusht_syn_dataset(params, val_only=False):
    """Build PushT video dataset."""
    args = dict(
        data_root=params.data_root,
        split='val',
        pusht_transform=BaseTransforms(params.resolution),
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
    )
    val_dataset = PushTSynDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = PushTSynDataset(**args)
    return train_dataset, val_dataset


def build_pusht_syn_slots_dataset(params, val_only=False):
    """Build pusht video dataset with pre-computed slots."""
    slots = load_obj(params.slots_root)
    args = dict(
        data_root=params.data_root,
        video_slots=slots['val'],
        split='val',
        pusht_transform=BaseTransforms(params.resolution),
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        resolution = params.resolution
    )
    val_dataset = PushTSynSlotsDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    args['video_slots'] = slots['train']
    train_dataset = PushTSynSlotsDataset(**args)
    return train_dataset, val_dataset
