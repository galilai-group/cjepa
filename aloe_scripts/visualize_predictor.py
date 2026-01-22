"""A unified training script for all models used in the SlotFormer project."""

import os
import sys
import importlib
from loguru import logger as logging

import torch
import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
from nerv.utils import mkdir_or_exist
from datetime import datetime
from torchvision.utils import save_image
from nerv.training import BaseDataModule
import shutil

from torch.nn import functional as F
from einops import rearrange, repeat
from custom_models.cjepa_predictor import MaskedSlotPredictor
from custom_models.dinowm_causal_savi import CausalWM_Savi
from slotformer.base_slots.models import build_model
import sys
import importlib
import pickle as pkl

from torch.utils.data import Dataset, DataLoader


DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches

class PushTSlotDataset(Dataset):
    """
    Dataset for pre-extracted slot representations from PushT.
    
    This class mirrors the behavior of swm.data.VideoDataset to ensure
    identical data processing. Key behaviors:
    - Window stride of 1 (not frameskip) for sample indices
    - Action is reshaped to (T, action_dim * frameskip) by VideoDataset
    - Normalization uses mean/std without clamping (same as WrapTorchTransform)
    - nan_to_num is only applied in forward pass, not in dataset
    
    Each sample contains:
    - pixels_embed: Pre-extracted slot embeddings (T, num_slots, slot_dim)
    - action: Action sequence (T, action_dim * frameskip)
    - proprio: Proprioception sequence (T, proprio_dim)
    - state: State sequence (T, state_dim) [optional, for evaluation]
    
    Args:
        slot_data: Dict mapping video_id to slot embeddings
        split: 'train' or 'val'
        history_size: Number of history frames
        num_preds: Number of future frames to predict
        action_dir: Path to action pickle file
        proprio_dir: Path to proprioception pickle file
        state_dir: Path to state pickle file (optional)
        frameskip: Frame skip factor (affects action reshaping)
        seed: Random seed for sampling
    """
    
    def __init__(
        self,
        slot_data: dict,
        split: str,
        history_size: int,
        num_preds: int,
        action_dir: str,
        proprio_dir: str,
        state_dir: str = None,
        frameskip: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        self.slot_data = slot_data
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.frameskip = frameskip
        self.n_steps = history_size + num_preds
        self.seed = seed
        
        # Load action and proprio data
        with open(action_dir, "rb") as f:
            action_data = pkl.load(f)
        self.action_data = action_data[split]
        
        with open(proprio_dir, "rb") as f:
            proprio_data = pkl.load(f)
        self.proprio_data = proprio_data[split]
        
        # State is optional (used for evaluation)
        self.state_data = None
        if state_dir is not None:
            with open(state_dir, "rb") as f:
                state_data = pkl.load(f)
            self.state_data = state_data[split]
        
        # Build index: list of (video_id, start_frame) tuples
        self.samples = self._build_sample_index()
        
        # Compute normalization statistics (matching WrapTorchTransform behavior)
        self._compute_normalization_stats()
        
        logging.info(f"[{split}] Created dataset with {len(self.samples)} samples from {len(self.slot_data)} videos")
    
    def _build_sample_index(self):
        """
        Build list of valid (video_id, start_frame) samples.
        
        Matches VideoDataset behavior: stride of 1, not frameskip.
        VideoDataset uses: episode_max_end = max(0, len(ep) - clip_len + 1)
        and iterates over all start positions with stride 1.
        """
        samples = []
        clip_len = self.n_steps * self.frameskip
        
        for video_id, slots in self.slot_data.items():
            num_frames = slots.shape[0]
            # max_start is inclusive, so we can start at positions 0 to max_start
            max_start = num_frames - clip_len
            
            if max_start < 0:
                continue
            
            # Stride 1 matching VideoDataset behavior
            for start_idx in range(0, max_start + 1):
                samples.append((video_id, start_idx))
        
        return samples
    
    def _compute_normalization_stats(self):
        """
        Compute mean and std for action and proprio normalization.
        
        Matches WrapTorchTransform(norm_col_transform(dataset, col)) behavior:
        - Computes stats over the RESHAPED action column (T, action_dim * frameskip)
        - No clamping of std (WrapTorchTransform doesn't clamp)
        - Uses tensor mean/std with unsqueeze(0)
        
        Note: VideoDataset reshapes action to (T, -1) before transform is applied.
        """
        # Collect all actions and proprios in their RESHAPED form
        # This matches how VideoDataset provides data to the transform
        all_actions = []
        all_proprios = []
        
        for video_id in self.action_data.keys():
            action_raw = self.action_data[video_id]  # (num_frames, action_dim)
            # Reshape to match VideoDataset's reshape: (T, action_dim * frameskip)
            # VideoDataset does: steps["action"].reshape(act_shape, -1)
            # where act_shape = num_steps and the raw actions are clip_len = n_steps * frameskip
            # So each T gets frameskip consecutive actions flattened
            num_frames = action_raw.shape[0]
            clip_len = self.n_steps * self.frameskip
            
            # Iterate over all possible clips (stride 1, matching _build_sample_index)
            for start_idx in range(0, num_frames - clip_len + 1):
                # Get clip_len consecutive raw actions
                action_clip = action_raw[start_idx:start_idx + clip_len]  # (clip_len, action_dim)
                # Reshape to (n_steps, action_dim * frameskip) - matching VideoDataset
                action_reshaped = action_clip.reshape(self.n_steps, -1)
                all_actions.append(action_reshaped)
        
        for video_id in self.proprio_data.keys():
            proprio_raw = self.proprio_data[video_id]  # (num_frames, proprio_dim)
            num_frames = proprio_raw.shape[0]
            clip_len = self.n_steps * self.frameskip
            
            for start_idx in range(0, num_frames - clip_len + 1):
                # Get frames with frameskip (matching VideoDataset: data[::frameskip])
                frame_indices = [start_idx + i * self.frameskip for i in range(self.n_steps)]
                if frame_indices[-1] < num_frames:
                    proprio_clip = proprio_raw[frame_indices]  # (n_steps, proprio_dim)
                    all_proprios.append(proprio_clip)
        
        # Stack and compute stats matching norm_col_transform:
        # data.mean(0).unsqueeze(0), data.std(0).unsqueeze(0)
        all_actions = torch.from_numpy(np.concatenate(all_actions, axis=0)).float()  # (N*T, action_dim*frameskip)
        all_proprios = torch.from_numpy(np.concatenate(all_proprios, axis=0)).float()  # (N*T, proprio_dim)
        
        # Match norm_col_transform: mean(0).unsqueeze(0), std(0).unsqueeze(0)
        self.action_mean = all_actions.mean(0).unsqueeze(0)  # (1, action_dim * frameskip)
        self.action_std = all_actions.std(0).unsqueeze(0)    # (1, action_dim * frameskip)
        
        self.proprio_mean = all_proprios.mean(0).unsqueeze(0)  # (1, proprio_dim)
        self.proprio_std = all_proprios.std(0).unsqueeze(0)    # (1, proprio_dim)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_id, start_idx = self.samples[idx]
        
        # clip_len = n_steps * frameskip raw frames
        clip_len = self.n_steps * self.frameskip
        
        # Get frame indices with frameskip for slots (matching VideoDataset: data[::frameskip])
        frame_indices = [start_idx + i * self.frameskip for i in range(self.n_steps)]
        
        # Extract slot embeddings: (n_steps, num_slots, slot_dim)
        slots = self.slot_data[video_id]
        pixels_embed = torch.from_numpy(slots[frame_indices]).float()
        
        # Extract and reshape actions (matching VideoDataset behavior)
        # VideoDataset gets clip_len consecutive raw actions, then reshapes to (n_steps, -1)
        action_raw = self.action_data[video_id]
        action_clip = action_raw[start_idx:start_idx + clip_len]  # (clip_len, action_dim)
        # Reshape to (n_steps, action_dim * frameskip) - matching VideoDataset's reshape
        action = torch.from_numpy(action_clip.reshape(self.n_steps, -1)).float()
        
        # Extract proprio with frameskip (matching VideoDataset: data[::frameskip])
        proprio_raw = self.proprio_data[video_id]
        proprio = torch.from_numpy(proprio_raw[frame_indices]).float()
        
        # Normalize action and proprio (matching WrapTorchTransform behavior)
        # Note: No nan_to_num here - that's done in forward pass like train_causalwm.py
        action = (action - self.action_mean) / self.action_std
        proprio = (proprio - self.proprio_mean) / self.proprio_std
        
        sample = {
            "pixels_embed": pixels_embed,  # (T, S, D)
            "action": action,              # (T, action_dim * frameskip)
            "proprio": proprio,            # (T, proprio_dim)
        }
        
        # Optionally include state
        if self.state_data is not None:
            state_raw = self.state_data[video_id]
            state = torch.from_numpy(state_raw[frame_indices]).float()
            sample["state"] = state
        
        return sample



@hydra.main(version_base=None, config_path="../configs", config_name="config_train_causal_pusht_slot_savi_visualize")
def main(cfg):
    # import `build_dataset/model/method` function according to `args.task`
    print(f'INFO: training model in {cfg.task} task!')
    task = importlib.import_module(f'slotformer.{cfg.task}')
    if cfg.ddp:
        cfg.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    build_dataset = task.build_dataset
    build_model = task.build_model
    build_method = task.build_method

    # load the params
    if cfg.params.endswith('.py'):
        cfg.params = cfg.params[:-3]
    sys.path.append(os.path.dirname(cfg.params))
    params = importlib.import_module(os.path.basename(cfg.params))
    params = params.SlotFormerParams()
    params.ddp = cfg.ddp

    if cfg.fp16:
        print('INFO: using FP16 training!')
    if cfg.ddp:
        print('INFO: using DDP training!')
    if cfg.cudnn:
        torch.backends.cudnn.benchmark = True
        print('INFO: using cudnn benchmark!')

    datasets = build_dataset(params)
    train_set, val_set = datasets[0], datasets[1]
    collate_fn = datasets[2] if len(datasets) == 3 else None
    datamodule = BaseDataModule(
        params,
        train_set=train_set,
        val_set=val_set,
        use_ddp=params.ddp,
        collate_fn=collate_fn,
    )

    decoder_model = build_model(params)
    ckpt = cfg.ckpt_for_decoder
    if ckpt and os.path.isfile(ckpt):
        logging.info(f"Loading model weights from {ckpt}")
        state_dict = torch.load(ckpt, map_location="cpu")
        missing, unexpected = decoder_model.load_state_dict(state_dict['state_dict'], strict=False)
        assert len(missing) == 0, f"Missing keys when loading pretrained weights: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys when loading pretrained weights: {unexpected}"

    # model = get_world_model(cfg)
    model = swm.policy.AutoCostModel(cfg.ckpt_for_worldmodel, cfg.cache_dir)
    predictor = model._modules['predictor']


    if cfg.exp_name is None:
        exp_name = os.path.basename(cfg.params)  + f"_LR{params.lr}"
    else:
        exp_name = cfg.exp_name  +  f"_LR{params.lr}"
    if 'aloe' in cfg.params:
        info = params.slots_root.split('/')[-1][:-4]
    elif 'savi' in cfg.params:
        info = 'savi'
    ckp_path = os.path.join(cfg.out_dir, exp_name, info)       
    method = build_method(
        model=decoder_model,
        datamodule=datamodule,
        params=params,
        ckp_path=ckp_path,
        local_rank=cfg.local_rank,
        use_ddp=cfg.ddp,
        use_fp16=cfg.fp16,
    )

    method.model.eval()
    dst = method.val_loader.dataset

    timestamp = 5

    sampled_idx = method._get_sample_idx(method.params.n_samples, dst)
    results, labels = [], []
    for i in sampled_idx:
        data_dict = dst.get_video(i.item())
        video, label = data_dict['video'].float().to(method.device), data_dict.get('label', None)  # label for PHYRE
        in_dict = {'img': video[None]}
        out_dict = method.model(in_dict) 
        out_dict = {k: v[0] for k, v in out_dict.items()}
        recon_combined, recons, masks = out_dict['post_recon_combined'], out_dict['post_recons'], out_dict['post_masks']
        post_slots = out_dict['post_slots'] # [50, 4, 128]
        predicted_slots = get_predicted_slots(predictor, post_slots)
        pred_post_recon_img, pred_post_recons, pred_post_masks, _ = method.model.decode(post_slots.flatten(0, 1))
        original_saveframe = video[timestamp]  # 3,64,64
        combined_saveframe = recon_combined[timestamp] # 3,64, 64
        recon_saveframe = [recons[timestamp][i] for i in range(recons.shape[1])] # list of 3,64,64
        masks_saveframe = [masks[timestamp][i] for i in range(masks.shape[1])] # list of 3,64,64
        #save torch tensor as image
        save_dir = os.path.join('savi_visualize', cfg.ckpt_for_decoder.split('/')[-1][:-4], f'video_{i.item()}')
        mkdir_or_exist(save_dir)
        save_image(original_saveframe, os.path.join(save_dir, f'original_frame_{timestamp}.png'))
        save_image(combined_saveframe, os.path.join(save_dir, f'combined_frame_{timestamp}.png'))
        for idx, m in enumerate(recon_saveframe):
            save_image(m, os.path.join(save_dir, f'recon_{idx}_frame_{timestamp}.png'))
        for idx, m in enumerate(masks_saveframe):
            save_image(m, os.path.join(save_dir, f'mask_{idx}_frame_{timestamp}.png'))

        imgs = video.type_as(recon_combined)
        save_video = method._make_video_grid(imgs, recon_combined, recons,
                                            masks)
        # results.append(save_video)

        video = method._convert_video([save_video], caption=None)
        video_path = video._path
        logging.info(f'Saved visualization video to {video_path}')
        
        shutil.copy(video_path, os.path.join(save_dir, 'visualization_video.gif'))

    # method.fit(n
    #     resume_from=args.weight, san_check_val_step=params.san_check_val_step)


if __name__ == "__main__":


    main()


