"""A unified training script for all models used in the SlotFormer project."""

import os
import sys
import pwd
import importlib
from pathlib import Path
import argparse
import wandb
from loguru import logger as logging
from omegaconf import OmegaConf
import torch
import hydra
from torch.utils.data import DataLoader
from model.dinowm_causal import CausalWM
from model.cjepa_predictor import MaskedSlotPredictor
from third_party.videosaur.videosaur import  models
import stable_pretraining as spt
import stable_worldmodel as swm
from nerv.utils import mkdir_or_exist
from datetime import datetime
from torchvision.utils import save_image
from nerv.training import BaseDataModule
import shutil

DINO_PATCH_SIZE = 14  # DINO encoder uses 14x14 patches

# ============================================================================
# Main Entry Point
# ============================================================================


def main(params):
    # build datamodule
    # data = get_data(cfg)
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

    model = build_model(params)
    ckpt = args.ckpt
    if ckpt and os.path.isfile(ckpt):
        logging.info(f"Loading model weights from {ckpt}")
        state_dict = torch.load(ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
        assert len(missing) == 0, f"Missing keys when loading pretrained weights: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys when loading pretrained weights: {unexpected}"


    if args.exp_name is None:
        exp_name = os.path.basename(args.params)  + f"_LR{params.lr}"
    else:
        exp_name = args.exp_name  +  f"_LR{params.lr}"
    if 'aloe' in args.params:
        info = params.slots_root.split('/')[-1][:-4]
    elif 'savi' in args.params:
        info = 'savi'
    ckp_path = os.path.join(args.out_dir, exp_name, info)       
    method = build_method(
        model=model,
        datamodule=datamodule,
        params=params,
        ckp_path=ckp_path,
        local_rank=args.local_rank,
        use_ddp=args.ddp,
        use_fp16=args.fp16,
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
        
        original_saveframe = video[timestamp]  # 3,64,64
        combined_saveframe = recon_combined[timestamp] # 3,64, 64
        recon_saveframe = [recons[timestamp][i] for i in range(recons.shape[1])] # list of 3,64,64
        masks_saveframe = [masks[timestamp][i] for i in range(masks.shape[1])] # list of 3,64,64
        #save torch tensor as image
        save_dir = os.path.join('savi_visualize', args.ckpt.split('/')[-1][:-4], f'video_{i.item()}')
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
    parser = argparse.ArgumentParser(description='SlotFormer training script')
    parser.add_argument('--task', type=str, default='clevrer_vqa')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--exp_name', type=str, default='vis')
    parser.add_argument('--out_dir', type=str, default='outputs/')
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--fp16', action='store_true', help='half-precision')
    parser.add_argument('--ddp', action='store_true', help='DDP training')
    parser.add_argument('--cudnn', action='store_true', help='cudnn benchmark')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ckpt', default="/cs/data/people/hnam16/savi_pretrained/ex101_20260108_023722_LR0.0001/savi/epoch/model_24.pth")
    args = parser.parse_args()

    # import `build_dataset/model/method` function according to `args.task`
    print(f'INFO: training model in {args.task} task!')
    task = importlib.import_module(f'slotformer.{args.task}')
    if args.ddp:
        args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    build_dataset = task.build_dataset
    build_model = task.build_model
    build_method = task.build_method

    # load the params
    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()
    params.ddp = args.ddp

    if args.fp16:
        print('INFO: using FP16 training!')
    if args.ddp:
        print('INFO: using DDP training!')
    if args.cudnn:
        torch.backends.cudnn.benchmark = True
        print('INFO: using cudnn benchmark!')

    main(params)
