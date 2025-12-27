"""Extract per-frame slot representations from CLEVRER videos using a Videosaur model.

This script mirrors the behavior of `extract_slots.py` for SAVi but uses Videosaur's
`ObjectCentricModel` API (initializer, encoder, processor). It processes each video
in non-overlapping temporal chunks and passes the last slot state of each chunk to
the next chunk so slots are continuous across long videos.

Saved format (pickle): {'train': {video_basename: slots_array}, 'val': {...}, 'test': {...}}
where `slots_array` has shape [T, n_slots, slot_dim].

Usage example (also included in README_extract_videosaur.md):

python slotformer/base_slots/extract_videosaur.py \
    --params configs/config_train.yaml \
    --videosaur_config videosaur/configs/inference/clevrer_dinov2.yml \
    --weight /path/to/checkpoint.ckpt \
    --save_path ./data/CLEVRER/videosaur_slots.pkl

Notes:
- The script uses `build_dataset(params)` to build the CLEVRER train/val datasets.
- Videosaur model is built via `videosaur.videosaur.models.build` using the provided
  YAML config, and checkpoint is loaded via `torch.load(checkpoint)['state_dict']`.
- We save `processor.state` (after-corrector slots) for each frame. If you prefer
  the initial state included, inspect `processor.all_slot_states` which this code also
  retains during forward.
"""

import os
import sys
import argparse
import importlib
import pickle
from tqdm import tqdm
import glob
import torchvision.transforms.v2 as transforms

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torchcodec.decoders import VideoDecoder
from nerv.utils import dump_obj, mkdir_or_exist

from slotformer.base_slots.datasets import build_dataset, build_clevrer_dataset
from videosaur.videosaur import configuration, models
# ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def read_video(video_path, num_frameskip, start_idx=0, return_idx=False, device='cuda'):
    '''
    Docstring for read_video
    input: video file path
    output: video frames for a whole video with a given frameskip
    '''
    video = VideoDecoder(video_path)
    if return_idx:
        idx = np.arange(len(video))[0 : -1 : num_frameskip]
        return video[start_idx : -1 : num_frameskip].to(device), idx
    else:
        return video[start_idx : -1 : num_frameskip].to(device)

@torch.no_grad()
def extract_video_slots_videosaur(model, dataset, chunk_len=None, num_frameskip=2, device='cuda'):
    """Extract slots for each video in `dataset`.

    Args:
        model: videosaur ObjectCentricModel, already loaded and eval().
        dataset: CLEVRERDataset or similar with `get_video(i)` returning dict{'video': Tensor[T,C,H,W]}.
        chunk_len: int or None. If None, process the entire video at once; else split into non-overlapping chunks.
        device: 'cuda' or 'cpu'.

    Returns:
        all_slots: list of np arrays, each with shape [T, n_slots, slot_dim].
    """
    model.eval()
    torch.cuda.empty_cache()

    all_slots = []

    for i in tqdm(range(len(dataset))):
        data = read_video(dataset[i], num_frameskip, return_idx=False)
        tfs = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),      
            transforms.Resize((196, 196)),                 
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        video = tfs(data)  # [T, C, H, W]
        T = video.shape[0]

        # reshape to batch dim
        video_b = video.unsqueeze(0).to(device)  # [1, T, C, H, W]

        # Use model.encoder to get features: should return dict with 'features' shaped [B, T, num_inputs, feat_dim]
        encoder_out = model.encoder(video_b)
        features = encoder_out['features']  # [B, T, N, D]

        B = 1

        # decide chunk length
        if chunk_len is None:
            # process whole video
            slots_init = model.initializer(batch_size=B).to(device)
            out = model.processor(slots_init, features)
            # `state` is [B, T, n, c]
            slots_np = out['state'][0].detach().cpu().numpy()
            all_slots.append(slots_np)
        else:
            prev = None
            collected = []
            for s in range(0, T, chunk_len):
                feats_chunk = features[:, s : min(s + chunk_len, T)]  # [B, t_chunk, N, D]
                if prev is None:
                    init = model.initializer(batch_size=B).to(device)
                else:
                    init = prev.to(device)
                out = model.processor(init, feats_chunk)
                state = out['state']  # [B, t_chunk, n, c]
                collected.append(state[0].detach().cpu().numpy())
                # take last state's predicted/next state for continuity
                prev = out['state'][:, -1].detach().clone()
            slots_np = np.concatenate(collected, axis=0)
            all_slots.append(slots_np)

    return all_slots


def process_videosaur(model, params, args):
    """Build CLEVRER dataset(s), extract slots and save to `args.save_path`"""

    train_set = glob.glob("/users/hnam16/scratch/.stable_worldmodel/clevrer_train/videos/*.mp4")
    val_set = glob.glob("/users/hnam16/scratch/.stable_worldmodel/clevrer_val/videos/*.mp4")

    # choose chunk length: prefer params.input_frames if exists
    chunk_len = None # getattr(params, 'input_frames', None)

    print(f'Processing {params.dataset} video val set...')
    val_slots = extract_video_slots_videosaur(model, val_set, chunk_len=chunk_len, num_frameskip=args.num_frameskip)
    print(f'Processing {params.dataset} video train set...')
    train_slots = extract_video_slots_videosaur(model, train_set, chunk_len=chunk_len, num_frameskip=args.num_frameskip)

    # also extract test_set for CLEVRER
    test_slots = None
    if params.dataset == 'clevrer':
        test_set = glob.glob("/users/hnam16/scratch/.stable_worldmodel/clevrer_test/videos/*.mp4")
        print(f'Processing {params.dataset} video test set...')
        test_slots = extract_video_slots_videosaur(model, test_set, chunk_len=chunk_len, num_frameskip=args.num_frameskip)

    # pack to dict: map video basename -> slots
    try:
        train_slots = {
            os.path.basename(train_set.files[i]): train_slots[i]
            for i in range(len(train_slots))
        }
        val_slots = {
            os.path.basename(val_set.files[i]): val_slots[i]
            for i in range(len(val_slots))
        }
        slots = {'train': train_slots, 'val': val_slots}

        if test_slots is not None:
            test_slots = {
                os.path.basename(test_set.files[i]): test_slots[i]
                for i in range(len(test_slots))
            }
            slots['test'] = test_slots

        mkdir_or_exist(os.path.dirname(args.save_path))
        dump_obj(slots, args.save_path)
        print(f'Finish {params.dataset} video dataset, '
              f'train: {len(train_slots)}/{train_set.num_videos}, '
              f'val: {len(val_slots)}/{val_set.num_videos}')
        if test_slots is not None:
            print(f'test: {len(test_slots)}/{test_set.num_videos}')
    except Exception:
        # debugging fallback
        import pdb
        pdb.set_trace()

    # create soft link to the weight dir
    ln_path = os.path.join(os.path.dirname(args.weight), 'videosaur_slots.pkl')
    os.system(r'ln -s {} {}'.format(args.save_path, ln_path))


def main():
    parser = argparse.ArgumentParser(description='Extract slots from videos (Videosaur)')
    parser.add_argument('--params', default="slotformer/clevrer_vqa/configs/aloe_clevrer_params.py", type=str, )
    parser.add_argument('--videosaur_config', default="videosaur/configs/videosaur/pusht_dinov2_hf.yml", type=str, 
                        help='path to videosaur YAML config')
    parser.add_argument('--weight', default = "logs/videosaur_pusht/2025-12-26-15-04-12_pusht_dinov2/checkpoints/step=100000_sim0.1.ckpt", type=str,  help='pretrained model weight')
    parser.add_argument('--save_path', default="/cs/data/people/hnam16/data/clevrer_slots", type=str,  help='path to save slots')
    parser.add_argument('--num_frameskip', default=1, type=int)

    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotFormerParams()

    # sanity check
    assert params.dataset in args.save_path

    # load videosaur config & model
    conf = configuration.load_config(args.videosaur_config)
    model = models.build(conf.model, conf.optimizer)
    ckpt = torch.load(args.weight, map_location='cpu')
    # load state-dict
    model.load_state_dict(ckpt['state_dict'])
    model = model.eval().cuda()

    # Append checkpoint basename to save_path
    ckpt_name = os.path.splitext(os.path.basename(args.weight))[0]
    save_dir = os.path.dirname(args.save_path)
    base, ext = os.path.splitext(os.path.basename(args.save_path))
    if ext == '':
        ext = '.pkl'
    args.save_path = os.path.join(save_dir, f"{base}_{ckpt_name}{ext}")

    # ensure initializer slots match if needed
    if hasattr(conf, 'model') and hasattr(conf.globals, 'NUM_SLOTS'):
        try:
            model.initializer.n_slots = conf.globals.NUM_SLOTS
        except Exception:
            pass

    # run extraction
    process_videosaur(model, params, args)


if __name__ == '__main__':
    main()
