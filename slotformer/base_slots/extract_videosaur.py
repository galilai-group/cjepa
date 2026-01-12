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
import traceback

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
    # sample indices to include the last frame (avoid using -1 which excludes it)
    indices = np.arange(len(video))[start_idx::num_frameskip]
    frames = video[start_idx::num_frameskip].to(device)
    if return_idx:
        return frames, indices
    else:
        return frames

@torch.no_grad()
def extract_video_slots_videosaur(model, dataset, chunk_len=None, num_frameskip=1, device='cuda'):
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
        # all_slots.append(dataset[i])
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

    # choose chunk length: prefer params.input_frames if exists
    chunk_len = None  # getattr(params, 'input_frames', None)

    # gather file lists
    train_set = glob.glob(os.path.join(args.data_root, f"{args.dataset}_train/videos/*.mp4"))
    val_set = glob.glob(os.path.join(args.data_root, f"{args.dataset}_val/videos/*.mp4"))
    test_set = glob.glob(os.path.join(args.data_root, f"{args.dataset}_test/videos/*.mp4")) 
    # also extract test_set for CLEVRER
    test_slots = None
    if len(test_set) > 0:
        print(f'Processing {params.dataset} video test set...')
        test_slots = extract_video_slots_videosaur(model, test_set, chunk_len=chunk_len, num_frameskip=args.num_frameskip)

    print(f'Processing {params.dataset} video val set...')
    val_slots = extract_video_slots_videosaur(model, val_set, chunk_len=chunk_len, num_frameskip=args.num_frameskip)

    print(f'Processing {params.dataset} video train set...')
    train_slots = extract_video_slots_videosaur(model, train_set, chunk_len=chunk_len, num_frameskip=args.num_frameskip)
    
    def map_files_slots(file_list, slots_list, split_name):
        if len(file_list) != len(slots_list):
            # save partial to help debugging
            partial = {os.path.basename(file_list[i]): slots_list[i]
                        for i in range(min(len(file_list), len(slots_list)))}
            partial_path = os.path.join(os.path.dirname(args.save_path), f'partial_{split_name}.pkl')
            try:
                mkdir_or_exist(os.path.dirname(partial_path))
                with open(partial_path, 'wb') as pf:
                    pickle.dump(partial, pf)
            except :
                print(f"Failed to write partial {split_name}")
        # return_dict = {}
        # for i in range(len(slots_list)):
        #     new_name = int(os.path.basename(file_list[i]).split('_')[1].split('.')[0]) + '_pixels.mp4'
        #     return_dict[new_name] = slots_list[i]
        # return return_dict
        return {os.path.basename(file_list[i]): slots_list[i] for i in range(len(slots_list))}


    train_slots_map = map_files_slots(train_set, train_slots, 'train')
    val_slots_map = map_files_slots(val_set, val_slots, 'val')
    slots = {'train': train_slots_map, 'val': val_slots_map}

    if test_slots is not None:
        test_slots_map = map_files_slots(test_set, test_slots, 'test')
        slots['test'] = test_slots_map

    # atomic save (write to temp then replace) and smoke-load to verify
    mkdir_or_exist(os.path.dirname(args.save_path))


    with open(args.save_path, 'wb') as f:
        pickle.dump(slots, f)



    # with open(args.save_path, "rb") as fr:
    #     obj = pickle.load(fr)


def main():
    parser = argparse.ArgumentParser(description='Extract slots from videos (Videosaur)')
    parser.add_argument('--params', default="slotformer/clevrer_vqa/configs/aloe_clevrer_params.py", type=str, )
    parser.add_argument('--data_root', default="/cs/data/people/hnam16/.stable_worldmodel")
    parser.add_argument('--videosaur_config', default="videosaur/configs/videosaur/clevrer_dinov2_hf.yml", type=str, 
                        help='path to videosaur YAML config')
    parser.add_argument('--weight', default = "logs/videosaur/2025-12-25-19-27-13_clevrer_dinov2/checkpoints/step=100000.ckpt", type=str,  help='pretrained model weight')
    parser.add_argument('--save_path', default="/cs/data/people/hnam16/data/clevrer_slots", type=str,  help='path to save slots')
    parser.add_argument('--num_frameskip', default=1, type=int)
    parser.add_argument('--limit', default=None, type=int, help='limit number of videos per split (for smoke testing)')
    parser.add_argument('--smoke', action='store_true', help='run a smoke test that skips model and uses fake slots to test save/validation logic')
    parser.add_argument('--dataset', default='clevrer', type=str, help='dataset name to match save_path')
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
