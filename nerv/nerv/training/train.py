"""Template train.py"""

import os
import sys
import pwd
import importlib
import argparse
import wandb

import torch

from datamodule import BaseDataModule
from method import BaseMethod
from model import BaseModel


def main(params):
    if args.fp16:
        print("INFO: using FP16 training!")
    if args.ddp:
        print("INFO: using DDP training!")

    print('Loading data...')
    # train_set, val_set = build_dataset()
    datamodule = BaseDataModule(
        params, train_set=None, val_set=None, use_ddp=args.ddp)

    print('Building model...')
    model = BaseModel()

    exp_name = os.path.basename(args.params)
    ckp_path = os.path.join(CHECKPOINT, exp_name, 'models')
    if args.local_rank == 0:
        os.makedirs(os.path.dirname(ckp_path), exist_ok=True)
        wandb_name = f'{exp_name}-{SLURM_JOB_ID}'

        # on clusters, quota is limited
        # soft link temp space for checkpointing
        if SLURM_JOB_ID and os.path.isdir('/checkpoint/'):
            if not os.path.exists(ckp_path):
                usr = pwd.getpwuid(os.getuid())[0]
                os.system(r'ln -s /checkpoint/{}/{}/ {}'.format(
                    usr, SLURM_JOB_ID, ckp_path))
                wandb_id = wandb_name
            else:
                os.makedirs(ckp_path, exist_ok=True)
                wandb_id = None

        wandb.init(
            project=params.project,
            name=wandb_name,
            id=wandb_id,
            dir=ckp_path,
        )
        # it's not good to hard-code the wandb id
        # but on preemption clusters, we want the job to resume the same
        # wandb process after resuming training
        # so I have to keep the same wandb id

    method = BaseMethod(
        model,
        datamodule,
        params,
        ckp_path=ckp_path,
        local_rank=args.local_rank,
        use_ddp=args.ddp,
        use_fp16=args.fp16,
    )

    print('Training...')
    method.fit(
        resume_from=args.weight, san_check_val_step=params.san_check_val_step)
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NERV model training')
    parser.add_argument('--params', type=str, default='params')
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.BaseParams()
    params.ddp = args.ddp

    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    CHECKPOINT = './checkpoint/'
    if args.cudnn:
        torch.backends.cudnn.benchmark = True

    main(params)
