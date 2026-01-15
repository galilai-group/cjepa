#!/bin/bash
#SBATCH --job-name=108videosaur
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=30G
#SBATCH --output=pusht_ind-%j.out
#SBATCH --error=pusht_ind-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

python videosaur/videosaur/train.py \
    videosaur/configs/videosaur/pusht_dinov2_hf.yml \
    globals.BATCH_SIZE_PER_GPU=64 \
    globals.BASE_LR=0.0001 \
    globals.SIM_WEIGHT=0.5 \
    globals.SIM_TEMP=0.15 \
    dataset.num_workers=8 \
    dataset.num_val_workers=1 \
    globals.SLOT_DIM=128 \
    dataset.val_pipeline.num_frameskip=2 \
    dataset.train_pipeline.num_frameskip=2 \
    # --continue logs/videosaur/2025-12-24-19-01-14_clevrer_dinov2  \

