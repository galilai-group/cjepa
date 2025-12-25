#!/bin/bash
#SBATCH --job-name=videosaur
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=sa-%j.out
#SBATCH --error=sa-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

python videosaur/videosaur/train.py \
    videosaur/configs/videosaur/clevrer_dinov2_hf.yml \
    globals.BATCH_SIZE_PER_GPU=32 \
    globals.BASE_LR=0.0001 \
    globals.SIM_WEIGHT=0.1 \
    dataset.num_workers=4 \
    dataset.num_val_workers=4 \
    --continue logs/videosaur/2025-12-24-19-00-21_clevrer_dinov2  \

