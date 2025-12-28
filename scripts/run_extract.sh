#!/bin/bash
#SBATCH --job-name=extract_videosaur
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --output=sa-%j.out
#SBATCH --error=sa-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

python slotformer/base_slots/extract_videosaur.py \
    --weight="logs/videosaur_clevrer/2025-12-27-02-42-16_clevrer_dinov2/checkpoints/step=100000_weight03_lr5e-4_clevrer.ckpt" \
    --data_root="/cs/data/people/hnam16/.stable_worldmodel" \
    --save_path="/cs/data/people/hnam16/data/clevrer_slots"