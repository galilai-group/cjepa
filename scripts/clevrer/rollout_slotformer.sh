#!/bin/bash
#SBATCH --job-name=rollout
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_titan_rtx:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --output=extract-%j.out
#SBATCH --error=extract-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

python slotformer/video_prediction/rollout_clevrer_slots.py \
    --params slotformer/video_prediction/configs/slotformer_clevrer_params.py \
    --weight /cs/data/people/hnam16/aloe_checkpoint/195_slotformer.pth \
    --save_path /cs/data/people/hnam16/data/modified_extraction/