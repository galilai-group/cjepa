#!/bin/bash
#SBATCH --job-name=rollout_ocvp
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --output=extract-%j.out
#SBATCH --error=extract-%j.err

echo "=============================================="
echo "OCVP Rollout (128 -> 160 frames)"
echo "=============================================="
echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

# Default paths (can be overridden via command line)
CHECKPOINT="/cs/data/people/hnam16/ocvp_checkpoints/ocvp_clevrer_epoch_20_imgfalse.pth"
SLOTS_PATH="/cs/data/people/hnam16/data/modified_extraction/clevrer_savi_reproduced.pkl"
SAVE_PATH="/cs/data/people/hnam16/data/modified_extraction"

python train/train_ocvp_clevrer.py \
    rollout.rollout_only=true \
    rollout.checkpoint=${CHECKPOINT} \
    rollout.slots_path=${SLOTS_PATH} \
    rollout.save_path=${SAVE_PATH} \
    rollout_batch_size=8 \
    "$@"

echo "=============================================="
echo "End Time: $(date)"
echo "=============================================="
