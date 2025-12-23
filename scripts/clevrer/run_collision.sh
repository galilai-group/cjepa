#!/bin/bash
#SBATCH --job-name=collision
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=11
#SBATCH --mem=150G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

python utils/train_collision.py \
    checkpoint_path="world_model_reg_epoch_30_object.ckpt" 