#!/bin/bash
#SBATCH --job-name=dinowmreg
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=3
#SBATCH --gres=gpu:nvidia_rtx_a6000:3
#SBATCH --cpus-per-task=5
#SBATCH --mem=70G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=3 aloe_scripts/train.py \
  --task clevrer_vqa \
  --params slotformer/clevrer_vqa/configs/aloe_clevrer_params.py \
  --fp16 --ddp --cudnn