#!/bin/bash
#SBATCH --job-name=aloe
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=6
#SBATCH --gres=gpu:nvidia_rtx_a6000:6
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --output=aloe-%j.out
#SBATCH --error=aloe-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=6 aloe_scripts/train.py \
  --task clevrer_vqa \
  --params slotformer/clevrer_vqa/configs/aloe_clevrer_params.py \
  --fp16 --ddp --cudnn