#!/bin/bash
#SBATCH --job-name=aloe
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=3
#SBATCH --gres=gpu:nvidia_rtx_a6000:3
#SBATCH --cpus-per-task=9
#SBATCH --mem=100G
#SBATCH --output=aloe-%j.out
#SBATCH --error=aloe-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=3 --master-port=29504 aloe_scripts/train.py \
  --task clevrer_vqa \
  --params slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
  --exp_name exp24 \
  --out_dir /cs/data/people/hnam16/aloe_checkpoint \
  --fp16 --ddp --cudnn