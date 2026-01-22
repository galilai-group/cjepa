#!/bin/bash
#SBATCH --job-name=aloe
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:geforce_gtx_2080_ti:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --output=aloe-%j.out
#SBATCH --error=aloe-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=3 --master-port=29504 aloe_scripts/train.py \
  --task clevrer_vqa \
  --params slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
  --exp_name exp215 \
  --out_dir /cs/data/people/hnam16/aloe_checkpoint \
  --slot_root_override '/cs/data/people/hnam16/data/modified_extraction/rollout_clevrer_slots_step=100000_weight03_lr1e-4_clevrer_lr0.0005_mask2_ratio0.14.pkl' \
  --fp16 --cudnn \
  # --ddp