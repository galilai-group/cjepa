#!/bin/bash
#SBATCH --job-name=100savi
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

# PRED=mlp  # 'mlp' or 'transformer'


torchrun --nproc_per_node=3 --master-port=29509 aloe_scripts/train.py \
  --task base_slots \
  --params slotformer/base_slots/configs/stosavi_pusht_params_transformer.py \
  --exp_name exp100 \
  --out_dir /cs/data/people/hnam16/savi_pretrained \
  --fp16 --ddp --cudnn


