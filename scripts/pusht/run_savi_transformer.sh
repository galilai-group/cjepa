#!/bin/bash
#SBATCH --job-name=90savi
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=2
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --cpus-per-task=9
#SBATCH --mem=100G
#SBATCH --output=aloe-%j.out
#SBATCH --error=aloe-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

# PRED=mlp  # 'mlp' or 'transformer'


torchrun --nproc_per_node=2 --master-port=29504 aloe_scripts/train.py \
  --task base_slots \
  --params slotformer/base_slots/configs/savi_pusht_params_transformer.py \
  --exp_name exp90 \
  --out_dir /cs/data/people/hnam16/savi_pretrained \
  --fp16 --ddp --cudnn


