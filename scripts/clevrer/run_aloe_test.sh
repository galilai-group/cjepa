#!/bin/bash
#SBATCH --job-name=dinowmreg
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=300G
#SBATCH --output=aloe-%j.out
#SBATCH --error=aloe-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

python slotformer/clevrer_vqa/test_clevrer_vqa.py \
  --params slotformer/clevrer_vqa/configs/aloe_clevrer_param_for_test.py \
  --weight '/cs/data/people/hnam16/aloe_checkpoint/exp80_model_400_savi_mask2.pth'