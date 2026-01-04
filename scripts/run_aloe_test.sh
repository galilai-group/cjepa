#!/bin/bash
#SBATCH --job-name=dinowmreg
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=6
#SBATCH --gres=gpu:nvidia_rtx_a6000:6
#SBATCH --cpus-per-task=5
#SBATCH --mem=300G
#SBATCH --output=aloe-%j.out
#SBATCH --error=aloe-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

python slotformer/clevrer_vqa/test_clevrer_vqa.py \
  --params slotformer/clevrer_vqa/configs/aloe_clevrer_param_for_test.py \
  --weight '/cs/data/people/hnam16/aloe_checkpoint/aloe_clevrer_params_20251231_124512/clevrer_slots_step=100000_weight01_lr1e-4_clevrer/epoch/model_400_salr1e-4_w01_aloelr_1e-3.pth' \
    # --weight '/cs/data/people/hnam16/aloe_checkpoint/aloe_clevrer_params_20251229_212441/clevrer_slots_step=100000_weight05_lr5e-4_clevrer/epoch/model_400_salr5e-4_w05_aloelr_1e-3.pth'
