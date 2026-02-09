#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_titan_rtx:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --output=aloe-%j.out
#SBATCH --error=aloe-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

python slotformer/clevrer_vqa/test_clevrer_vqa.py \
  --params slotformer/clevrer_vqa/configs/aloe_clevrer_param_for_test.py \
  --weight '/cs/data/people/hnam16/aloe_checkpoint/226_model_400.pth' \
  --slots_root_override '/cs/data/people/hnam16/data/modified_extraction/rollout_clevrer_savi_reproduced_ocvp.pkl' \
  --validate