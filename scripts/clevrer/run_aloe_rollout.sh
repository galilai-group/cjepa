#!/bin/bash
#SBATCH --job-name=228
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=50G
#SBATCH --output=aloe-%j.out
#SBATCH --error=aloe-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

# torchrun --nproc_per_node=2 --master-port=29501 \
python src/aloe_train.py \
  --task clevrer_vqa \
  --params slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
  --exp_name 228 \
  --out_dir /cs/data/people/hnam16/aloe_checkpoint \
  --slot_root_override '/cs/data/people/hnam16/data/modified_extraction/rollout_clevrer_slots_step=100000_weight03_lr1e-4_clevrer_lr0.0005_exp202p.pkl' \
  --fp16 --cudnn \
  # --ddp
