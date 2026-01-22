#!/bin/bash
#SBATCH --job-name=slotformer
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

torchrun --nproc_per_node=2 --master-port=29501 \
    aloe_scripts/train.py --task video_prediction \
    --exp_name "196_slotformer" \
    --out_dir /cs/data/people/hnam16/aloe_checkpoint \
    --params slotformer/video_prediction/configs/slotformer_clevrer_params.py \
    --fp16 --ddp --cudnn

# python aloe_scripts/train.py --task video_prediction \
#     --params slotformer/video_prediction/configs/slotformer_clevrer_params.py \
#     --fp16 --cudnn