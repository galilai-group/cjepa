#!/bin/bash
#SBATCH --job-name=aloe
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=100G
#SBATCH --output=aloe-%j.out
#SBATCH --error=aloe-%j.err


echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

# torchrun --nproc_per_node=3 --master-port=29504 \
    # aloe_scripts/train.py --task video_prediction \
    # --params slotformer/video_prediction/configs/slotformer_clevrer_params.py \
    # --fp16 --ddp --cudnn

python aloe_scripts/train.py --task video_prediction \
    --params slotformer/video_prediction/configs/slotformer_clevrer_params.py \
    --fp16 --cudnn