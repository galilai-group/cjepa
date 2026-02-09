#!/bin/bash
#SBATCH --job-name=dinowm
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=4
#SBATCH --gres=gpu:nvidia_titan_rtx:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=4 --master-port=29503 \
    train/train_videowm.py \
    cache_dir="/users/hnam16/scratch/.stable_worldmodel" \
    output_model_name="clevrer_world_model" \
    dataset_name="clevrer" \
    num_workers=4 \
    batch_size=64 \
    trainer.max_epochs=30 \
    predictor_lr=5e-4 \
    proprio_encoder_lr=5e-4 \
    action_encoder_lr=5e-4 \
    dinowm.history_size=6 \
    dinowm.num_preds=10 \
    frameskip=2
