#!/bin/bash
#SBATCH --job-name=dinowmreg
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=8
#SBATCH --gres=gpu:geforce_gtx_2080_ti:8
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=8 --master-port=29501 \
    train/train_videowm_reg.py \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
    output_model_name="222" \
    dataset_name="pusht_expert" \
    num_workers=1 \
    batch_size=64 \
    trainer.max_epochs=10 \
    predictor_lr=5e-4 \
    proprio_encoder_lr=5e-4 \
    action_encoder_lr=5e-4 \
    dinowm.history_size=3 \
    dinowm.num_preds=1 \
    frameskip=5
