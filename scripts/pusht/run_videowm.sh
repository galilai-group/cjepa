#!/bin/bash
#SBATCH --job-name=dinowm
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=4
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

# python -m pdb train/train_videowm.py \

torchrun --nproc_per_node=4 --master-port=29503 \
    train/train_videowm.py \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
    output_model_name="193" \
    dataset_name="pusht_expert" \
    num_workers=8 \
    batch_size=64 \
    trainer.max_epochs=30 \
    predictor_lr=5e-4 \
    proprio_encoder_lr=5e-4 \
    action_encoder_lr=5e-4 \
    dinowm.history_size=3 \
    dinowm.num_preds=1 \
    frameskip=5 \

