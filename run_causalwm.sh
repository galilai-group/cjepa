#!/bin/bash
#SBATCH --job-name=dinowmreg
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=3
#SBATCH --gres=gpu:nvidia_rtx_a6000:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=3 --master-port=29502 \
    train/train_causalwm.py \
    output_model_name="world_model_causal_4mask" \
    dataset_name="clevrer_train" \
    num_workers=10 \
    batch_size=64 \
    trainer.max_epochs=30 \
    num_masked_slots=4 \
    model.load_weights="/cs/data/people/hnam16/.stable_worldmodel/artifacts/oc-checkpoints/oc_ckpt.ckpt"