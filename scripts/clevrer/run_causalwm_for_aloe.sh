#!/bin/bash
#SBATCH --job-name=dinowmreg
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=3
#SBATCH --gres=gpu:nvidia_rtx_a6000:3
#SBATCH --cpus-per-task=11
#SBATCH --mem=150G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=3 --master-port=29501 \
    train/train_causalwm.py \
    output_model_name="world_model_causal_nomask_aloe" \
    dataset_name="clevrer_train" \
    num_workers=10 \
    batch_size=64 \
    trainer.max_epochs=30 \
    num_masked_slots=0 \
    dinowm.history_size=6 \
    dinowm.num_preds=10 \
    frameskip=2 \
    model.load_weights="/cs/data/people/hnam16/.stable_worldmodel/artifacts/oc-checkpoints/oc_ckpt.ckpt"