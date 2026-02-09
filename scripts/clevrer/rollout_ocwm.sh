#!/bin/bash
#SBATCH --job-name=ocwm-rollout
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=30G
#SBATCH --output=rollout-ocwm-%j.out
#SBATCH --error=rollout-ocwm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)
# Pre-extracted slots path (128 frames)
export SLOTPATH="/cs/data/people/hnam16/data/modified_extraction/clevrer_slots_step\=100000_weight03_lr1e-4_clevrer.pkl"

# Trained OCWM predictor checkpoint (adjust path to your trained model)
export CKPT_PATH="/cs/data/people/hnam16/.stable_worldmodel/202p_final_predictor.ckpt"

python train/train_ocwm_from_clevrer_slot.py \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
    output_model_name="202p" \
    dataset_name="clevrer" \
    num_workers=8 \
    batch_size=256 \
    dinowm.history_size=6 \
    dinowm.num_preds=10 \
    frameskip=2 \
    videosaur.NUM_SLOTS=7 \
    videosaur.SLOT_DIM=128 \
    predictor.heads=16 \
    embedding_dir=$SLOTPATH \
    rollout.rollout_only=true \
    rollout.rollout_checkpoint=$CKPT_PATH \
    rollout.rollout_batch_size=64

echo "Rollout completed at: $(date)"
