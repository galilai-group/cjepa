#!/bin/bash
#SBATCH --job-name=clevrer-cjepa
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=50G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)
export CKPTPATH='/cs/data/people/hnam16/.stable_worldmodel/119p_final_predictor.ckpt'
export SLOTPATH="/cs/data/people/hnam16/data/modified_extraction/clevrer_savi_reproduced.pkl"


python train/train_causalwm_from_clevrer_slot.py \
    rollout.rollout_only=true \
    rollout.rollout_checkpoint=$CKPTPATH \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
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
    predictor_lr=5e-4 \
    num_masked_slots=0



