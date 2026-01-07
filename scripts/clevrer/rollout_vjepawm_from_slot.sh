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
export CKPTPATH='/cs/data/people/hnam16/.stable_worldmodel/95p_final_predictor.ckpt'
export SLOTPATH="/cs/data/people/hnam16/data/modified_extraction/clevrer_slots_step\=100000_weight03_lr1e-4_clevrer.pkl"

# @quentin Change mask_ratio to 0.25 for 91p, 0.5 for 94p, 0.75 for 95p in the below command accordingly!
# @quentin You will the same SLOTHPATH, and you used the sameone before! 

python train/train_vjepawm_from_clevrer_slot.py \
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
    num_masked_blocks=2 \
    mask_ratio=0.75



