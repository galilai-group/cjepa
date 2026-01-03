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
# becareful if you have special characters in the path like '=': Need escape it with '\'
export SLOTPATH="/cs/data/people/hnam16/data/modified_extraction/clevrer_slots_step\=100000_weight03_lr1e-4_clevrer.pkl"

# torchrun --nproc_per_node=3 --master-port=29501 \

# python train/train_causalwm_from_clevrer_slot.py \
#     cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
#     output_model_name="51p" \
#     dataset_name="clevrer" \
#     num_workers=8 \
#     batch_size=256 \
#     trainer.max_epochs=30 \
#     num_masked_slots=4 \
#     predictor_lr=5e-4 \
#     dinowm.history_size=6 \
#     dinowm.num_preds=10 \
#     frameskip=2 \
#     videosaur.NUM_SLOTS=7 \
#     videosaur.SLOT_DIM=128 \
#     predictor.heads=16 \
#     embedding_dir=$SLOTPATH \

python train/train_causalwm_from_clevrer_slot.py \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
    output_model_name="19p" \
    dataset_name="clevrer" \
    num_workers=8 \
    batch_size=256 \
    trainer.max_epochs=30 \
    num_masked_slots=0 \
    predictor_lr=5e-4 \
    dinowm.history_size=6 \
    dinowm.num_preds=10 \
    frameskip=2 \
    videosaur.NUM_SLOTS=7 \
    videosaur.SLOT_DIM=128 \
    predictor.heads=16 \
    embedding_dir=$SLOTPATH \
    rollout.rollout_only=true \
    rollout.rollout_checkpoint='/cs/data/people/hnam16/.stable_worldmodel/19p_final_predictor.ckpt'


