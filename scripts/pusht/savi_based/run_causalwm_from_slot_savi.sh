#!/bin/bash
#SBATCH --job-name=cjepa
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=30G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)
# becareful if you have special characters in the path like '=': Need escape it with '\'
# export SLOTPATH="/cs/data/people/hnam16/data/modified_extraction/clevrer_slots_step\=100000_weight03_lr1e-4_clevrer.pkl"
export SLOTPATH="/cs/data/people/hnam16/data/modified_extraction/pusht_savi_101.pkl"

# this is for saving swm ckpt for smooth planning.. this should be matched with the pusht ckpt used for slot extraction
export OC_CKPT="/cs/data/people/hnam16/.stable_worldmodel/artifacts/oc-checkpoints/pusht_savi_101_24.pth"

# torchrun --nproc_per_node=3 --master-port=29501 \

# predictor head should be 12 because embedding dimension (including action, proprio) should be devisible with num head

python train/train_causalwm_from_pusht_slot_savi.py \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
    output_model_name="139p" \
    dataset_name="pusht_expert" \
    num_workers=8 \
    batch_size=256 \
    trainer.max_epochs=30 \
    num_masked_slots=2 \
    predictor_lr=5e-4 \
    proprio_encoder_lr=5e-4 \
    action_encoder_lr=5e-4 \
    dinowm.history_size=5 \
    dinowm.num_preds=3 \
    dinowm.proprio_embed_dim=12 \
    dinowm.action_embed_dim=10 \
    frameskip=3 \
    image_size=64 \
    savi.weight=$OC_CKPT \
    predictor.heads=15 \
    embedding_dir=${SLOTPATH} \





