#!/bin/bash
#SBATCH --job-name=210p
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_titan_rtx:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

# don't forget to escape special characters like '=' with '\'
export SLOTPATH="/cs/data/people/hnam16/data/modified_extraction/pusht_expert_slots_videosaur_172.pkl"
# export SLOTPATH="/cs/data/people/hnam16/data/modified_extraction/pusht_savi_101.pkl"

# this is for saving swm ckpt for smooth planning.. this should be matched with the pusht ckpt used for slot extraction
export CKPT_PATH="/cs/data/people/hnam16/.stable_worldmodel/artifacts/oc-checkpoints/videosaur_172.ckpt"

# torchrun --nproc_per_node=3 --master-port=29501 \

# Caution!! Set output_model_name properly 
python train/train_vjepawm_AP_node_pusht_slot.py \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
    output_model_name="210p" \
    dataset_name="pusht_expert" \
    num_workers=4 \
    batch_size=256 \
    trainer.max_epochs=30 \
    predictor_lr=5e-4 \
    proprio_encoder_lr=5e-4 \
    action_encoder_lr=5e-4 \
    dinowm.history_size=3 \
    dinowm.num_preds=1 \
    dinowm.proprio_embed_dim=128 \
    dinowm.action_embed_dim=128 \
    frameskip=5 \
    videosaur.NUM_SLOTS=4 \
    videosaur.SLOT_DIM=128 \
    predictor.heads=16 \
    embedding_dir=${SLOTPATH} \
    model.load_weights=${CKPT_PATH} \
    use_hungarian_matching=false \
    mask_ratio=0.5 \
    num_masked_blocks=2




