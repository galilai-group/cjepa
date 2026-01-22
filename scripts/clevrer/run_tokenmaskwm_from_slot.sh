#!/bin/bash
#SBATCH --job-name=206p
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_titan_rtx:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)
# becareful if you have special characters in the path like '=': Need escape it with '\'
export SLOTPATH="/cs/data/people/hnam16/data/modified_extraction/clevrer_slots_step\=100000_weight03_lr1e-4_clevrer.pkl"
# export SLOTPATH="/cs/data/people/hnam16/.stable_worldmodel/artifacts/oc-checkpoints/savi_slots.pkl"

# torchrun --nproc_per_node=3 --master-port=29501 \

python train/train_tokenmaskwm_from_clevrer_slot.py \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
    output_model_name="206p" \
    dataset_name="clevrer" \
    num_workers=4 \
    batch_size=256 \
    trainer.max_epochs=30 \
    predictor_lr=5e-4 \
    dinowm.history_size=6 \
    dinowm.num_preds=10 \
    frameskip=2 \
    videosaur.NUM_SLOTS=7 \
    videosaur.SLOT_DIM=128 \
    predictor.heads=16 \
    embedding_dir=$SLOTPATH \
    mask_ratio=0.56
    # load_checkpoint=true \
    # ckpt_path="/cs/data/people/hnam16/.stable_worldmodel/205p_epoch_10_predictor.ckpt"



