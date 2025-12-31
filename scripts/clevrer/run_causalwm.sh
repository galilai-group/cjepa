#!/bin/bash
#SBATCH --job-name=dinowmreg
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=6
#SBATCH --gres=gpu:nvidia_rtx_a6000:6
#SBATCH --cpus-per-task=5
#SBATCH --mem=200G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)
# becareful if you have special characters in the path like '=': Need escape it with '\'
export CKPT_PATH="/cs/data/people/hnam16/.stable_worldmodel/artifacts/oc-checkpoints/step\=100000_weight05_lr1e-4_clevrer.ckpt"

torchrun --nproc_per_node=6 --master-port=29502 \
    train/train_causalwm.py \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
    output_model_name="clevrer_causal_wm" \
    dataset_name="clevrer" \
    num_workers=4 \
    batch_size=32 \
    trainer.max_epochs=30 \
    num_masked_slots=4 \
    model.load_weights=${CKPT_PATH} \
    predictor_lr=5e-4 \
    proprio_encoder_lr=5e-4 \
    action_encoder_lr=5e-4 \
    dinowm.history_size=6 \
    dinowm.num_preds=10 \
    frameskip=2 \
    videosaur.NUM_SLOTS=7 \
    videosaur.SLOT_DIM=128 

