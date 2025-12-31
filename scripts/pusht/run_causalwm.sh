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
export CKPT_PATH="/cs/data/people/hnam16/.stable_worldmodel/artifacts/oc-checkpoints/step\=100000_weight05_lr5e-4_pusht.ckpt"

torchrun --nproc_per_node=6 --master-port=29502 \
    train/train_causalwm.py \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
    output_model_name="pusht_causal_wm" \
    dataset_name="pusht_expert" \
    num_workers=4 \
    batch_size=128 \
    trainer.max_epochs=10 \
    num_masked_slots=2 \
    model.load_weights=${CKPT_PATH} \
    predictor_lr=5e-4 \
    proprio_encoder_lr=5e-4 \
    action_encoder_lr=5e-4 \
    dinowm.history_size=3 \
    dinowm.num_preds=1 \
    frameskip=5 \
    videosaur.NUM_SLOTS=4 \
    videosaur.SLOT_DIM=64 \
    predictor.heads=12

