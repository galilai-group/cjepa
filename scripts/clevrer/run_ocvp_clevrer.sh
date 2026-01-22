#!/bin/bash
#SBATCH --job-name=ocvp_noimg
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=3
#SBATCH --gres=gpu:nvidia_l40:3
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Start Time: $(date)"
echo "=============================================="



# Create log directory if not exists
mkdir -p logs



# Set PYTHONPATH
export PYTHONPATH="."

# Training configuration overrides (can be customized via command line)
# Default: OCVP-Seq, slot loss only (no image reconstruction)
DEFAULT_ARGS=(
    "ocvp.predictor_type=ocvp_seq"
    "loss.pred_slot_weight=1.0"
    "loss.pred_img_weight=1.0"
    "trainer.max_epochs=30"
    "batch_size=64"
)

# Merge default args with command line args
ARGS="${DEFAULT_ARGS[@]} $@"

# Run training
echo "Running OCVP training..."
echo "Arguments: ${ARGS}"

torchrun --nproc_per_node=3 --master-port=29503  train/train_ocvp_clevrer.py ${ARGS}

echo "=============================================="
echo "End Time: $(date)"
echo "=============================================="
