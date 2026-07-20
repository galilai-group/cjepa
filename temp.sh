#!/bin/bash
#SBATCH --job-name=dino2
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=bio_ai
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=2-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclude=bamgpu02,bamgpu07,bamgpu17,bamgpu20

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

if (( $# )); then
    python eval.py "$@"
else
    python eval.py policy=cjepa/pusht_m1 --download
fi
