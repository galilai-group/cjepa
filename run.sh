#!/bin/bash
#SBATCH --job-name=job
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_titan_rtx:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=aloe-%j.out
#SBATCH --error=aloe-%j.err

# Avoid CPU oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python - << 'EOF'
import time
import torch

assert torch.cuda.is_available(), "CUDA not available"

while True:
    # Minimal but real GPU workload
    x = torch.randn(2048, 2048, device="cuda")
    _ = x @ x
    torch.cuda.synchronize()

    # Keep overhead low
    time.sleep(5)
EOF