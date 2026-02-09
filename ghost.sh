#!/bin/bash
#SBATCH --job-name=hold_ghost_gpu
#SBATCH --partition=gpus              # 네 클러스터 파티션 이름
#SBATCH --gres=gpu:geforce_gtx_2080_ti:1                  # 유령 GPU를 잡기 위한 핵심
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=7-00:00:00             
#SBATCH --output=hold_ghost_gpu.out
#SBATCH --error=hold_ghost_gpu.err

echo "Holding ghost GPU on node: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# CUDA를 절대 건드리지 않음
# 무한 루프 (슬럼 time limit까지 유지)
while true; do
    sleep 3600
done