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


python scripts/extract_slots.py \
  "~/.stable_worldmodel/pusht_expert_train.h5" \
  "~/.stable_worldmodel/pusht_expert_train_slots.h5" \
  --dataset pusht