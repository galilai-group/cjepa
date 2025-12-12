#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=3
#SBATCH --gres=gpu:nvidia_rtx_a6000:3
#SBATCH --cpus-per-task=9
#SBATCH --mem=100G
#SBATCH --output=causalwm-%j.out
#SBATCH --error=causalwm-%j.err


echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

torchrun --nproc_per_node=3 --master-port=29502 train_causalwm.py num_workers=8 batch_size=64 causal_mask_predict=false

# torchrun --nproc_per_node=4 train_causalwm.py num_workers=8 batch_size=64 causal_mask_predict=true