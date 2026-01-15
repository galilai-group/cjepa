#!/bin/bash
#SBATCH --job-name=extract_videosaur
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --output=sa-%j.out
#SBATCH --error=sa-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

python slotformer/base_slots/extract_videosaur.py \
    --weight="logs/videosaur_pusht/2026-01-08-20-52-41_pusht_dinov2/checkpoints/pushtnoise_videosaur_lr1e-4_w03_step=100000.ckpt" \
    --data_root="/cs/data/people/hnam16/.stable_worldmodel" \
    --save_path="/cs/data/people/hnam16/data/modified_extraction/pusht_expert_slots" \
    --videosaur_config="videosaur/configs/videosaur/pusht_dinov2_hf.yml" \
    --dataset="pusht_expert" \
    --params="slotformer/aloe_pusht_params.py" \