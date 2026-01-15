#!/bin/bash
#SBATCH --job-name=clevrer-cjepa
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=50G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

WEIGHT="/cs/data/people/hnam16/savi_pretrained/ex101_20260108_023722_LR0.0001/savi/epoch/model_24.pth"

python slotformer/base_slots/extract_slot_pusht.py \
    --params slotformer/base_slots/configs/savi_pusht_ind_params_mlp.py \
    --weight $WEIGHT \
    --save_path /cs/data/people/hnam16/data/modified_extraction/pusht_savi_101.pkl



# python -m pdb slotformer/base_slots/extract_slots.py \
#     --params slotformer/base_slots/configs/stosavi_clevrer_params.py \
#     --weight "/cs/data/people/hnam16/savi_pretrained/clevrer_savi_reproduce_20260109_102641_LR0.0001/savi/epoch/model_8.pth" \
#     --save_path /cs/data/people/hnam16/data/modified_extraction/clevrer_savi_reproduced.pkl