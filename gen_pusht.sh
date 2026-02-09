#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --partition=gpus            
#SBATCH --gres=gpu:nvidia_rtx_a6000:1                  
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=7-00:00:00             
#SBATCH --output=gen.out
#SBATCH --error=gen.err


python dataset/pusht/pusht_all_moving_videogen.py --num_videos 10000 --jpeg_quality 60