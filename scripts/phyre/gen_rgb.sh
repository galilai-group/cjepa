#!/bin/bash
#SBATCH --job-name=gen_video
#SBATCH --time=5-00:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --output=extract-%j.out
#SBATCH --error=extract-%j.err


echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

python dataset/PYHRE/convert_rgb.py