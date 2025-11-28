#!/bin/bash
# #SBATCH --nodelist=gpu2201
#SBATCH --job-name=debug
#SBATCH --time=48:00:00
#SBATCH --partition=batch
# #SBATCH --gres=gpu:1           
#SBATCH --cpus-per-task=20  
#SBATCH --ntasks-per-node=1        
#SBATCH --mem=90G

#SBATCH -J MySerialJob
# Specify an output file
# %j is a special variable that is replaced by the JobID when the job starts
#SBATCH -o MySerialJob-%j.out
#SBATCH -e MySerialJob-%j.err

python train_ocwm.py 