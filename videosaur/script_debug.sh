#!/bin/bash
# #SBATCH --nodelist=gpu2201
#SBATCH --job-name=test
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=100G

#SBATCH -J MySerialJob
# Specify an output file
# %j is a special variable that is replaced by the JobID when the job starts
#SBATCH -o MySerialJob-%j.out
#SBATCH -e MySerialJob-%j.err

poetry run python -m videosaur.train configs/videosaur/clevrer_dinov2_debug.yml