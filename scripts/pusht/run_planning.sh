#!/bin/bash
#SBATCH --job-name=dinowm-planning
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do
    echo "Running experiment with seed=${SEED}"

    HYDRA_FULL_ERROR=1 
    STABLEWM_HOME=~/vast/.stable_worldmodel/ 
    python plan/run.py 
        seed=${SEED} 
        policy=pusht_expert_world_model_epoch_10 
        plan_config.horizon=5 
        eval.eval_budget=50 
        output.filename=cjepa_pusht_seed_${SEED}.txt
done

echo "All experiments finished on: $(date)"

