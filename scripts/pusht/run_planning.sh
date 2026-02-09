#!/bin/bash
#SBATCH --job-name=planning
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=4
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

SEEDS=(0 1 2)
EXPNUM=263p

for SEED in "${SEEDS[@]}"; do
    echo "Running experiment with seed=${SEED}"


    HYDRA_FULL_ERROR=1 
    STABLEWM_HOME=/cs/data/people/hnam16/.stable_worldmodel/ 
    python src/plan/run.py \
        seed=${SEED} \
        policy=${EXPNUM}_epoch_20 \
        world.history_size=1 \
        world.frame_skip=1 \
        plan_config.horizon=5 \
        plan_config.receding_horizon=5 \
        plan_config.action_block=5 \
        eval.eval_budget=50 \
        output.filename=cjepa_pusht_from_${EXPNUM}_seed_${SEED}.txt \
        eval.dataset_name=pusht_expert_train \
        eval.goal_offset_steps=25
    # python plan/run.py \
    #     seed=${SEED} \
    #     policy=${EXPNUM}_epoch_30 \
    #     world.history_size=1 \
    #     world.frame_skip=1 \
    #     plan_config.horizon=5 \
    #     plan_config.receding_horizon=5 \
    #     plan_config.action_block=5 \
    #     eval.eval_budget=50 \
    #     output.filename=cjepa_pusht_from_${EXPNUM}_seed_${SEED}_what_wrong.txt \
    #     eval.dataset_name=pusht_expert_val
done

echo "All experiments finished on: $(date)"

trap : TERM INT
sleep infinity