export PYTHONPATH=$(pwd)

SEEDS=(0 1 2)
# POLICY is the name of the C-JEPA ckpt, but after removing "_object.ckpt"
# For example, if the ckpt name is "pusht_videosaur_1_object.ckpt", then POLICY should be "pusht_videosaur_1"
# This is because of the planning script by stable-worldmodel library.
POLICY="pusht_videosaur_1"

for SEED in "${SEEDS[@]}"; do
    echo "Running experiment with seed=${SEED}"
    HYDRA_FULL_ERROR=1 
    STABLEWM_HOME='~/.stable_worldmodel/'
    python src/plan/run.py \
        seed=${SEED} \
        policy=${POLICY} \
        world.history_size=1 \
        world.frame_skip=1 \
        plan_config.horizon=5 \
        plan_config.receding_horizon=5 \
        plan_config.action_block=5 \
        eval.eval_budget=50 \
        output.filename=planning_${POLICY}_seed_${SEED}.txt \
        eval.dataset_name=pusht_expert_train \
        eval.goal_offset_steps=25
done

echo "All experiments finished on: $(date)"
