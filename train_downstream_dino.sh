#!/bin/bash
#SBATCH --job-name=dino2
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=bio_ai
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=2-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclude=bamgpu02,bamgpu07,bamgpu17,bamgpu20

# cheetah-run
# VQPT="/grid/klindt/home/nam/ckpt/vqvae_runs/vqvae_K_16_28005_20260603_142448/checkpoints/motion_factor_vqvae_vqvae_K_16_28005_20260603_142448_final.pt"
# VQPT="/grid/klindt/home/nam/ckpt/vqvae_runs/vqvae_K_32_28006_20260603_142538/checkpoints/motion_factor_vqvae_vqvae_K_32_28006_20260603_142538_final.pt"
# VQPT="/grid/klindt/home/nam/ckpt/vqvae_runs/vqvae_K_64_28007_20260603_142543/checkpoints/motion_factor_vqvae_vqvae_K_64_28007_20260603_142543_final.pt"
# VQPT="/grid/klindt/home/nam/ckpt/vqvae_runs/vqvae_K_128_28008_20260603_142550/checkpoints/motion_factor_vqvae_vqvae_K_128_28008_20260603_142550_final.pt"

# walker-run
# VQPT="/grid/klindt/home/nam/ckpt/vqvae_runs/vqvae_K_16_28009_20260603_142616/checkpoints/motion_factor_vqvae_vqvae_K_16_28009_20260603_142616_final.pt"
VQPT="/grid/klindt/home/nam/ckpt/vqvae_runs/vqvae_K_32_28010_20260603_142621/checkpoints/motion_factor_vqvae_vqvae_K_32_28010_20260603_142621_final.pt"
# VQPT="/grid/klindt/home/nam/ckpt/vqvae_runs/vqvae_K_64_2514166_20260612_204551/checkpoints/motion_factor_vqvae_vqvae_K_64_2514166_20260612_204551_final.pt"
# VQPT="/grid/klindt/home/nam/ckpt/vqvae_runs/vqvae_K_128_28012_20260603_142635/checkpoints/motion_factor_vqvae_vqvae_K_128_28012_20260603_142635_final.pt"



# TASK="cheetah-run"
# DINO_PT="/grid/klindt/home/nam/ckpt/dinolam_runs/cheetah_dinolam_global_only_job28311_20260611_105619/checkpoints/dinolam_final_step020000.pt"  # cheetah 16 
# DINO_PT="/grid/klindt/home/nam/ckpt/dinolam_runs/cheetah_dinolam_global_only_job28312_20260611_105920/checkpoints/dinolam_final_step020000.pt"  # cheetah 32
# DINO_PT="/grid/klindt/home/nam/ckpt/dinolam_runs/cheetah_dinolam_global_only_job2567602_20260619_233030/checkpoints/dinolam_final_step020000.pt"  # cheetah 64
# DINO_PT="/grid/klindt/home/nam/ckpt/dinolam_runs/cheetah_dinolam_global_only_job28320_20260611_195029/checkpoints/dinolam_final_step020000.pt"  # cheetah 128 


TASK="walker-run"
# DINO_PT="/grid/klindt/home/nam/ckpt/dinolam_runs/walker_dinolam_global_only_job28321_20260611_195057/checkpoints/dinolam_final_step020000.pt"  # walker 16
DINO_PT="/grid/klindt/home/nam/ckpt/dinolam_runs/walker_dinolam_global_only_job2514064_20260612_203357/checkpoints/dinolam_final_step020000.pt"  # walker 32 
# DINO_PT="/grid/klindt/home/nam/ckpt/dinolam_runs/walker_dinolam_global_only_job2519990_20260613_112546/checkpoints/dinolam_final_step020000.pt"  # walker 64 
# DINO_PT="/grid/klindt/home/nam/ckpt/dinolam_runs/walker_dinolam_global_only_job2514066_20260612_203439/checkpoints/dinolam_final_step020000.pt"  # walker 128 






DOWNSTREAM_ROOT="${DOWNSTREAM_ROOT:-/grid/klindt/home/nam/ckpt/downstream_dino2}"
DCS_BACKGROUNDS_PATH="${DCS_BACKGROUNDS_PATH:-/grid/klindt/home/nam/data/DAVIS/JPEGImages/480p}"



export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl


DINO_RUN_DIR="$(basename "$(dirname "$(dirname "$DINO_PT")")")"
VQDIR="${VQPT#*/vqvae_runs/}"
VQDIR="${VQDIR%%/*}"
RUN_PREFIX="${RUN_PREFIX:-${VQDIR}_dino_${DINO_RUN_DIR}}"

BC_EVAL_SEED="${BC_EVAL_SEED:-1}"
ACT_DECODER_EVAL_SEED="${ACT_DECODER_EVAL_SEED:-1}"
ACTION_DECODER_LATENTACTION_TYPE="${ACTION_DECODER_LATENTACTION_TYPE:-pred}"
AUTO_RESUME_DOWNSTREAM="${AUTO_RESUME_DOWNSTREAM:-true}"
BC_RESUME_RUN_PREFIX="${BC_RESUME_RUN_PREFIX:-}"

find_bc_checkpoint() {
  local run_dir="$1"
  local epoch_ckpt
  if [[ -f "${run_dir}/bc-final.pt" ]]; then
    echo "${run_dir}/bc-final.pt"
    return 0
  fi
  epoch_ckpt="$(find "$run_dir" -maxdepth 1 -type f -name 'bc-epoch_*.pt' -print 2>/dev/null | sort -V | tail -n 1)"
  if [[ -n "$epoch_ckpt" ]]; then
    echo "$epoch_ckpt"
    return 0
  fi
  if [[ -f "${run_dir}/bc-latest.pt" ]]; then
    echo "${run_dir}/bc-latest.pt"
    return 0
  fi
  return 1
}

find_action_decoder_checkpoint() {
  local run_dir="$1"
  local step_ckpt
  if [[ -f "${run_dir}/act-decoder-final.pt" ]]; then
    echo "${run_dir}/act-decoder-final.pt"
    return 0
  fi
  step_ckpt="$(find "$run_dir" -maxdepth 1 -type f -name 'act-decoder-step_*.pt' -print 2>/dev/null | sort -V | tail -n 1)"
  if [[ -n "$step_ckpt" ]]; then
    echo "$step_ckpt"
    return 0
  fi
  if [[ -f "${run_dir}/act-decoder-latest.pt" ]]; then
    echo "${run_dir}/act-decoder-latest.pt"
    return 0
  fi
  return 1
}

echo "DINO-LAM checkpoint: $DINO_PT"
echo "VQ-VAE checkpoint: $VQPT"
echo "Task: $TASK"
echo "Downstream root: $DOWNSTREAM_ROOT"
echo "Action decoder latent action type: $ACTION_DECODER_LATENTACTION_TYPE"
echo "Auto resume downstream: $AUTO_RESUME_DOWNSTREAM"

for TRAIN_SEED in 0 1 2; do

  DOWNSTREAM_RUN_NAME="${RUN_PREFIX}_seed${TRAIN_SEED}"
  DOWNSTREAM_OUTPUT_DIR="${DOWNSTREAM_ROOT}/${DOWNSTREAM_RUN_NAME}"
  BC_ARGS=(--train_bc=true)
  ACT_DECODER_ARGS=()
  if [[ -n "$BC_RESUME_RUN_PREFIX" ]]; then
    DOWNSTREAM_OUTPUT_DIR="${BC_RESUME_RUN_PREFIX}_seed${TRAIN_SEED}"
    DOWNSTREAM_RUN_NAME="$(basename "$DOWNSTREAM_OUTPUT_DIR")"
  fi

  echo "Resolved downstream run name: ${DOWNSTREAM_RUN_NAME}"
  echo "Resolved downstream output dir: ${DOWNSTREAM_OUTPUT_DIR}"

  if [[ "$AUTO_RESUME_DOWNSTREAM" == "true" ]]; then
    if BC_RESUME_PATH="$(find_bc_checkpoint "$DOWNSTREAM_OUTPUT_DIR")"; then
      BC_ARGS=(--train_bc=false --bc_resume_path="$BC_RESUME_PATH" --no_auto_bc_resume)
      echo "Found BC checkpoint for seed ${TRAIN_SEED}: ${BC_RESUME_PATH}"
      echo "Will skip BC and start from action decoder for seed ${TRAIN_SEED}."
    else
      echo "No BC checkpoint found for seed ${TRAIN_SEED}; will train BC from scratch."
    fi

    if ACT_DECODER_RESUME_PATH="$(find_action_decoder_checkpoint "$DOWNSTREAM_OUTPUT_DIR")"; then
      ACT_DECODER_ARGS=(--act_decoder_resume_path="$ACT_DECODER_RESUME_PATH")
      echo "Found action decoder checkpoint for seed ${TRAIN_SEED}: ${ACT_DECODER_RESUME_PATH}"
      echo "Will resume action decoder from that checkpoint."
    else
      echo "No action decoder checkpoint found for seed ${TRAIN_SEED}; action decoder will start from scratch."
    fi
  elif [[ -n "$BC_RESUME_RUN_PREFIX" ]]; then
    BC_RESUME_PATH="${DOWNSTREAM_OUTPUT_DIR}/bc-epoch_10.pt"
    BC_ARGS=(--train_bc=false --bc_resume_path="$BC_RESUME_PATH" --no_auto_bc_resume)
    echo "Auto resume disabled; using BC_RESUME_RUN_PREFIX checkpoint: ${BC_RESUME_PATH}"
  fi

  python factorized_lam/train_downstream.py \
    --lam_type=dino \
    --dinolam_checkpoint_path="$DINO_PT" \
    --vqvae_checkpoint_path="$VQPT" \
    --data_path="/grid/klindt/home/nam/data/dcs/${TASK}.hdf5" \
    --output_dir="${DOWNSTREAM_OUTPUT_DIR}" \
    --seed="${TRAIN_SEED}" \
    --bc_eval_seed="${BC_EVAL_SEED}" \
    --act_decoder_eval_seed="${ACT_DECODER_EVAL_SEED}" \
    --action_decoder_latentaction_type="${ACTION_DECODER_LATENTACTION_TYPE}" \
    --bc_dcs_backgrounds_path="${DCS_BACKGROUNDS_PATH}" \
    --act_decoder_dcs_backgrounds_path="${DCS_BACKGROUNDS_PATH}" \
    "${BC_ARGS[@]}" \
    "${ACT_DECODER_ARGS[@]}" \
    --wandb_project="lam" \
    --wandb_run_name="${DOWNSTREAM_RUN_NAME}" \
    --wandb_tags "lam_type_dino" "training_seed_${TRAIN_SEED}" "bc_eval_seed_${BC_EVAL_SEED}" "act_decoder_eval_seed_${ACT_DECODER_EVAL_SEED}"
done
