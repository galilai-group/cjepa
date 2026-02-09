#!/bin/bash
#SBATCH --job-name=119p_from_pixel
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=3
#SBATCH --gres=gpu:nvidia_rtx_a6000:3
#SBATCH --cpus-per-task=9
#SBATCH --mem=100G
#SBATCH --output=pusht-%j.out
#SBATCH --error=pusht-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)
# becareful if you have special characters in the path like '=': Need escape it with '\'
# export CKPT_PATH="/cs/data/people/hnam16/.stable_worldmodel/artifacts/oc-checkpoints/pushtnoise_videosaur_lr1e-4_w03_step\=100000.ckpt"

export OC_CKPT="/cs/data/people/hnam16/.stable_worldmodel/artifacts/oc-checkpoints/pusht_savi_101_24.pth"

# torchrun --nproc_per_node=3 --master-port=29502 \
python -m pdb train/train_causalwm_savi.py \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
    output_model_name="119p_from_pixel" \
    dataset_name="pusht_expert" \
    num_workers=8 \
    batch_size=64 \
    trainer.max_epochs=30 \
    num_masked_slots=0 \
    predictor_lr=5e-4 \
    proprio_encoder_lr=5e-4 \
    action_encoder_lr=5e-4 \
    dinowm.history_size=3 \
    dinowm.proprio_embed_dim=12 \
    dinowm.action_embed_dim=10 \
    predictor.heads=15 \
    dinowm.num_preds=1 \
    frameskip=5 \
    image_size=64 \
    savi.weight=$OC_CKPT

