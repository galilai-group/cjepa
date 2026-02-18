echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)
export CKPTPATH='to/path/to/causalwm_checkpoint.ckpt' 
export SLOTPATH="to/path/to/extracted/slot/pkl"


python src/train/train_causalwm_from_clevrer_slot.py \
    rollout.rollout_only=true \
    rollout.rollout_checkpoint=$CKPTPATH \
    cache_dir="~/.stable_worldmodel" \
    dataset_name="clevrer" \
    num_workers=8 \
    batch_size=256 \
    dinowm.history_size=6 \
    dinowm.num_preds=10 \
    frameskip=2 \
    videosaur.NUM_SLOTS=7 \
    videosaur.SLOT_DIM=128 \
    predictor.heads=16 \
    embedding_dir=$SLOTPATH \
    predictor_lr=5e-4 \
    num_masked_slots=0



