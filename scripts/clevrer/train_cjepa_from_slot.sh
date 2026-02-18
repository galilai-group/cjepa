export PYTHONPATH=$(pwd)
# becareful if you have special characters in the path like '=': Need escape it with '\'
export SLOTPATH="path/to/extracted/slot/pkl"


python src/train/train_causalwm_from_clevrer_slot.py \
    cache_dir="~/.stable_worldmodel" \
    output_model_name="clevrer_cjepa" \
    dataset_name="clevrer" \
    num_workers=8 \
    batch_size=256 \
    trainer.max_epochs=30 \
    num_masked_slots=4 \
    predictor_lr=5e-4 \
    dinowm.history_size=6 \
    dinowm.num_preds=10 \
    frameskip=2 \
    videosaur.NUM_SLOTS=7 \
    videosaur.SLOT_DIM=128 \
    predictor.heads=16 \
    embedding_dir=$SLOTPATH 



