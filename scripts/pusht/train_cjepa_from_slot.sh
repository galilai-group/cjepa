export PYTHONPATH=$(pwd)

# don't forget to escape special characters like '=' with '\'
export SLOTPATH="/cs/data/people/hnam16/data/modified_extraction/pusht_expert_slots_videosaur_172.pkl"
# export SLOTPATH="/cs/data/people/hnam16/data/modified_extraction/pusht_savi_101.pkl"

# this is for saving swm ckpt for smooth planning.. this should be matched with the pusht ckpt used for slot extraction
export CKPT_PATH="/cs/data/people/hnam16/.stable_worldmodel/artifacts/oc-checkpoints/videosaur_172.ckpt"

# torchrun --nproc_per_node=3 --master-port=29501 \

# Caution!! Set output_model_name properly 
python train/train_causalwm_AP_node_pusht_slot.py \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
    output_model_name="263p" \
    dataset_name="pusht_expert" \
    num_workers=8 \
    batch_size=256 \
    trainer.max_epochs=20 \
    num_masked_slots=1 \
    predictor_lr=5e-4 \
    proprio_encoder_lr=1e-4  \
    action_encoder_lr=5e-4  \
    dinowm.history_size=3 \
    dinowm.num_preds=1 \
    dinowm.proprio_embed_dim=128 \
    dinowm.action_embed_dim=128 \
    frameskip=5 \
    videosaur.NUM_SLOTS=4 \
    videosaur.SLOT_DIM=128 \
    predictor.heads=16 \
    embedding_dir=${SLOTPATH} \
    model.load_weights=${CKPT_PATH} \
    use_hungarian_matching=false \




