export PYTHONPATH=$(pwd)

# Don't forget to escape special characters like '=' with '\'
export SLOTPATH="path/to/extracted/slot/pkl"
# Put the path to the pretrained OC checkpoint here
export OC_CKPT_PATH="path/to/pretrained/oc/ckpt.ckpt"


# action, proprio, state meta files can be downloaded from the huggingface

python src/train/train_causalwm_AP_node_pusht_slot.py \
    cache_dir="~/.stable_worldmodel" \
    output_model_name="pusht_cjepa" \
    dataset_name="pusht_expert" \
    num_workers=8 \
    batch_size=256 \
    trainer.max_epochs=30 \
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
    model.load_weights=${OC_CKPT_PATH} \
    action_dir='/your/own/path/pusht_expert_action_meta.pkl' \
    proprio_dir='/your/own/path/pusht_expert_proprio_meta.pkl' \
    state_dir='/your/own/path/pusht_expert_state_meta.pkl' \
    use_hungarian_matching=false 




