export PYTHONPATH=$(pwd)

# becareful if you have special characters in the path like '=': Need escape it with '\'
export CKPT_PATH="/cs/data/people/hnam16/.stable_worldmodel/artifacts/oc-checkpoints/step\=100000_weight03_lr5e-4_pusht.ckpt"

torchrun --nproc_per_node=2 --master-port=29505 \
    train/train_causalwm_AP_node.py \
    cache_dir="/cs/data/people/hnam16/.stable_worldmodel" \
    output_model_name="57" \
    dataset_name="pusht_expert" \
    num_workers=4 \
    batch_size=128 \
    trainer.max_epochs=10 \
    num_masked_slots=4 \
    model.load_weights=${CKPT_PATH} \
    predictor_lr=5e-4 \
    proprio_encoder_lr=5e-4 \
    action_encoder_lr=5e-4 \
    dinowm.history_size=3 \
    dinowm.num_preds=1 \
    frameskip=5 \
    videosaur.NUM_SLOTS=4 \
    videosaur.SLOT_DIM=64 \
    predictor.heads=16

