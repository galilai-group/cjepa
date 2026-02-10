export PYTHONPATH=$(pwd)

# torchrun --nproc_per_node=2 --master-port=29501 \
python src/aloe_train.py \
  --task clevrer_vqa \
  --params slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
  --exp_name 228 \
  --out_dir /cs/data/people/hnam16/aloe_checkpoint \
  --slot_root_override '/cs/data/people/hnam16/data/modified_extraction/rollout_clevrer_slots_step=100000_weight03_lr1e-4_clevrer_lr0.0005_exp202p.pkl' \
  --fp16 --cudnn \
  # --ddp
