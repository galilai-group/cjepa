export PYTHONPATH=$(pwd)


# we recommend using multli gpu for training aloe.
# torchrun --nproc_per_node=3 \
python src/aloe_train.py \
  --task clevrer_vqa \
  --params src/third_party/slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
  --exp_name 'aloe_clevrer' \
  --out_dir /path/to/output/directory \
  --slot_root_override 'to/path/to/rollout/slot' \
  --fp16 --cudnn \
  # --ddp
