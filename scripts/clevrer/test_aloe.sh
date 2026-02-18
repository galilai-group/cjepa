export PYTHONPATH=$(pwd)

python src/third_party/slotformer/clevrer_vqa/test_clevrer_vqa.py \
  --params src/third_party/slotformer/clevrer_vqa/configs/aloe_clevrer_param_for_test.py \
  --weight 'path/to/aloe/model/weight' \
  --slots_root_override '/path/to/rollout/slot' \
  --validate