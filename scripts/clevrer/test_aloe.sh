echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

python slotformer/clevrer_vqa/test_clevrer_vqa.py \
  --params slotformer/clevrer_vqa/configs/aloe_clevrer_param_for_test.py \
  --weight '/cs/data/people/hnam16/aloe_checkpoint/226_model_400.pth' \
  --slots_root_override '/cs/data/people/hnam16/data/modified_extraction/rollout_clevrer_savi_reproduced_ocvp.pkl' \
  --validate