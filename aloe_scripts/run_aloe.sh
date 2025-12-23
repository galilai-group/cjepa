python scripts/train.py --task clevrer_vqa \
    --params slotformer/clevrer_vqa/configs/aloe_clevrer_params.py \
    --fp16 --ddp --cudnn



python scripts/train.py --task clevrer_vqa \
    --params slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
    --fp16 --ddp --cudnn


#eval 
python slotformer/clevrer_vqa/test_clevrer_vqa.py \
    --params slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
    --weight pretrained/aloe_clevrer_params-rollout/model_400.pth