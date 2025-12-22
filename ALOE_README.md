# CLEVRER

We experiment on video prediction and VQA task in this dataset.


## VQA

For the VQA task, we leverage the SlotFormer model trained above.
We explicitly unroll videos to future frames, and provide them as inputs to train the downstream VQA task model (`Aloe`).

### Unroll SlotFormer for VQA task

To unroll videos, please use [rollout_clevrer_slots.py](../slotformer/video_prediction/rollout_clevrer_slots.py) and run:

```
python slotformer/video_prediction/rollout_clevrer_slots.py \
    --params slotformer/video_prediction/configs/slotformer_clevrer_params.py \
    --weight $WEIGHT \
    --save_path $SAVE_PATH (e.g. './data/CLEVRER/rollout_slots.pkl')
```

This will unroll slots for CLEVRER videos, and save them into a `.pkl` file (~16G).

Alternatively, we provide rollout slots as described in [benchmark.md](./benchmark.md).

### Train Aloe VQA model

To train an Aloe model only on the observed slots (this is the baseline in our paper), run:

```
python scripts/train.py --task clevrer_vqa \
    --params slotformer/clevrer_vqa/configs/aloe_clevrer_params.py \
    --fp16 --ddp --cudnn
```

To train an Aloe model on both observed and explicitly unrolled slots, run:

```
python scripts/train.py --task clevrer_vqa \
    --params slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
    --fp16 --ddp --cudnn
```

All the settings, except the slots are the same in these two experiments.
Alternatively, we also provide **pre-trained Aloe weight** as `pretrained/aloe_clevrer_params-rollout/model_400.pth`.

### Evaluate VQA results

Finally, to evaluate the VQA model on the test set, please use [test_clevrer_vqa.py](../slotformer/clevrer_vqa/test_clevrer_vqa.py) and run:

```
python slotformer/clevrer_vqa/test_clevrer_vqa.py \
    --params slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
    --weight $WEIGHT
```

This will save the results as `CLEVRER.json` under the same directory as the weight (we attach our result file as `pretrained/aloe_clevrer_params-rollout/CLEVRER.json`).
You can submit it to the [evaluation server](https://eval.ai/web/challenges/challenge-page/667/overview) of CLEVRER as see the results.
