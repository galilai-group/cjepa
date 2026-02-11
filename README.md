# C-JEPA
## Causal-JEPA: Learning World Models through Object-Level Latent Interventions
by [Heejeong Nam](https://hazel-heejeong-nam.github.io/), [Quentin Le Lidec\*](https://quentinll.github.io/),[Lucas Maes\*](https://lucasmaes.bearblog.dev/), [Yann LeCun](http://yann.lecun.com/), [Randall Balestriero](https://randallbalestriero.github.io/).

* Paper: [soon](soon)

![architecture](static/architecture.png)

## Summary
World models require robust relational understanding to support prediction, reasoning, and control. While object-centric representations provide a useful abstraction, they are not sufficient to capture interaction-dependent dynamics. We therefore propose C-JEPA, a simple and flexible object-centric world model that extends masked joint embedding prediction from image patches to object-centric representations. 
By applying object-level masking that requires an object's state to be inferred from other objects, C-JEPA induces latent interventions with counterfactual-like effects and prevents shortcut solutions, making interaction reasoning essential.
Empirically, C-JEPA leads to consistent gains in visual question answering, with **an absolute improvement of about 20\% in counterfactual reasoning** compared to the same architecture without object-level masking. On agent control tasks, C-JEPA enables substantially more efficient planning by **using only 1\% of the total latent input features required by patch-based world models**, while achieving comparable performance. Finally, we provide a formal analysis demonstrating that object-level masking induces a causal inductive bias via latent interventions.


## Setup
* Please refer to [ENV.md](docs/ENV.md) for environment setup.
* C-JEPA is mainly built on top of [Stable-WorldModel](https://galilai-group.github.io/stable-worldmodel/) and [Stable-Pretraining](https://galilai-group.github.io/stable-pretraining/).
* We adopt [original repo here](https://github.com/martius-lab/videosaur) for training VideoSAUR.
* We adopt [original repo here](https://github.com/pairlab/SlotFormer) for SAVi and VQA model for CLEVRER (a.k.a. ALOE).

## Dataset Preparation
Please refer to [DATASET.md](docs/DATASET.md) for dataset preparation.

## Object-Centric Model / Representations for C-JEPA
C-JEPA relies on object-centric encoders to extract object-centric representations. You can train the encoder by yourself, or download the model checkpoints from HuggingFace, or download the pre-extracted slot representations.

![visualization](static/encoder_vis.png)

### 1. Train Object-Centric Encoder
* Train VideoSAUR on CLEVRER
  ```
  PYTHONPATH=. python src/thrid_party/videosaur/videosaur/train.py \
      src/thrid_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml \
      dataset.train_shards="clevrer-train_videos-{000000..0000XX}.tar" \
      dataset.val_shards="clevrer-val_videos-{000000..0000XX}.tar"
  ```

* Train SAVi on CLEVRER

  refer to slotformer repo to setup data for train savi

  ```
  PYTHONPATH=. torchrun --nproc_per_node=3 src/aloe_train.py \
    --task base_slots \
    --params src/thrid_party/slotformer/base_slots/configs/stosavi_clevrer_params.py \
    --exp_name clevrer_savi_reproduce \
    --out_dir $OUTDIR \
    --fp16 --ddp --cudnn
  ```

* Train VideoSAUR on PushT
  ```
  PYTHONPATH=. python src/thrid_party/videosaur/videosaur/train.py \
      src/thrid_party/videosaur/configs/videosaur/pusht_dinov2_hf.yml \
      dataset.train_shards="pusht_mixed/train/pusht-train-{000000..000034}.tar" \
      dataset.val_shards="pusht_mixed/validation/pusht-val-{000000..000007}.tar"
  ```

### 2. Model Checkpoints
| Dataset      | Encoder    |  Hyperparams                      | Checkpoint Link                                                                                             |
|--------------|------------------|---------------------------|------------------------------------------------------------------------------------------------------------|
| CLEVRER       | VideoSAUR   | clevrer_dinov2_hf.yml        | [Checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/clevrer_videosaur_model.ckpt) |
| CLEVRER       | SAVi   | stosavi_clevrer_params.py        | [Checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/clevrer_savi_model.pth) |
| Push-T       | VideoSAUR   | pusht_dinov2_hf.yml        | [Checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/pusht_videosaur_model.ckpt) |

### 3. Pre-extracted Slot Representations
| Dataset      | Encoder    |  Config                      | Checkpoint Link                                                                                             |
|--------------|------------------|---------------------------|------------------------------------------------------------------------------------------------------------|
| CLEVRER       | VideoSAUR   | clevrer/ocwm_clevrer.yml        | [Checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/clevrer_videosaur_slots.pkl) |
| CLEVRER       | SAVi   | clevrer/causalwm_clevrer.yml        | [Checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/clevrer_savi_slots.pkl) |
| PUSHT       | VideoSAUR   | pusht/ocwm_pusht.yml        | [Checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/pusht_videosaur_slots.pkl) |


## Train / Download C-JEPA

### Extract slot representations with the checkpoints

* CLEVRER SAVi slots
  ```
  PYTHONPATH=. python src/third_party/slotformer/base_slots/extract_slots.py --params slotformer/base_slots/configs/stosavi_clevrer_params.py  --weight $WEIGHT  --save_path clevrer_savi_slots.pkl
  ```
* CLEVRER VideoSAUR 
  ```
  PYTHONPATH=. python src/third_party/slotformer/base_slots/extract_videosaur.py --weight $WEIGHT --data_root="~/.stable_worldmodel"   --save_path=$SAVE_DIR --dataset="clevrer"  --videosaur_config="src/thrid_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml" 
  ```
* PushT VideoSAUR Slots
  ```
  PYTHONPATH=. python src/third_party/slotformer/base_slots/extract_videosaur.py --weight $WEIGHT --data_root="~/.stable_worldmodel"   --save_path=$SAVE_DIR  --dataset="pusht_expert"  --videosaur_config="videosaur/configs/videosaur/pusht_dinov2_hf.yml"   --params="slotformer/aloe_pusht_params.py"
  ```

* Extracted pkl will look like:

  ```
  {
      'train': {'0_pixels.mp4': slots, '1_pixels.mp4': slots, ...},  # slots: [T, N, 128] each
      'val': {...},
      'test': {...}
  }
  ```

### 2. Run C-JEPA with pre-extracted slot representations

Use scripts below, or refer to the command if you are not using slurm.

```sh
sh script/clevrer/run_causalwm_from_slot.sh
sh script/pusht/run_causalwm_AP_node_from_slot_videosaur.sh 
```

### 3. Download C-JEPA checkpoints
* We release checkpoints for C-JEPA trained with different object-centric backbones and different number of masked slots. 
* (*) means it performs best. (See the paper for details.)

| Dataset      |   OC-Backbone    |    Masked Slots / Total Slots    | Checkpoint Link                                                 |
|--------------|------------------|---------------------------|------------------------------------------------------------------|
| CLEVRER       | VideoSAUR   | 0/7       | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_videosaur_0.ckpt)                             |
| CLEVRER       | VideoSAUR   | 1/7       | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_videosaur_1.ckpt)                             |
| CLEVRER       | VideoSAUR   | 2/7      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_videosaur_2.ckpt)                             |
| CLEVRER       | VideoSAUR   | 3/7      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_videosaur_3.ckpt)                             |
| CLEVRER       | VideoSAUR   | 4/7 (*)     | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_videosaur_4.ckpt)                             |
| CLEVRER       | SAVi        | 0/7      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_savi_0.ckpt)                             |
| CLEVRER       | SAVi        | 1/7      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_savi_1.ckpt)                             |
| CLEVRER       | SAVi        | 2/7 (*)     | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_savi_2.ckpt)                             |
| CLEVRER       | SAVi        | 3/7      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_savi_3.ckpt)                             |
| CLEVRER       | SAVi        | 4/7      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_savi_4.ckpt)                             |
| PUSHT       | VideoSAUR   | 0/4      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/pusht_videosaur_0.ckpt)                             |
| PUSHT       | VideoSAUR   | 1/4  (*)    | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/pusht_videosaur_1.ckpt)                             |
| PUSHT       | VideoSAUR   | 2/4      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/pusht_videosaur_2.ckpt)                             |   

## Evaluation
### Evaluate Control on Push-T
  ```
  sh scripts/pusht/run_planning.sh
  ```

### Evaluate Visual Reasoning on CLEVRER

  * We will first rollout slots (from 128 frame to 160 frame) with C-JEPA checkpoint and pre-extracted slots.

  ```
  # change CKPTPATH and SLOTPATH, `predictor_lr`, `num_masked_slots` before.
  # `predictor_lr`, `num_masked_slots` should be matched with config is CKPT that you are using
  # output name will be like : rollout_{SLOTPATH name}_{CKPT lr, CKPT masked slot}
  sbatch scripts/clevrer/rollout_from_slot.sh
  ```
  * This will save
  ```
  # rollout_clevrer_slots_{configuration}.pkl :
  {
      'train': {'0_pixels.mp4': slots, '1_pixels.mp4': slots, ...},  # slots: [160, N, 128] each
      'val': {...},
      'test': {...}
  }
  ```

  * Before running the code, replace whole `src/third_party/nerv/nerv/utils/misc.py` with `src/custom_codes/misc.py`. This is because the original code is based on `pytorch-lightning==0.8.*` while we are using `pytorch-lightning==2.6.*`.
  * You should change params manually in `sloformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py`. For example, 
    * `gpu` (it should exactly match the number of the visible devices)
    * `slots_root` (path to `rollout_clevrer_slots_{configuration}.pkl`)
    * `lr`
  * Then run script below to train and test aloe.

  ```
  sh scripts/clevrer/train_aloe.sh
  sh scripts/clevrer/test_aloe.sh
  ```
