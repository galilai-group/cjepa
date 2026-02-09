# 1. Environment Setup
```
conda create -n dino310 python=3.10 -y
conda activate dino310
conda install anaconda::ffmpeg
pip install seaborn webdataset swig einops uv torchcodec av
uv pip install -e ./stable-pretraining % make sure you init submodule
uv pip install -e ./stable-worldmodel % make sure you init submodule
uv pip install accelerate tensorboard tensorboardX hickle
```
to run ALOE for clevrer VQA, install `nerv` and s`pycocotools` as well.
```
git clone https://github.com/Wuziyi616/nerv.git
cd nerv
git checkout v0.1.0  # tested with v0.1.0 release
uv pip install -e .
pip install pycocotools
```

# 2. Dataset
## 2.1 Download clevrer (~24G total)
```sh
#!/usr/bin/env bash

ROOT_DIR="/users/hnam16/scratch/clevrer_video"

mkdir -p \
  ${ROOT_DIR}/train \
  ${ROOT_DIR}/val \
  ${ROOT_DIR}/test

echo "Downloading CLEVRER videos..."

wget -nc -P ${ROOT_DIR}/train \
  http://data.csail.mit.edu/clevrer/videos/train/video_train.zip

wget -nc -P ${ROOT_DIR}/val \
  http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip

wget -nc -P ${ROOT_DIR}/test \
  http://data.csail.mit.edu/clevrer/videos/test/video_test.zip

echo "Unzipping..."
unzip -q ${ROOT_DIR}/train/video_train.zip -d ${ROOT_DIR}/train
unzip -q ${ROOT_DIR}/val/video_validation.zip -d ${ROOT_DIR}/val
unzip -q ${ROOT_DIR}/test/video_test.zip -d ${ROOT_DIR}/test

echo "Flattening mp4 files..."

for split in train val test; do
  find ${ROOT_DIR}/${split} -type f -name "*.mp4" -exec mv {} ${ROOT_DIR}/${split}/ \;
  find ${ROOT_DIR}/${split} -type d ! -path ${ROOT_DIR}/${split} -exec rm -rf {} +
done

echo "Done."
```

This will give you 
```
ROOT_DIR/
├── train/
│   ├── video_00000.mp4
│   ├── video_00001.mp4
│   └── ...
├── val/
│   ├── video_10000.mp4
│   └── ...
└── test/
    ├── video_15000.mp4
    └── ...
```

## 2.2 Prepare CLEVRER Stable-WM dataset
```
% set ROOT_DIR in the file first
python dataset/clevrer/clevrer.py
```
* This will create clevrer dataset under stable-wm cache directory (by calling `swm.data.utils.get_cache_dir()`) in a desired format.
* We will use deterministic train / val  setup - your cache directory will look like

```
.stable_worldmodel
├── clevrer_train/
|    ├── data-00000-of-000001.arrow
|    ├── dataset_info.json
|    ├── state.json
|    └── videos
|         └──0_pixels.mp4 ...
├── clevrer_val/
|    ├── data-00000-of-000001.arrow
|    ├── dataset_info.json
|    ├── state.json
|    └── videos
|         └──10000_pixels.mp4 ...
└── clevrer_test/
     ├── data-00000-of-000001.arrow
     ├── dataset_info.json
     ├── state.json
     └── videos
          └──15000_pixels.mp4 ...
```

## 2.3 Prepare CLEVRER Videosaur dataset
```
% You don't need this if you are not running videosaur.
% set ROOT_DIR in the file first
python dataset/clevrer/save_clevrer_webdataset_mp4.py
```
This will give you 
```
ROOT_DIR/
├── train/
├── val/
├── test/
└── clevrre_wds_mp4
    ├── train
    |   └── clevrer-train-000000.tar ...
    └── val
        └── clevrer-val-000000.tar ...

```

## 2.4 Download PushT
* Download data from https://drive.google.com/drive/folders/1M7PfMRzoSujcUkqZxEfwjzGBIpRMdl88
* Unzip and put them under `swm.data.utils.get_cache_dir()`.
* rename folder as a desired format

```
mv pusht_expert_train_video pusht_expert_train
mv pusht_expert_val_video pusht_expert_val

```

This will give you

```
.stable_worldmodel
├── pusht_expert_train/
|    ├── data-00000-of-000001.arrow
|    ├── dataset_info.json
|    ├── state.json
|    └── videos
|         └──0_pixels.mp4 ...
└── pusht_expert_val/
     ├── data-00000-of-000001.arrow
     ├── dataset_info.json
     ├── state.json
     └── videos
          └──0_pixels.mp4 ...
```

## 2.5 Generate randomly-moving PushT data (if needed)
```
PYTHONPATH=. python dataset/pusht/pusht_all_moving_videogen.py \
    --num_videos 10000 \
    --output_dir my_dataset \
```

# 3. Training and WM-checkpoints

## 3.1 How to Run


Use scripts below, or refer to the command if you are not using slurm.

```sh
sbatch script/{dataset}/run_videowm.sh # run DINOwm with mp4
sbatch script/{dataset}/run_videowm_reg.sh # run DINOwm, but with dinov2_with_register checkpoint
sbatch script/{dataset}/run_ocwm.sh # run object centric world model, need VIDEOSAUR checkpoint downloaded from above.

# make sure you load the correct checkpoint while running below!!
sbatch script/{dataset}/run_ocwm.sh # run object centric world model, need VIDEOSAUR checkpoint downloaded from above.
sbatch script/{dataset}/run_causalwm.sh # run causalwm, which has causal slot masking. style predictor.

# for clevrer, running causalwm with pre-extracted slot embeddings is available. (Extraction in Sec 4.1)
# x10 times faster
sbatch script/clevrer/run_causalwm_from_slot.sh

# only for pusht. When training cjepa predictor, it considers Action and Proprioception as causal nodes as well.
sbatch script/pusht/run_causalwm_AP_node.sh 
```

* All config files are in `configs/`.
* All customed models are in `custom_models/`.
* Actual training files are in `train/`.


# 4. CLEVRER VQA

## 4.1 Prepare data for ALOE
 * Download checkpoints from https://drive.google.com/drive/u/3/folders/1by2KKB6f8y4KEQaMiRt1R8PswgAgoaft
 * Extract slot representation from each checkpoint by 

  ```sh
  % PATH-TO-DATA should have clevrer_train clever_val clevrer_test with /videos under them respectively.

  PYTHONPATH=. python slotformer/base_slots/extract_videosaur.py \
    --weight=PATH-TO-CKPT-FILE \
    --data_root=PATH-TO-DATA   \ % swm.data.utils.get_cache_dir() or equivalent
    --save_path=DIR-TO-SAVE-PIL  % should contain "clevrer_slots"
  ```
  * Extracted pkl will look like:
  ```
  # clevrer_slots_{configuration}.pkl :
  {
      'train': {'0_pixels.mp4': slots, '1_pixels.mp4': slots, ...},  # slots: [128, 7, 128] each
      'val': {...},
      'test': {...}
  }
  ```


  <!-- ## 4.2 Run ALOE w/o Predictor
  * 4.1 should be done
  * Before running the code, replace `nerv/nerv/utils/misc.py` with `custom_codes/misc.py`. This is because the original code is based on `pytorch-lightning==0.8.*` while we are using `pytorch-lightning==2.6.*`.
  * You should change params manually in `sloformer/clevrer_vqa/configs/aloe_clevrer_params.py`. For example, 
    * `gpu` (it should exactly match the number of the visible devices)
    * `slots_root`
    * `lr`

  ```
  sbatch scripts/run_aloe.sh
  ``` -->


  ## 4.2 Run ALOE 
  * This step needs slots from 4.1 and CJEPA checkpoint from 3.1
  * We will first rollout slots (from 128 frame to 160 frame) with checkpoint

  ```
  # change CKPTPATH and SLOTPATH, `predictor_lr`, `num_masked_slots` before.
  # `predictor_lr`, `num_masked_slots` should be matched with config is CKPT that you are using
  # output name will be like : rollout_{SLOTPATH name}_{CKPT lr, CKPT masked slot}
  sbatch scripts/clevrer/rollout_causalwm_from_slot.sh
  ```
  * This will save
  ```
  # rollout_clevrer_slots_{configuration}.pkl :
  {
      'train': {'0_pixels.mp4': slots, '1_pixels.mp4': slots, ...},  # slots: [160, 7, 128] each
      'val': {...},
      'test': {...}
  }
  ```

  * Now we are ready for running ALOE.
  * Before running the code, replace `nerv/nerv/utils/misc.py` with `custom_codes/misc.py`. This is because the original code is based on `pytorch-lightning==0.8.*` while we are using `pytorch-lightning==2.6.*`.
  * You should change params manually in `sloformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py`. For example, 
    * `gpu` (it should exactly match the number of the visible devices)
    * `slots_root` (path to `rollout_clevrer_slots_{configuration}.pkl`)
    * `lr`
  * Then finally run
  ```
  sbatch scripts/run_aloe_rollout.sh
  ```
