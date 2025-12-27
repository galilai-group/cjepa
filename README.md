# Environment Setup
```
conda create -n dino310 python=3.10 -y
conda activate dino310
conda install anaconda::ffmpeg
pip install seaborn webdataset swig einops uv torchcodec
uv pip install -e ./stable-pretraining % make sure you init submodule
uv pip install -e ./stable-worldmodel % make sure you init submodule
uv pip install accelerate tensorboard tensorboardX.  
```
to run ALOE for clevrer VQA, install `nerv` and s`pycocotools` as well.
```
git clone https://github.com/Wuziyi616/nerv.git
cd nerv
git checkout v0.1.0  # tested with v0.1.0 release
uv pip install -e .
pip install pycocotools
```

# Dataset
## Download clevrer (~24G total)
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

## Prepare CLEVRER Stable-WM dataset
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
└── clevrer_val/
     ├── data-00000-of-000001.arrow
     ├── dataset_info.json
     ├── state.json
     └── videos
          └──10000_pixels.mp4 ...
```

## Prepare CLEVRER Videosaur dataset
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

## Download PushT
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

# Training and WM-checkpoints

## How to Run


Use scripts below, or refer to the command if you are not using slurm.

```sh
sbatch script/{dataset}/run_videowm.sh # run DINOwm with mp4
sbatch script/{dataset}/run_videowm_reg.sh # run DINOwm, but with dinov2_with_register checkpoint
sbatch script/{dataset}/run_ocwm.sh # run object centric world model, need VIDEOSAUR checkpoint downloaded from above.
% [WIP] sbatch script/{dataset}/causalwm.sh # run causalwm, which has causal slot masking with VJEPA style predictor.
```

* All config files are in `configs/`.
* All customed models are in `custom_models/`.
* Actual training files are in `train/`.


# CLEVRER VQA

```
# clevrer_slots.pkl :
{
    'train': {'video_0001': slots, 'video_0002': slots, ...},  # [128, 7, 64] each
    'val': {...},
    'test': {...}
}
```
