# Dataset
* If you are using pre-extracted slots for training C-JEPA, you can skip everything here.
* If you are using VideoSAUR for object centric encoder (either by training yourself or downloading the checkpoint), you need to follow the instruction here to prepare the dataset.
* If you are using SAVi for object centric encoder (either by training yourself or downloading the checkpoint), please follow the [instruction](https://github.com/pairlab/SlotFormer/blob/master/docs/data.md) in slotformer repo to setup data. Although, we only use SAVi for CLEVRER dataset, you can also use SAVi for PushT by following the similar data preparation instruction.
* If you are testing downstream (VQA or planning), you need to prepare the dataset for evaluation. 


## CLEVRER
### 1. Download original data (~24G total)
```sh
#!/usr/bin/env bash

ROOT_DIR="./clevrer_video"

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

### 2. Reformat CLEVRER for Stable-WorldModel
If you are using pre-extracted slots, you can skip this step.
This step is required for extracting slots from object-centric encoders.
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

### 3 Prepare CLEVRER Videosaur dataset
```
% You don't need this if you are not running videosaur for CLEVRER.
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

## Push-T

### 1. Download PushT for Stable-WorldModel
* Download `pusht_expert_{train/val}_video` data from [link](https://drive.google.com/drive/folders/1M7PfMRzoSujcUkqZxEfwjzGBIpRMdl88).
* Unzip and put them under `swm.data.utils.get_cache_dir()`. Default directory is `~/.stable_worldmodel/`. But you can put them anywhere and set the `cache_dir` argument in the file before running.
```
pip install gdown
gdown https://drive.google.com/uc?id=1vpcCwxsFNEzsZbWSyei-qt_cw4OA7DT-
gdown https://drive.google.com/uc?id=17m8-1JLF2nA7MZnVtvNmz3sFXcqGKxxQ
tar --zstd -xvf pusht_expert_train_video.tar.zst
tar --zstd -xvf pusht_expert_val_video.tar.zst
```
* Rename folder as a desired format.
* This naming is required when you want to work with your own dataset: {dataset_name}_train and {dataset_name}_val. 

```
mv pusht_expert_train_video pusht_expert_train # so it should be .stable_worldmodel/pusht_expert_train
mv pusht_expert_val_video pusht_expert_val  # sol it should be .stable_worldmodel/pusht_expert_val
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

### 2. Prepare PushT Videosaur dataset

* Generate randomly-moving PushT data for better object-centric learning. (10000 for train, 1000 for val)
```
PYTHONPATH=. python dataset/pusht/pusht_all_moving_videogen.py \
    --num_videos 11000 \
    --output_dir my_dataset 
```

* Generate webdataset shards for VideoSAUR training
We will mix the original videos (video_10000.mp4 - video_18684.mp4) with the 10000 randomly moving videos.
You can set the directory paths in the file before running.
```
PYTHONPATH=. python dataset/pusht/save_mixed_pusht_webdataset_mp4.py
```