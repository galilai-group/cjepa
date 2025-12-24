#  1. Huggingface-based VideoSAUR (Recommended)
* Python version : 3.12 recommended
```
git clone https://github.com/rbalestr-lab/cjepa.git
conda create -n videosaur312 python=3.12 -y
conda activate videosaur312
pip install -r videosaur/requirements_transition_py312.txt
```


# 2. CLEVRER Data Preparation

* Download data
```
wget http://data.csail.mit.edu/clevrer/videos/train/video_train.zip 
wget http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip 
wget http://data.csail.mit.edu/clevrer/videos/test/video_test.zip 
```

  * Unzip in each folder train, validation, test
  * Run the code below to convert video into `webdataset` format (to make it compatible to videosaur)

```
python dataset/clevrer/save_clevrer_webdataset.py 
% you should change the directory in this code

Temporal subsampling (num_frameskip)
----------------------------------

You can optionally subsample videos temporally by adding `num_frameskip` to your pipeline config.
Set `num_frameskip: N` to keep every N-th frame (frames: 0, N, 2N, ...). Importantly, the subsampling is
applied *before* chunking. This means that if you set `chunk_size=8` and `num_frameskip=2`, the chunk
will contain 8 frames sampled as indices 0,2,4,...,14 (i.e., the first option), not just 4 frames.

Where to set it in the config:
- At the pipeline level (recommended):
Set `num_frameskip` at the pipeline level (recommended):

```yaml
train_pipeline:
  chunk_size: 8
  num_frameskip: 2
  keys: [video]
  transforms: {...}
```
```

# 4. How to Run
* This is slotformer configs: I will use the same number of slots.
  * CLEVERERL num-slot =7, d-slot = 128, trainingstep = 200k burn-in-T = 6, rollout = 10
  * PHYRE num-slot = 8, d-slot = 128, trainingstep = 370k

* Now custom config file : `videosaur/configs/videosaur/clevrer_dinov2.yml` is the one use should use


  * What you can (might) change for system:

    * BATCH_SIZE_PER_GPU
    * NUM_GPUS
    * train_shards
    * val_shards
    * val_batch_size
    * num_workers
    * num_val_workers
    * accelerator

  * This is the main parameters that you can do param-tuning
    * BASE_LR
    * SIM_WEIGHT
    * SIM_TEMP


If you are not using wandb, you should set wandb config to `False`

### Now you can run the code

* HF-based
```
PYTHONPATH=. python -m  pdb videosaur/train.py configs/videosaur/clevrer_dinov2_hf.yml 
```
* Timm-based
```
poetry run python -m videosaur.train configs/videosaur/clevrer_dinov2.yml
```


**FYI, original repo uses segmentation mask to evaluate (should need segmentation mask for each frame), but we doesn't have those segmentation masks for cleverer. Thus I removed semgentation related part from the original code.**





# [Official Readme] VideoSAUR

This is the code release for the paper **Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities (NeurIPS 2023)**, by Andrii Zadaianchuk, Maximilian Seitzer and Georg Martius.

- Webpage: https://martius-lab.github.io/videosaur
- Arxiv: https://arxiv.org/abs/2306.04829
- OpenReview: https://openreview.net/forum?id=t1jLRFvBqm

![Temporal Feature Similarities](https://zadaianchuk.github.io/videosaur/static/images/sim_loss-1.png)

## Summary

Unsupervised video-based object-centric learning is a promising avenue to learn structured representations from large, unlabeled video collections, but previous approaches have only managed to scale to real-world datasets in restricted domains. Recently, it was shown that the reconstruction of pre-trained self-supervised features leads to object-centric representations on unconstrained real-world image datasets. Building on this approach, we propose a novel way to use such pre-trained features in the form of a temporal feature similarity loss. This loss encodes semantic and temporal correlations between image patches and is a natural way to introduce a motion bias for object discovery. We demonstrate that this loss leads to state-of-the-art performance on the challenging synthetic MOVi datasets. When used in combination with the feature reconstruction loss, our model is the first object-centric video model that scales to unconstrained video datasets such as YouTube-VIS.

## Usage

### Setup

First, setup the python environment setup. We use [Poetry](https://python-poetry.org/) for this:

```
poetry install
```

Then you could run a test configuration to see if everything works:

```
poetry run python -m videosaur.train tests/configs/test_dummy_image.yml
```

Second, to download the datasets used in this work, follow the instructions in [data/README.md](data/README.md).
By default, datasets are expected to be contained in the folder `./data`.
You can change this to the actual folder your data is in by setting the environment variable `VIDEOSAUR_DATA_PATH`, or by running `train.py` with the `--data-dir` option.

### Training

Run one of the configurations in `configs/videosaur`, for example:

```
poetry run python -m videosaur.train configs/videosaur/movi_c.yml
```

The results are stored in a folder created under the log root folder (by defaults `./logs`, changeable by the argument `--log-dir`).
If you want to continue training from a previous run, you can use the `--continue` argument, like in the following command:

```
poetry run python -m videosaur.train --continue <path_to_log_dir_or_checkpoint_file> configs/videosaur/movi_c.yml
```

### Inference
If you want to run one of the released checkpoints (see below) on your own video you can use inference script with corresponding config file:

```
poetry run python -m videosaur.inference --config configs/inference/movi_c.yml
```
in the released config, please change `checkpoint: path/to/videosaur-movi-c.ckpt` to the real path to your checkpoint.
For different video formats you would need to modify corresponding transformations in `build_inference_transform` function.

## Results

### VideoSAUR

We list the results you should roughly be able to obtain with the configs included in this repository:

| Dataset      | Model Variant    | Video ARI | Video mBO | Config                      | Checkpoint Link                                                                                             |
|--------------|------------------|-----------|-----------|-----------------------------|------------------------------------------------------------------------------------------------------------|
| MOVi-C       | ViT-B/8, DINO    | 64.8      | 38.9      | videosaur/movi_c.yml        | [Checkpoint](https://huggingface.co/andriizadaianchuk/videosaur-movi-c/resolve/main/videosaur-movi-c.ckpt) |
| MOVi-E       | ViT-B/8, DINO    | 73.9      | 35.6      | videosaur/movi_e.yml        | [Checkpoint](https://huggingface.co/andriizadaianchuk/videosaur-movi-e/resolve/main/videosaur-movi-e.ckpt) |
| YT-VIS 2021  | ViT-B/16, DINO   | 39.5      | 29.1      | videosaur/ytvis.yml         | [Checkpoint](https://huggingface.co/andriizadaianchuk/videosaur-ytvis/resolve/main/videosaur-ytvis.ckpt)   |
| YT-VIS 2021  | ViT-B/14, DINOv2 | 39.7      | 35.6      | videosaur/ytvis_dinov2.yml  | [Checkpoint](https://huggingface.co/andriizadaianchuk/videosaur-ytvis-dinov2-518/resolve/main/videosaur_dinov2.ckpt) |

### DINOSAUR

We also include a configuration for the DINOSAUR model from our previous paper [Bridging the gap to real-world object-centric learning](https://arxiv.org/abs/2209.14860).
This configuration yields improved results compared to the DINOSAUR model in the original paper (mainly
due to using DINOv2 pre-trained features).
Note that there might be minor differences in the metrics, as the numbers here are computed for 224x224 masks, compared to 320x320 masks in the DINOSAUR paper.

| Dataset | Model Variant    | Image ARI | Image mBO | Config                           | Checkpoint                                                                                             |
|---------|------------------|-----------|-----------|----------------------------------|--------------------------------------------------------------------------------------------------------|
| COCO    | ViT-B/14, DINOv2 | 45.6      | 29.6      | dinosaur/coco_base14_dinov2.yml  | [Checkpoint](https://huggingface.co/andriizadaianchuk/dinosaur_dinov2/resolve/main/coco_dinosaur_base14_dinov2.ckpt) |

## Citation

If you make use of this repository, please use the following bibtex entry to cite us:

```
  @inproceedings{zadaianchuk2023objectcentric,
      title={Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities},
      author={Zadaianchuk, Andrii and Seitzer, Maximilian and Martius, Georg},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS 2023)},
      year={2023},
  }
```

## License

This codebase is released under the MIT license.
Some parts of the codebase were adapted from other codebases.
A comment was added to the code where this is the case.
Those parts are governed by their respective licenses.
