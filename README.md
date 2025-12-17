# Environment Setup
```
conda create -n dino310 python=3.10 -y
conda activate dino310
conda install anaconda::ffmpeg
pip install seaborn webdataset swig einops
pip install -e ./stable-pretraining
pip install -e ./stable-worldmodel
```

# Dataset
## download clevrer
```sh
mkdir clevrer_video
cd clevrer_video
wget http://data.csail.mit.edu/clevrer/videos/train/video_train.zip
wget http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip
wget http://data.csail.mit.edu/clevrer/videos/test/video_test.zip
```
Create a folder named `videos/`, place the required videos inside it, and then run `python clevrer/clevrer.py`.

# Videosaur Checkpoints
| weight_sim | lr        | last checkpoint step | checkpoint      |
|------------|-----------|----------------------|-----------------|
| 0.1        | 1.00E-04  | 81k                  | [checkpoint](https://drive.google.com/file/d/1qZwWyXXTKbUMJYJ_h65QaO4_fgj_8wBL/view?usp=drive_link)     |
| 0.3        | 1.00E-04  | 81k                  | [checkpoint](https://drive.google.com/file/d/105BOoK1GYk3R9S95Sbmkrp8tO5ZO_AKD/view?usp=drive_link)     |
| 0.5        | 1.00E-04  | 82k                  | [checkpoint](https://drive.google.com/file/d/105BOoK1GYk3R9S95Sbmkrp8tO5ZO_AKD/view?usp=drive_link)     |

* If you want to train your own VIDEOSAUR model with Clevrer, please refer to [videosaur-branch-readme-here](https://github.com/rbalestr-lab/cjepa/blob/videosaur/README.md).
* These checkpoints are trained with `video_00000` from `video_05000`.

# Training and WM-checkpoints

## How to Run
Use scripts below, or refer to the command if you are not using slurm.

```sh
sbatch videowm.sh # run DINOwm
sbatch videowm_reg.sh # run DINOwm, but with dinov2_with_register checkpoint
sbatch ocwm.sh # run object centric world model, need VIDEOSAUR checkpoint downloaded from above.
sbatch causalwm.sh # run causalwm, which has causal slot masking with VJEPA style predictor.
```

* All config files are in `configs/`.
* All customed models are in `custom_models/`.
* Actual training files are in `train/`.

## Trained checkpoints

| model        | Predictor lr  | epoch  | checkpoint     |
|--------------|---------------|--------|----------------|
| videowm      | 5.00E-04      | 30     | [checkpoint](https://drive.google.com/file/d/12YVmnTK5NipNrjnQ4ysvXWhi0SOK8AZP/view?usp=drive_link) |
| videowm      | 5.00E-04      | 50     | [checkpoint](https://drive.google.com/file/d/1oinaGKVxFlt3OavlkYKaJR5co0r7NwhX/view?usp=drive_link) |
| videowm_reg  | 5.00E-04      | 30     | [checkpoint](https://drive.google.com/file/d/1pASFjlbjx4wJRfRNWL-bj2pSBqrK-E8j/view?usp=drive_link) |
| ocwm         | 5.00E-04      | 30     | [checkpoint](https://drive.google.com/file/d/1eapE0F4qGpWRwQ2QeS6QMoKna5eddgxb/view?usp=drive_link) |
| causalwm     | 5.00E-04      | 30     | [checkpoint](https://drive.google.com/file/d/15_0egh6YSJUsS0_6eJarYb0wY4GFrIKm/view?usp=drive_link) |
| causalwm     | 5.00E-04      | 65     | [checkpoint](https://drive.google.com/file/d/1wn82EWY0uSfVJ-8f8JNgt_8GkU6DtwVw/view?usp=drive_link) |

* OCWM and Causal WMs are trained with VIDEOSAUR checkpoint with `weight_sim=0.1`
* These checkpoints are trained with `train_split=0.8` and `seed=42`, among videos from `video_00000` from `video_10000`. (So there are some data leakage now, sorry my bad)

# Evaluation and Visualization 

## How to run 
```sh
PYTHONPATH=. python test/test_videowm.py checkpoint_path=ckpt/world_model_epoch_30_object.ckpt
PYTHONPATH=. python test/test_videowm_reg.py checkpoint_path=ckpt/world_model_reg_epoch_30_object.ckpt
PYTHONPATH=. python test/test_ocwm.py checkpoint_path=ckpt/world_model_oc_epoch_30_object.ckpt
PYTHONPATH=. python test/test_causalwm.py checkpoint_path=ckpt/causal_world_model_ver2_epoch_30_object.ckpt 
```
* Currently it calculates Rankme and Fréchet Joint Distance. Will keep update! (Last Update: Dec 17)

## Visualization (Last update Dec 17)
* Implemented to observe slot interactions.
* Current version visualizes 3D PCA slot trajectories.
