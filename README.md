# Env
```
conda create -n dino310 python=3.10 -y
conda activate dino310
pip install -e ./stable-pretraining
pip install -e ./stable-worldmodel
pip install einops imageio av
pip install seaborn webdataset # videosaur
```
if running oc wm
```
pip install webdataset # not using
```
# Dataset
## download clevrer
```
wget http://data.csail.mit.edu/clevrer/videos/train/video_train.zip
wget http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip
wget http://data.csail.mit.edu/clevrer/videos/test/video_test.zip
```
Then you should change path `root_dir` in `clevrer/{split}.json`

## ~~download phyre~~ (in progress)
Phyre only supports on-the-fly data reading, but because the python version conflicts, we manually should extract video.
```
conda create -n phyre36 python=3.6 
conda activate phyre36
pip install phyre opencv-python
cd phyre
python api_to_dataset.py --actions-per-task 8 --setup ball_within_template --output-dir "../../../../scratch/phyre_videos"
```


# Video-training added (only clevrer now)

* SWM added in repo
* `cfg.training_type` added -> choices ["video", "wm"]
* `swm.data.VideoStepsDataset` created
* To make it compatible to the `StepsDataest`, `{cfg.dataset_name}/{split}.json` is needed. (already pushed for clevrer)
* if `training_type == video` -> use `VideoStepsDataset`
* No action, proprio for video
* No action, proprio dimension for predictor if video
* action_encoder, video_encoder = None if video 
* Do not encode action, proprio of those encoders are None


# Todo
* caching / dataset-class is not optimized.. takes forever lol
* get_img_pipeline is not doing img resizing for some reason in my code