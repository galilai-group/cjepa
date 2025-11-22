# Env
```
conda create -n dino310 python=3.10 -y
conda activate dino310
pip install -e stable-pretraining
pip install einops opencv-python imageio
```

# Video-training added

* SWM added in repo
* `cfg.training_type` added -> choices ["video", "wm"]
* `swm.data.VideoStepsDataset` created
* To make it compatible to the `StepsDataest`, `{cfg.dataset_name}/{split}.json` is needed. (already pushed for clevrer)
* if `training_type == video` -> use `VideoStepsDataset`
* No action, proprio for video
* No action, proprio dimension for predictor if video
* action_encoder, video_encoder = None if video 
* Do not encode action, proprio of those encoders are None

# Question

* why history is not used? I guess n_step should be history +1 ?
* wandb not working with different config

# Todo
* caching / dataset-class is not optimized.. takes forever lol