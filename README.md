conda create -n dino310 python=3.10 -y
conda activate dino310
pip install -e stable-pretraining
pip install einops opencv-python imageio

add swm in repo
added cfg.training_type -> choices "video", "wm"
creat swm.data.VideoStepsDataset
to make it compatible to StepsDataest, {cfg.dataset_name}/{split}.json is needed (pushed for clevrer)
if training_type = video -> use VideoStepsDataset
no action, proprio for video
no action, proprio dimension for predictor if video
action_encoder, video_encoder = None if video 
Do not encode action, proprio of those encoders are None
Only pixel encoder included

# question
why history is not used? I guess n_step should be history +1 ?
wandb not working with different config
caching / dataset-class is not optimized.. takes forever lol