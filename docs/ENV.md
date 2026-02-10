# 1. Environment Setup
```
conda create -n dino310 python=3.10 -y
conda activate dino310
conda install anaconda::ffmpeg
pip install seaborn webdataset swig einops uv torchcodec av

git clone https://github.com/galilai-group/stable-pretraining.git
cd stable-pretraining 
git checkout 92b5841
git clone https://github.com/galilai-group/stable-worldmodel.git
cd stable-worldmodel
git checkout 221ac82
uv pip install -e ./stable-pretraining 
uv pip install -e ./stable-worldmodel
uv pip install accelerate tensorboard tensorboardX hickle
```
to run ALOE for clevrer VQA, install `nerv` and s`pycocotools` as well.
```
cd thrid_party
git clone https://github.com/Wuziyi616/nerv.git
cd nerv
git checkout v0.1.0  # tested with v0.1.0 release
uv pip install -e .
pip install pycocotools
```