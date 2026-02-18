# Install

We recommend using conda to set up the environment.

### 1. Create and activate conda environment
```
conda create -n cjepa python=3.10 -y
conda activate cjepa
git clone https://github.com/galilai-group/cjepa.git
cd cjepa
```


### 2. Install system dependencies

We use ffmpeg for video processing:
```
conda install anaconda::ffmpeg
```


### 3. Install basic Python dependencies
We recommend using `uv` to install dependencies. You can also use `pip` if you prefer.
```
pip install uv
uv pip install seaborn webdataset swig einops torchcodec av accelerate tensorboard tensorboardX hickle pycocotools wget
```

### 4. Install third-party libraries
* Every third party library should be installed under `src/third_party`. Please follow the instruction to install the library.
* Below will install `stable-pretraining`, `stable-worldmodel` and `nerv`.
* All third-party repositories are installed in editable mode to ensure smooth development.
* Specific commits/tags are pinned for reproducibility.
* The environment is tested with Python 3.10.



```
cd src/third_party
git clone https://github.com/galilai-group/stable-pretraining.git
cd stable-pretraining
git checkout 92b5841
uv pip install -e .

cd ..
git clone https://github.com/galilai-group/stable-worldmodel.git
cd stable-worldmodel
git checkout 221ac82
uv pip install -e .


cd ../
git clone https://github.com/Wuziyi616/nerv.git
cd nerv
git checkout v0.1.0   # tested with v0.1.0 release
uv pip install -e .
```
