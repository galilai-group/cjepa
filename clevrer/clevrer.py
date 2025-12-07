import os
from pathlib import Path

import datasets
from torchcodec.decoders import VideoDecoder

import stable_worldmodel as swm


# path to json containing video metadata
# metadata_json = Path("/path/to/metadata.json")
# you can add the metadata of clevrer as extra columns, I leave it to you ;)

# expect path towards a single directory containing all .mp4 video files of clevrer
video_dir = Path("/cs/data/people/hnam16/data/clevrer/videos/")


target_name = "clevrer_train"  # train, val, test
records = {"episode_idx": [], "step_idx": [], "pixels": [], "episode_len": []}

# create dataset directory if it doesn't exist
# dataset_dir = swm.data.utils.get_cache_dir() / target_name
dataset_dir = Path('/cs/data/people/hnam16/.stable_worldmodel/clevrer_train')
os.makedirs(dataset_dir, exist_ok=True)

# make videos dataset
(dataset_dir / "videos").mkdir(exist_ok=True)

for ep_idx, video in enumerate(video_dir.iterdir()):
    if video.suffix != ".mp4":
        continue

    print(f"Processing video: {video.name}")

    video_name = f"videos/{ep_idx}_pixels.mp4"
    decoder = VideoDecoder(video.as_posix())
    num_frames = len(decoder)

    # extend dataset
    records["episode_idx"].extend([ep_idx] * num_frames)
    records["step_idx"].extend(list(range(num_frames)))
    records["episode_len"].extend([num_frames] * num_frames)
    records["pixels"].extend([video_name] * num_frames)

    # save video to dataset directory
    target_path = dataset_dir / video_name

    # copy video file
    if not target_path.exists():
        with open(video, "rb") as src_file:
            with open(target_path, "wb") as dst_file:
                dst_file.write(src_file.read())

# make dataset
dataset = datasets.Dataset.from_dict(records)
dataset.save_to_disk(dataset_dir.as_posix())

print(f"Dataset saved to {dataset_dir.as_posix()}")