import os
from pathlib import Path

import datasets
from torchcodec.decoders import VideoDecoder

import stable_worldmodel as swm

ROOT_DIR="your/path/to/data/clevrer"
DATASET='clevrer'
SAVE_DIR = swm.data.utils.get_cache_dir()
# or if you want to set your own,
# SAVE_DIR = Path("/path/to/your/dataset/directory")

def save_swm_dataset(dir, split):
    # expect path towards a single directory containing all .mp4 video files of clevrer
    video_dir = Path(dir) / split
    dataset_dir = SAVE_DIR / str(DATASET +f'_{split}')


    records = {"episode_idx": [], "step_idx": [], "pixels": [], "episode_len": []}
    os.makedirs(dataset_dir, exist_ok=True)

    # make videos dataset
    (dataset_dir / "videos").mkdir(exist_ok=True)

    for video in video_dir.iterdir():
        if video.suffix != ".mp4":
            continue

        print(f"Processing video: {video.name}")


        ep_idx = int(video.stem.split("_")[1])

        video_name = f"videos/{ep_idx}_pixels.mp4"
        decoder = VideoDecoder(video.as_posix())
        num_frames = len(decoder)
        # Convert to Python int to avoid numpy types
        num_frames = int(num_frames)

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


save_swm_dataset(ROOT_DIR, "train")
save_swm_dataset(ROOT_DIR, "val")
# save_swm_dataset(ROOT_DIR, "test")