import argparse
import os
import os.path
import zipfile
from pathlib import Path

import numpy as np
import tqdm
import webdataset as wds
from PIL import Image
from utils import download_file, get_default_dir

parser = argparse.ArgumentParser("""Generate sharded dataset from original DAVIS data.""")
parser.add_argument(
    "--split", default="train", choices=["train", "validation"], help="Which splits to write"
)
parser.add_argument("--root", default=get_default_dir("davis_raw"))
parser.add_argument("--maxcount", type=int, default=2, help="Max number of samples per shard")
parser.add_argument("--only-video", action="store_true", help="Whether to store only videos")
parser.add_argument(
    "--out_path",
    default=get_default_dir("davis"),
    help="Directory where shards are written",
)


def listdirs(rootdir):
    dirs = []
    for path in Path(rootdir).iterdir():
        if path.is_dir():
            dir_name = str(path)
            folder_name = dir_name.split("/")[-1]
            dirs.append((dir_name, folder_name))
    return dirs


def get_split_dirs(split_file):
    with open(split_file) as f:
        split = [line.rstrip() for line in f]
    return split


def get_davis(dir):
    # url = "https://data.vision.ee.ethz.ch/jpont/davis"
    # file_trainval = "DAVIS-2017-trainval-480p.zip"
    # should be more consisent for unsupervised object segmentation
    url = "https://data.vision.ee.ethz.ch/csergi/share/davis"
    file_trainval = "DAVIS-2017-Unsupervised-trainval-480p.zip"
    data_dir = os.path.join(dir, "train_val")
    os.makedirs(data_dir, exist_ok=True)

    file_local = os.path.join(data_dir, file_trainval)
    if not os.path.isfile(file_local):
        print("Downloading DAVIS 2017 (train-val)...")
        download_file(f"{url}/{file_trainval}", dest_dir=data_dir)
        with zipfile.ZipFile(file_local, "r") as zip_ref:
            zip_ref.extractall(data_dir)
    else:
        print("Found DAVIS 2017 (train-val) file. Assume that dataset is already extracted.")


def make_dataset(root):
    root = os.path.join(root, "train_val/DAVIS/")
    video_root = os.path.join(root, "JPEGImages/480p/")
    segmentation_root = os.path.join(root, "Annotations_unsupervised/480p/")
    train_split = get_split_dirs(os.path.join(root, "ImageSets/2017/train.txt"))
    val_split = get_split_dirs(os.path.join(root, "ImageSets/2017/val.txt"))

    train_data, val_data = [], []
    train_pathes, val_pathes = [], []
    for video_folder, target in listdirs(video_root):
        segmentation_folder = os.path.join(segmentation_root, target)
        if not os.path.isdir(video_folder):
            continue
        for root, _, fnames in sorted(os.walk(video_folder)):
            video_frame_pathes = []
            video_frames = []
            segmentation_frames = []
            for fname in sorted(fnames):
                video_frame_path = os.path.join(root, fname)
                segmentation_frame_path = os.path.join(segmentation_folder, f"{fname[:-4]}.png")
                with Image.open(video_frame_path) as img:
                    video_frame = np.asarray(img)
                with Image.open(segmentation_frame_path) as img:
                    segmentation_frame = np.asarray(img)
                video_frames.append(video_frame[None])
                segmentation_frames.append(segmentation_frame[None])
                item = (video_frame_path, segmentation_frame_path)
                video_frame_pathes.append(item)
        # video shape should be [B, F, H, W, 3]
        # segmentation shape should be [B, F, H, W, 1]
        video_frames = np.concatenate(video_frames)
        segmentation_frames = np.concatenate(segmentation_frames)[..., None]
        if target in train_split:
            train_data.append({"video": video_frames, "segmentations": segmentation_frames})
            train_pathes.append(video_frame_pathes)
        elif target in val_split:
            val_data.append({"video": video_frames, "segmentations": segmentation_frames})
            val_pathes.append(video_frame_pathes)
        else:
            raise ValueError(f"{target} in not in splits")
    return train_data, train_pathes, val_data, val_pathes


def write_dataset(data, split, out_path, only_video=False):

    if not os.path.isdir(os.path.join(out_path, ".")):
        os.makedirs(os.path.join(out_path, "."), exist_ok=True)
    # This is the output pattern under which we write shards.
    pattern = os.path.join(out_path, f"davis-{split}-%06d.tar")

    if only_video:
        keys_to_save = ["video"]
    else:
        keys_to_save = ["video", "segmentations"]

    print("Storing properties:", *keys_to_save)

    # with wds.TarWriter(pattern) as sink:
    with wds.ShardWriter(pattern, maxcount=args.maxcount) as sink:
        for ind, record in enumerate(tqdm.tqdm(data)):
            # Construct a sample.
            sample = {"__key__": str(ind)}
            for k in keys_to_save:
                sample[f"{k}.npy"] = record[k]
            # Write the sample to the sharded tar archives.
            sink.write(sample)


if __name__ == "__main__":
    args = parser.parse_args()
    get_davis(args.root)
    train_data, train_pathes, val_data, val_pathes = make_dataset(args.root)

    print("DAVIS dataset loaded")
    write_dataset(
        train_data,
        "train",
        out_path=args.out_path,
        only_video=args.only_video,
    )
    write_dataset(
        val_data,
        "validation",
        out_path=args.out_path,
        only_video=args.only_video,
    )
