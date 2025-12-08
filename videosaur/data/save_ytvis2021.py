"""Script to download and shard YouTube VIS dataset."""
import argparse
import os

import einops
import numpy as np
import torch
import tqdm
import webdataset as wds
from einops import rearrange
from PIL import Image
from utils import get_default_dir
from ytvis import YTVOS as YTVIS

from videosaur.data.transforms import Resize
from videosaur.data.transforms_video import FromTensorVideo, ToTensorVideo

# TODO: add download from google drive
# for now use gdrive_download script with IDs above.
# TRAIN_GDRIVE_ID = "1fLO_d8ys_03CPPpCm-lL5SgCJsbNN5L9"
# VAL_GDRIVE_ID = "1Z0VEOQp4d0-aiZNDt8Hd5KdcCU4s8JaY"

SPLITS_TO_SUFFIX = {
    "train": "train",
    "validation": "valid",
}
SPLITS_TO_INSTANCES_FILENAME = {
    "train": "instances.json",
    "validation": "instances.json",
}


parser = argparse.ArgumentParser("Generate sharded dataset from original YouTube-VIS data.")
parser.add_argument(
    "--split", default="train", choices=list(SPLITS_TO_SUFFIX), help="Which splits to write"
)
parser.add_argument(
    "--download-dir",
    default=get_default_dir("ytvis2021_raw"),
    help="Directory where YTVIS 2021 videos and annotations are downloaded to",
)
parser.add_argument("--maxcount", type=int, default=32, help="Max number of samples per shard")
parser.add_argument("--only-videos", action="store_true", help="Whether to store only videos")
parser.add_argument("--resize", action="store_true", help="Whether to resize videos")
parser.add_argument(
    "--resize_size", type=int, default=320, help="The size of min(H,W) after resizing"
)
parser.add_argument("--resize_mode", type=str, default="nearest-exact", help="Resize mode")
parser.add_argument(
    "--out-path",
    default=get_default_dir("ytvis2021_resized2"),
    help="Directory where shards are written",
)


def write_dataset(video_dir, out_dir, split, add_annotations=True, annotations_file=None):
    if add_annotations and annotations_file is None:
        raise ValueError("Requested to add annotations but annotations file not given")

    if not os.path.isdir(os.path.join(out_dir, ".")):
        os.makedirs(os.path.join(out_dir, "."), exist_ok=True)

    # This is the output pattern under which we write shards.
    pattern = os.path.join(out_dir, f"ytvis-{split}-%06d.tar")
    if args.resize:
        clamp_zero_one = args.resize_mode == "bicubic"
        resize = Resize(
            args.resize_size, args.resize_mode, clamp_zero_one=clamp_zero_one, short_side_scale=True
        )
        segmentation_resize = Resize(
            args.resize_size, "nearest-exact", clamp_zero_one=False, short_side_scale=True
        )
        to_video = ToTensorVideo()
        from_video = FromTensorVideo()
    ytvis = YTVIS(annotations_file)
    all_ids = ytvis.getVidIds()
    val_ids = all_ids[-300:]
    train_ids = all_ids[:-300]
    ids = train_ids if split == "train" else val_ids
    print(f"Saving {len(ids)} from {len(all_ids)} as a {split} part of the datasets.")
    with wds.ShardWriter(pattern, maxcount=args.maxcount) as sink:
        max_num_instances = 0
        for vid_id in tqdm.tqdm(ids):
            vid = ytvis.vids[vid_id]
            length = int(vid["length"])
            height = int(vid["height"])
            width = int(vid["width"])
            video = []

            for _, file_name in enumerate(vid["file_names"]):
                frame_path = os.path.join(video_dir, file_name)
                with Image.open(frame_path) as img:
                    img = np.asarray(img, dtype="uint8")
                video.append(img[None])
            video = np.concatenate(video)
            # video shape should be [B, F, H, W, 3]
            assert video.shape == (length, height, width, 3)
            if args.resize:
                video = to_video(torch.from_numpy(video))
                video = from_video(resize(video)).cpu().numpy()

            if not add_annotations:
                sample = {
                    "__key__": str(vid_id),
                    "video.npy": video,
                }
            else:

                segmentations = []
                ann_ids = ytvis.getAnnIds(vidIds=[vid_id], iscrowd=False)
                anns = ytvis.loadAnns(ann_ids)
                if anns:
                    for f_id, _ in enumerate(vid["file_names"]):
                        seg = []
                        num_obj = len(anns)
                        max_num_instances = max(max_num_instances, num_obj)
                        for ann in anns:
                            try:
                                mask = ytvis.annToMask(ann, f_id) * ann["category_id"]
                            except TypeError:
                                # TODO: check if this is the right way to do this.
                                # e.g. that error actually means that instance
                                # is not visible in the image
                                mask = np.zeros(
                                    (int(vid["height"]), int(vid["width"])), dtype=np.uint8
                                )
                            seg.append(mask)
                        segmentations.append(seg)
                    segmentations = np.asarray(segmentations, dtype=np.uint8)
                    segmentations = rearrange(segmentations, "f i h w -> f h w i")

                    # segmentation shape should be [B, F, H, W, 1]
                    assert segmentations.shape == (
                        length,
                        height,
                        width,
                        num_obj,
                    )
                else:
                    segmentations = np.zeros(
                        (length, int(vid["height"]), int(vid["width"]), 1), dtype=np.uint8
                    )
                if args.resize:
                    segmentations = torch.from_numpy(segmentations)
                    segmentations = einops.rearrange(segmentations, "f h w c -> f c h w")
                    segmentations = segmentation_resize(segmentations).cpu().numpy()
                    segmentations = einops.rearrange(segmentations, "f c h w -> f h w c")
                sample = {
                    "__key__": str(vid_id),
                    "video.npy": video,
                    "segmentations.npy": segmentations,
                }
            sink.write(sample)

    if add_annotations:
        print(f"Maximal number of instances per video in {split} split is: {max_num_instances}.")


if __name__ == "__main__":
    args = parser.parse_args()
    root = os.path.join(args.download_dir, "train")
    annotations_file = os.path.join(root, "instances.json")
    video_dir = os.path.join(root, "JPEGImages")
    write_dataset(
        video_dir=video_dir,
        out_dir=args.out_path,
        split=args.split,
        add_annotations=annotations_file is not None and not args.only_videos,
        annotations_file=annotations_file,
    )
