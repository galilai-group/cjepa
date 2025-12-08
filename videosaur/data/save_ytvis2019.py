import argparse
import os
import os.path

# import subprocess
import einops
import numpy as np
import tensorflow_datasets as tfds
import torch
import tqdm
import webdataset as wds
from utils import get_default_dir

from videosaur.data.transforms import Resize

parser = argparse.ArgumentParser(
    """Generate sharded dataset from tensorflow datasets YouTube VIS data."""
)
parser.add_argument(
    "--split", default="train", choices=["train", "validation"], help="Which splits to write"
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
    default=get_default_dir("ytvis2019_resized"),
    help="Directory where shards are written",
)
parser.add_argument(
    "--download-dir",
    default=get_default_dir("ytvis2019_raw"),
    help="Directory where YTVIS 2019 videos and annotations are downloaded to",
)

# Download is currently not supported
#  because of the networks error during gdrive downloads
# Folow intstuctions here
# https://www.tensorflow.org/datasets/catalog/youtube_vis
# for manual download

# GDRIVE_IDS = {
#     # "valid.json": "106ozt8JJyQWI1RO8Uh18aOBqngznUhir",
#     # "test.json": "1GwkyVcc6wVvpPmnMWHa5yLcJ2jJ67vNW",
#     # "train.json": "1OgdUC29rPN3jAcM6DjpORv6AlNEpVtGc",
#     # "train_all_frames.zip": "1eRO0rQjO6gEh0T9Ua5REfbreFCNY7cfq",
#     # "valid_all_frames.zip": "1rWQzZcMskgpEQOZdJPJ7eTmLCBEIIpEN",
#     # "test_all_frames.zip": "1WAc-AyR2vQ6UpOv8iw1p1MoBry-D8Rav",
# }

# def download_dataset(download_dir):
#     for file_name, gdrive_id in GDRIVE_IDS.items():
#         print(f"Done for {file_name}")
#         subprocess.run(
#             [
#                 "bash",
#                 "/is/sg2/azadaianchuk/projects/videosaur/data/gdrive_download.sh",
#                 gdrive_id,
#                 f"{download_dir}/{file_name}",
#             ]
#         )


def write_dataset(args):
    split = args.split
    ds, _ = tfds.load(
        "youtube_vis/only_frames_with_labels_train_split",
        data_dir=args.download_dir,
        with_info=True,
    )
    dataset_iter = iter(tfds.as_numpy(ds[args.split]))

    # This is the output pattern under which we write shards.
    pattern = os.path.join(args.out_path, f"ytvis-{split}-%06d.tar")
    if args.resize:
        assert args.resize_mode in [
            "nearest-exact"
        ], "Not sure how downscaling works with other modes."
        resize = Resize(
            args.resize_size, args.resize_mode, clamp_zero_one=False, short_side_scale=True
        )
    with wds.ShardWriter(pattern, maxcount=args.maxcount) as sink:
        for ind, record in enumerate(tqdm.tqdm(dataset_iter)):
            # Construct a sample.
            sample = {"__key__": str(ind)}
            video = record["video"]
            if args.resize:
                video = torch.from_numpy(record["video"])
                video = einops.rearrange(video, "f h w c -> f c h w")
                video = resize(video).cpu().numpy()
                video = einops.rearrange(video, "f c h w -> f h w c")
            sample["video.npy"] = video
            if not args.only_videos:
                length, h, w, _ = record["video"].shape

                n_obj = len(record["tracks"]["segmentations"].numpy())
                if n_obj == 0:
                    out = np.zeros(shape=(length, h, w, 1), dtype=np.uint8)
                else:
                    out = np.zeros(shape=(length, h, w, n_obj), dtype=np.uint8)
                    for i, (frame_id, cat_id, seg) in enumerate(
                        zip(
                            record["tracks"]["frames"],
                            record["tracks"]["category"],
                            record["tracks"]["segmentations"],
                        )
                    ):
                        out[frame_id.numpy(), :, :, i : i + 1] = seg.numpy() * cat_id
                    assert (
                        np.unique(out)
                        == np.unique(np.asarray(list(record["tracks"]["category"]) + [0]))
                    ).all()
                if args.resize:
                    out = torch.from_numpy(out)
                    out = einops.rearrange(out, "f h w c -> f c h w")
                    out = resize(out).cpu().numpy()
                    out = einops.rearrange(out, "f c h w -> f h w c")
                sample["segmentations.npy"] = out
                assert out.shape[:-1] == video.shape[:-1]
            # Write the sample to the sharded tar archives.
            sink.write(sample)


if __name__ == "__main__":
    args = parser.parse_args()
    # if not os.path.isdir(args.download_dir):
    #     os.makedirs(args.download_dir, exist_ok=True)
    # download_dataset(args.download_dir)
    if not os.path.isdir(os.path.join(args.out_path, ".")):
        os.makedirs(os.path.join(args.out_path, "."), exist_ok=True)
    write_dataset(args)
