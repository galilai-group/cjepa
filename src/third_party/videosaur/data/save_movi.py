import argparse
import os
import os.path

import numpy as np
import tensorflow_datasets as tfds
import tqdm
import webdataset as wds

parser = argparse.ArgumentParser("""Generate sharded dataset from original MOVI data.""")
parser.add_argument(
    "--split", default="train", choices=["train", "validation"], help="Which splits to write"
)
parser.add_argument("--level", default="e")
parser.add_argument("--version", default="1.0.0")
parser.add_argument("--maxcount", type=int, default=32, help="Max number of samples per shard")
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument("--only-video", action="store_true", help="Whether to store only videos")
parser.add_argument(
    "out_path",
    help="Directory where shards are written",
)


def write_dataset(args):
    split = args.split
    ds, ds_info = tfds.load(
        f"movi_{args.level}/{args.image_size}x{args.image_size}:{args.version}",
        data_dir="gs://kubric-public/tfds",
        with_info=True,
    )
    print(ds_info)
    train_iter = iter(tfds.as_numpy(ds[args.split]))

    # This is the output pattern under which we write shards.
    pattern = os.path.join(args.out_path, f"movi_{args.level}-{split}-%06d.tar")

    if args.only_video:
        keys_to_save = ["video"]
    else:
        keys_to_save = ["video", "segmentations", "forward_flow"]

    print("Storing properties:", *keys_to_save)

    with wds.ShardWriter(pattern, maxcount=args.maxcount) as sink:
        for ind, record in enumerate(tqdm.tqdm(train_iter)):
            # Construct a sample.
            sample = {"__key__": str(ind)}
            for k in keys_to_save:
                if k == "forward_flow":
                    out = record[k].astype(np.float32)
                else:
                    out = record[k]
                sample[f"{k}.npy"] = out
            # Write the sample to the sharded tar archives.
            sink.write(sample)


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.isdir(os.path.join(args.out_path, ".")):
        os.makedirs(os.path.join(args.out_path, "."), exist_ok=True)
    write_dataset(args)
