#!/usr/bin/env python3
"""Add pre-extracted VideoSAUR slots to a stable-worldmodel HDF5 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import hdf5plugin
import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from encoder import build_object_encoder


def image_transform(size):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            v2.Resize((size, size)),
        ]
    )


def create_column(output, name, source, sample_shape, dtype):
    compression = {}
    if name in {"slots", "pixels"}:
        compression = hdf5plugin.Blosc(
            cname="zstd",
            clevel=5,
            shuffle=hdf5plugin.Blosc.SHUFFLE,
        )
    chunk_len = 16 if name == "pixels" else 128
    return output.create_dataset(
        name,
        shape=(0, *sample_shape),
        maxshape=(None, *sample_shape),
        chunks=(chunk_len, *sample_shape),
        dtype=dtype,
        **compression,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", choices=("clevrer", "pusht"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=196)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "encoder"
        / f"{args.dataset}.yaml"
    )
    encoder = build_object_encoder(
        dataset=args.dataset,
        config=config,
        checkpoint=args.checkpoint,
    ).to(args.device)
    transform = image_transform(args.image_size)

    with (
        h5py.File(args.input, "r") as source,
        h5py.File(args.output, "w", libver="latest") as output,
    ):
        lengths = source["ep_len"][:]
        offsets = source["ep_offset"][:]
        if args.max_episodes is not None:
            lengths = lengths[: args.max_episodes]
            offsets = offsets[: args.max_episodes]

        columns = [
            key for key in source.keys() if key not in {"pixels", "ep_len", "ep_offset"}
        ]
        created = {}
        global_offset = 0
        output_lengths = []
        output_offsets = []

        for length, offset in tqdm(
            list(zip(lengths, offsets, strict=True)), desc="Extracting slots"
        ):
            length, offset = int(length), int(offset)
            pixels = source["pixels"][offset : offset + length]
            frames = torch.from_numpy(pixels).permute(0, 3, 1, 2)
            frames = transform(frames).to(args.device)
            with torch.inference_mode():
                slots = encoder(frames.unsqueeze(0))[0].float().cpu().numpy()

            values = {"slots": slots}
            for key in columns:
                values[key] = source[key][offset : offset + length]

            if not created:
                for key, value in values.items():
                    created[key] = create_column(
                        output, key, source, value.shape[1:], value.dtype
                    )
            end = global_offset + length
            for key, value in values.items():
                created[key].resize(end, axis=0)
                created[key][global_offset:end] = value
            output_lengths.append(length)
            output_offsets.append(global_offset)
            global_offset = end

        output.create_dataset("ep_len", data=np.asarray(output_lengths, np.int32))
        output.create_dataset("ep_offset", data=np.asarray(output_offsets, np.int64))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
