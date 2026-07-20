#!/usr/bin/env python3
"""Download CLEVRER videos and write stable-worldmodel HDF5 datasets."""

from __future__ import annotations

import argparse
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import av
import cv2
import h5py
import hdf5plugin
import numpy as np
from remotezip import RemoteZip
from tqdm import tqdm

SPLITS = {
    "train": "https://data.csail.mit.edu/clevrer/videos/train/video_train.zip",
    "val": "https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip",
    "test": "https://data.csail.mit.edu/clevrer/videos/test/video_test.zip",
}


def default_output_root() -> Path:
    root = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    return root / "datasets"


def download(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length", 0)) or None
        mode = "ab" if temporary.exists() else "wb"
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset:
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
            response.close()
            response = urllib.request.urlopen(request)
        with (
            temporary.open(mode) as file,
            tqdm(
                total=total,
                initial=offset,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            ) as progress,
        ):
            while chunk := response.read(8 * 1024 * 1024):
                file.write(chunk)
                progress.update(len(chunk))
    temporary.replace(destination)
    return destination


def decode_video(path: Path, size: int) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            image = frame.to_ndarray(format="rgb24")
            if image.shape[:2] != (size, size):
                image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(image)
    if not frames:
        raise RuntimeError(f"No frames were decoded from {path}")
    return np.stack(frames).astype(np.uint8, copy=False)


class ClevrerWriter:
    """Compressed writer that follows stable-worldmodel's HDF5 schema."""

    def __init__(self, path: Path, size: int, overwrite: bool):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w", libver="latest")
        compression = hdf5plugin.Blosc(
            cname="zstd",
            clevel=5,
            shuffle=hdf5plugin.Blosc.SHUFFLE,
        )
        self.file.create_dataset(
            "pixels",
            shape=(0, size, size, 3),
            maxshape=(None, size, size, 3),
            chunks=(16, size, size, 3),
            dtype=np.uint8,
            **compression,
        )
        for name in ("episode_idx", "step_idx"):
            self.file.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=(1024,),
                dtype=np.int64,
            )
        self.file.create_dataset(
            "ep_len",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=np.int32,
        )
        self.file.create_dataset(
            "ep_offset",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=np.int64,
        )

    def write(self, frames: np.ndarray, episode_id: int):
        episode = self.file["ep_len"].shape[0]
        offset = self.file["pixels"].shape[0]
        end = offset + len(frames)
        values = {
            "pixels": frames,
            # Keep CLEVRER's canonical scene id (0, 10000, 15000, ...).
            # ALOE question annotations use this id to join questions to slots.
            "episode_idx": np.full(len(frames), episode_id, dtype=np.int64),
            "step_idx": np.arange(len(frames), dtype=np.int64),
        }
        for name, data in values.items():
            self.file[name].resize(end, axis=0)
            self.file[name][offset:end] = data
        self.file["ep_len"].resize(episode + 1, axis=0)
        self.file["ep_len"][episode] = len(frames)
        self.file["ep_offset"].resize(episode + 1, axis=0)
        self.file["ep_offset"][episode] = offset
        self.file.flush()

    def close(self):
        self.file.close()


def video_names(archive) -> list[str]:
    return sorted(
        name
        for name in archive.namelist()
        if name.lower().endswith(".mp4") and not name.startswith("__MACOSX")
    )


def convert_split(args, split: str):
    output = args.output_root / f"clevrer_{split}.h5"
    writer = ClevrerWriter(output, args.size, args.overwrite)
    work_dir = args.work_dir or Path(tempfile.gettempdir()) / "cjepa-clevrer"
    work_dir.mkdir(parents=True, exist_ok=True)
    url = SPLITS[split]
    range_mode = args.download_mode == "range" or (
        args.download_mode == "auto" and args.max_videos is not None
    )

    archive = None
    try:
        if range_mode:
            archive = RemoteZip(url)
        else:
            zip_path = download(url, work_dir / f"{split}.zip")
            archive = zipfile.ZipFile(zip_path)

        names = video_names(archive)
        if args.max_videos is not None:
            names = names[: args.max_videos]
        if not names:
            raise RuntimeError(f"No MP4 videos found in the {split} archive")

        with tempfile.TemporaryDirectory(dir=work_dir) as temporary:
            temporary = Path(temporary)
            for name in tqdm(names, desc=f"Converting {split}"):
                archive.extract(name, path=temporary)
                video_path = temporary / name
                scene_id = int(video_path.stem.rsplit("_", 1)[-1])
                writer.write(decode_video(video_path, args.size), scene_id)
                video_path.unlink()
    finally:
        writer.close()
        if archive is not None:
            archive.close()
    print(f"Wrote {len(names)} episodes to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLITS),
        default=list(SPLITS),
    )
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--size", type=int, default=196)
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Convert only the first N videos per split (useful for smoke tests)",
    )
    parser.add_argument(
        "--download-mode",
        choices=("auto", "full", "range"),
        default="auto",
        help="Range mode fetches selected MP4 members without storing the full ZIP",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    if args.work_dir:
        args.work_dir = args.work_dir.expanduser().resolve()
    for split in args.splits:
        convert_split(args, split)


if __name__ == "__main__":
    main()
