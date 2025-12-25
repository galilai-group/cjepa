import abc
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import webdataset as wds

from videosaur.videosaur.data import transforms
from videosaur.videosaur.data.utils import remap_dict
from videosaur.videosaur.utils import config_as_kwargs


def build(config, name: Optional[str] = "VideoPipeline", **kwargs):
    name = config.get("name") or name
    if name in ("video", "VideoPipeline"):
        tfs = transforms.build(config.transforms) if config.transforms else None
        # num_frameskip is only supported at the pipeline level (e.g. train_pipeline.num_frameskip)
        # num_frameskip = config.get("num_frameskip", None)

        pipeline = VideoPipeline(
            transforms=tfs,
            # num_frameskip=int(num_frameskip) if num_frameskip is not None else None,
            **config_as_kwargs(config, to_filter=("transforms",), defaults=kwargs), # it includes num_frameskip
        )
    elif name in ("image", "ImagePipeline"):
        tfs = transforms.build(config.transforms) if config.transforms else None
        pipeline = ImagePipeline(
            transforms=tfs,
            **config_as_kwargs(config, to_filter=("transforms",), defaults=kwargs),
        )
    else:
        raise ValueError(f"Unknown pipeline `{name}`")

    return pipeline


class DataPipeline(abc.ABC):
    """Abstract base class for data pipelines.

    A data pipeline defines how a stream of raw data samples is transformed to the stream of
    output samples.
    """

    def __init__(self, keys: Optional[Tuple[str]]):
        self._keys = keys

    @property
    def keys(self) -> Optional[Tuple[str]]:
        """Keys of properties to keep in dataset after filtering."""
        return self._keys

    @abc.abstractmethod
    def get_num_samples(self, num_orig_samples: int) -> Optional[int]:
        """Number of samples after pipeline is applied, given original number of samples in dataset.

        If number can not be computed, may return `None`.
        """
        ...

    @abc.abstractmethod
    def apply(self, dataset: wds.WebDataset) -> wds.WebDataset:
        """Apply pipeline to dataset.

        Input dataset contains dicts of samples after decoding.
        """
        ...


class ImagePipeline(DataPipeline):
    def __init__(
        self,
        transforms: Dict[str, Optional[Callable]] = None,
        keys: Optional[Tuple[str]] = None,
        is_video_dataset: bool = True,
        video_size: Optional[int] = None,
        one_frame_per_video: bool = False,
        shuffle: bool = False,
        shuffle_size: int = 1000,
        duplicate: Optional[Dict[str, str]] = None,
    ):
        super().__init__(keys)
        self.transforms = transforms
        self.video_size = video_size
        self.shuffle = shuffle
        self.shuffle_size = shuffle_size
        self.is_video_dataset = is_video_dataset
        self.one_frame_per_video = one_frame_per_video
        self.duplicate = duplicate

    def get_num_samples(self, num_orig_samples: int) -> Optional[int]:
        if self.video_size is not None:
            return num_orig_samples * self.video_size
        else:
            return None  # Can not provide the number of samples

    def apply(self, dataset: wds.WebDataset) -> wds.WebDataset:
        if self.is_video_dataset:
            split_fn = partial(
                split_to_chunks,
                keys_to_split=self.keys,
                chunk_size=1,
                shuffle=self.shuffle,
                one_chunk_per_video=self.one_frame_per_video,
            )
            dataset = dataset.compose(split_fn)
            rename_video = partial(remap_dict, rename_dict={"video": "image"})
            dataset.map(rename_video)

        if self.shuffle:
            # First chunking, then shuffling
            dataset = dataset.shuffle(self.shuffle_size)

        if self.duplicate:
            dataset.map(partial(copy_dict_entries, copy_from_to=self.duplicate))

        if self.transforms is not None:
            dataset = dataset.map_dict(**self.transforms)

        return dataset


class VideoPipeline(DataPipeline):
    def __init__(
        self,
        transforms: Dict[str, Optional[Callable]] = None,
        keys: Optional[Tuple[str]] = None,
        video_size: Optional[int] = None,
        chunk_size: int = 6,
        use_chunks: bool = True,
        sample_one_chunk_per_video: bool = False,
        shuffle: bool = False,
        shuffle_size: int = 100,
        duplicate: Optional[Dict[str, str]] = None,
        num_frameskip: Optional[int] = None,
    ):
        super().__init__(keys)
        self.transforms = transforms
        self.video_size = video_size
        self.chunk_size = chunk_size
        self.use_chunks = use_chunks
        if sample_one_chunk_per_video and not use_chunks:
            raise ValueError("For sampling one chunk per video, use_chunks needs to be True")
        self.sample_one_chunk_per_video = sample_one_chunk_per_video
        self.shuffle = shuffle
        self.shuffle_size = shuffle_size
        self.duplicate = duplicate
        self.num_frameskip = num_frameskip

    def get_num_samples(self, num_orig_samples: int) -> Optional[int]:
        if self.use_chunks:
            if self.sample_one_chunk_per_video:
                return num_orig_samples
            else:
                if self.video_size is not None:
                    return num_orig_samples * self.video_size // self.chunk_size
                else:
                    return None  # Can not provide the number of samples
        else:
            return num_orig_samples

    def apply(self, dataset: wds.WebDataset) -> wds.WebDataset:
        if self.use_chunks:
            # Apply temporal subsampling BEFORE splitting into chunks if requested.
            if self.num_frameskip is not None and int(self.num_frameskip) > 1:
                dataset = dataset.map(partial(temporal_subsample_keys, keys_to_subsample=self.keys, stride=int(self.num_frameskip)))
            # If sampling chunks, need to shuffle as well to pick a random chunk
            shuffle = self.shuffle or self.sample_one_chunk_per_video
            split_fn = partial(
                split_to_chunks,
                keys_to_split=self.keys,
                shuffle=shuffle,
                one_chunk_per_video=self.sample_one_chunk_per_video,
                chunk_size=self.chunk_size,
            )
            dataset = dataset.compose(split_fn)

        if self.shuffle:
            # First chunking, then shuffling
            dataset = dataset.shuffle(self.shuffle_size)

        if self.duplicate:
            dataset.map(partial(copy_dict_entries, copy_from_to=self.duplicate))

        if self.transforms is not None:
            dataset = dataset.map_dict(**self.transforms)

        return dataset


def split_to_chunks(
    data,
    keys_to_split: Tuple[str],
    chunk_size: int,
    shuffle: bool,
    one_chunk_per_video: bool,
    axis: int = 0,
):
    """Split video to chunks with chunk_size size."""
    for sample in data:
        key = sample["__key__"]
        video_size = sample[keys_to_split[0]].shape[0]

        num_chunks = video_size // chunk_size

        data_chunks = [
            np.array_split(
                sample[key],
                range(chunk_size, video_size, chunk_size),
                axis=axis,
            )[:num_chunks]
            for key in keys_to_split
        ]

        if shuffle:
            chunks_ids = np.random.permutation(range(num_chunks))
        else:
            chunks_ids = list(range(num_chunks))

        if one_chunk_per_video:
            chunks_ids = chunks_ids[:1]

        for chunk_id in chunks_ids:
            chunked_data = {
                key: data_key[chunk_id] for key, data_key in zip(keys_to_split, data_chunks)
            }
            for key, value in sample.items():
                if key in keys_to_split:
                    continue
                # Data that is not splitted is just repeated across all chunks
                chunked_data[key] = value
            chunked_data["__key__"] = f"{key}_{chunk_id}"
            yield chunked_data


def copy_dict_entries(dictionary: Dict[str, Any], copy_from_to: Dict[str, str]) -> Dict[str, Any]:
    for from_key, to_key in copy_from_to.items():
        dictionary[to_key] = dictionary[from_key]

    return dictionary


def temporal_subsample_keys(dictionary: Dict[str, Any], keys_to_subsample: Tuple[str], stride: int = 1) -> Dict[str, Any]:
    """Subsample temporal axis (first axis) for the given keys in the sample dict.

    Expects numpy arrays with shape (F, ...). Returns the modified dictionary.
    """
    for key in keys_to_subsample:
        if key in dictionary and isinstance(dictionary[key], (list, tuple, np.ndarray)):
            arr = dictionary[key]
            try:
                dictionary[key] = arr[0:: stride]
            except Exception:
                # If not sliceable, keep original
                pass

    return dictionary
