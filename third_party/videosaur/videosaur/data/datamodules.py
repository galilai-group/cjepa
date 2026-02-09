import math
import os
import tempfile
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import webdataset as wds
from omegaconf import ListConfig
from torch.utils.data._utils import collate as torch_collate

from third_party.videosaur.videosaur.data import pipelines, transforms
from third_party.videosaur.videosaur.data.utils import get_data_root_dir, worker_init_function
from third_party.videosaur.videosaur.utils import config_as_kwargs


def build(config, name: Optional[str] = "WebdatasetDataModule", data_dir: Optional[str] = None):
    name = config.get("name") or name
    if name == "WebdatasetDataModule":
        train_pipeline = None
        if config.train_pipeline:
            train_pipeline = pipelines.build(config.train_pipeline, shuffle=True)

        val_pipeline = None
        if config.val_pipeline:
            val_pipeline = pipelines.build(config.val_pipeline, shuffle=False)

        return WebdatasetDataModule(
            data_dir=data_dir,
            train_pipeline=train_pipeline,
            val_pipeline=val_pipeline,
            **config_as_kwargs(config, to_filter=("train_pipeline", "val_pipeline")),
        )
    elif name == "DummyDataModule":
        return DummyDataModule(
            train_transforms=transforms.build(config.train_transforms),
            val_transforms=transforms.build(config.val_transforms),
            **config_as_kwargs(
                config,
                to_filter=(
                    "train_transforms",
                    "val_transforms",
                ),
            ),
        )
    else:
        raise ValueError(f"Unknown dataset module `{name}`")


class WebdatasetDataModule(pl.LightningDataModule):
    """DatasetModule for webdataset datasets.

    We primarily rely on iteration-based instead of epoch-based training. Epoch-based training is
    difficult to realize with distributed training (i.e. multi-GPU), because it is hard to ensure
    that each sample is sampled exactly once per epoch. Instead, iteration-based training instead
    just samples a random stream of data. That said, this module does also support epoch-based
    training by setting the `samples_per_epoch` argument. In this case, the dataloader stops after
    `samples_per_epoch // (batch_size * num_nodes)` samples.

    For validation/testing, we need to make sure that each sample is seen exactly once. To do so,
    this module adds padding entries that should be ignored using the "batch_padding_mask" key.
    With distributed training, it is required to specify the number of samples the dataset contains
    using the `val_size`, `test_size` arguments.

    The arguments `val_size` and `test_size` refer to the total number of input samples contained
    in all shards of the split. As the number of samples can be changed by the data pipeline, it is
    the responsibility of the data pipeline to correctly specify how many samples it will output
    using the `get_num_samples` method.
    """

    BATCH_PADDING_MASK_KEY = "batch_padding_mask"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        train_shards: Optional[Union[str, List[str]]] = None,
        val_shards: Optional[Union[str, List[str]]] = None,
        test_shards: Optional[Union[str, List[str]]] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        samples_per_epoch: Optional[int] = None,
        train_pipeline: Optional[pipelines.DataPipeline] = None,
        val_pipeline: Optional[pipelines.DataPipeline] = None,
        test_pipeline: Optional[pipelines.DataPipeline] = None,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        num_val_workers: Optional[int] = None,
        cache_train: bool = False,
        cache_val: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        data_dir = data_dir if data_dir else get_data_root_dir()

        def get_shards_and_num_shards(shards):
            if shards is None:
                return None, 0
            if isinstance(shards, ListConfig):
                new_shards = []
                for s in shards:
                    new_shards.extend(get_shards_and_num_shards(s)[0])
                shards = new_shards
            else:
                shards = _to_abs_shard_path(shards, data_dir)
                shards = wds.shardlists.expand_urls(shards)
            return shards, len(shards)

        self.train_shards, self.num_train_shards = get_shards_and_num_shards(train_shards)
        self.val_shards, self.num_val_shards = get_shards_and_num_shards(val_shards)
        self.test_shards, self.num_test_shards = get_shards_and_num_shards(test_shards)
        self.val_size = val_size
        self.test_size = test_size
        self.samples_per_epoch = samples_per_epoch
        self.train_pipeline = train_pipeline
        self.val_pipeline = val_pipeline
        self.test_pipeline = test_pipeline if test_pipeline else val_pipeline
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_train_workers = num_workers
        self.num_val_workers = num_val_workers if num_val_workers is not None else num_workers

        if cache_dir is None and (cache_train or cache_val):
            cache_dir = tempfile.mkdtemp(prefix="wds_shardcache_", dir="/tmp")
        self.cache_dir = cache_dir
        self.cache_train = cache_train
        self.cache_val = cache_val

        self.num_nodes = None  # Set lazily

    def __str__(self) -> str:
        val_size = "?" if self.val_size is None else self.val_size
        test_size = "?" if self.test_size is None else self.test_size
        samples_per_epoch = (
            "unspecified" if self.samples_per_epoch is None else self.samples_per_epoch
        )
        res = [
            "WebdatasetDataModule",
            f"  - Number of train shards: {self.num_train_shards}",
            f"  - Number of val shards: {self.num_val_shards}",
            f"  - Number of test shards: {self.num_test_shards}",
            f"  - Assumed number of val samples: {val_size}",
            f"  - Assumed number of test samples: {test_size}",
            f"  - Specified length of training epoch: {samples_per_epoch}",
            f"  - Train batch size: {self.batch_size}",
            f"  - Eval batch size: {self.val_batch_size}",
            f"  - Number of train workers: {self.num_train_workers}",
            f"  - Number of eval workers: {self.num_val_workers}",
        ]
        return "\n".join(res)

    def _verify_settings_lazy(self):
        """Check that we have appropriate settings for the distributed setting.

        We can only do this lazily (once a dataloader is requested) because on __init__ the
        number of nodes is not known.
        """
        if self.num_nodes is not None:
            return

        self.num_nodes = wds.utils.pytorch_worker_info()[1]

        if self.num_nodes > 1:
            if self.val_shards and self.val_size is None:
                raise ValueError("Need to specify `val_size` in distributed setting")
            if self.test_shards and self.test_size is None:
                raise ValueError("Need to specify `test_size` in distributed setting")

        def _check_workers_and_shards(split: str, num_workers: int, num_shards: int) -> int:
            min_shards_per_node = num_shards // self.num_nodes

            if min_shards_per_node == 0:
                raise ValueError(
                    f"The number of compute nodes is {self.num_nodes}, but the "
                    f"number of {split} shards is only {num_shards}. Increase the number of shards."
                )

            if num_workers > min_shards_per_node:
                raise ValueError(
                    f"The number of {split} workers is {num_workers}, but the minimum number of "
                    f"shards per compute node is only {min_shards_per_node}. Reduce the number of "
                    f"workers to <={min_shards_per_node}."
                )

        if self.train_shards:
            _check_workers_and_shards("train", self.num_train_workers, self.num_train_shards)
        if self.val_shards:
            _check_workers_and_shards("val", self.num_val_workers, self.num_val_shards)
        if self.test_shards:
            _check_workers_and_shards("test", self.num_val_workers, self.num_test_shards)

    @staticmethod
    def _filter_properties(
        input_dict: Dict[str, Any], prefixes_to_keep: Tuple[str]
    ) -> Dict[str, Any]:
        prefixes_to_keep = ("_",) + prefixes_to_keep  # Keep underscore properties like "__key__"
        out_dict = {}
        for key, value in input_dict.items():
            if any(key.startswith(prefix) for prefix in prefixes_to_keep):
                out_dict[key] = value

        return out_dict

    @staticmethod
    def _remove_extensions(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {name.split(".")[0]: value for name, value in input_dict.items()}

    @staticmethod
    def _pad(dataset: Iterable[Dict[str, Any]], n_samples: int) -> Iterable[Dict[str, Any]]:
        """Iterates dataset, then adds dummy samples until reaching the specified number of samples.

        Dummy samples are constructed by copying structure of the first encountered sample.
        Also adds a special property "batch_padding_mask" to indicate which entries are padding.
        """
        example = None
        count = 0
        for sample in dataset:
            if example is None:
                example = sample
            count += 1
            yield {**sample, WebdatasetDataModule.BATCH_PADDING_MASK_KEY: np.array(False)}

        while count < n_samples:
            if example is None:
                sample = {}  # Dataset was empty, should not really happen
            else:
                sample = {
                    key: WebdatasetDataModule._get_padding(key, value)
                    for key, value in example.items()
                }
            sample[WebdatasetDataModule.BATCH_PADDING_MASK_KEY] = np.array(True)
            count += 1
            yield sample

    @staticmethod
    def _get_padding(key: str, value: Any):
        """Construct padding for property."""
        if isinstance(value, str):
            return "PADDING"
        elif isinstance(value, torch.Tensor):
            return torch.zeros_like(value)
        else:
            return np.zeros_like(value)

    def _get_max_samples_per_worker(
        self, dataset_size: int, num_shards: int, num_workers: int
    ) -> int:
        """Estimate upper bound on the number of samples per data worker.

        It is only approximate because we don't know the exact composition of the shards. If the
        number of samples per shard is very different, this estimate may be too low.
        """
        num_workers = max(num_workers, 1)
        max_samples_per_shard = int(math.ceil(dataset_size / num_shards))
        max_shards_per_node = int(math.ceil(num_shards / self.num_nodes))
        max_shards_per_worker = int(math.ceil(max_shards_per_node / num_workers))
        return max_shards_per_worker * max_samples_per_shard

    @staticmethod
    def _get_webdataset(
        urls, resampled=False, splitter=None, cache_size=-1, cache_dir=None
    ) -> wds.FluidWrapper:
        """Create pipeline object serving same function as wds.WebDataset.

        We do this instead of directly using wds.WebDataset in order to have control over
        the `always` argument for caching.

        This method either creates a shuffling, resampling dataset for `resampled=True`, or a
        deterministic, non-shuffling for `resample=False`. We do not need other modes for now.
        """
        if resampled:
            shardlist = wds.shardlists.ResampledShards(urls, deterministic=True)
        else:
            shardlist = wds.shardlists.SimpleShardList(urls)

        dataset = wds.FluidWrapper(shardlist)

        if not resampled:
            if splitter is None:
                splitter = wds.shardlists.single_node_only
            dataset.append(splitter)
            dataset.append(wds.shardlists.split_by_worker)

        handler = wds.filters.reraise_exception
        if cache_dir is None or cache_size == 0:
            dataset.append(wds.tariterators.tarfile_to_samples(handler=handler))
        else:
            assert cache_size == -1 or cache_size > 0
            dataset.append(
                wds.cache.cached_tarfile_to_samples(
                    handler=handler,
                    verbose=False,
                    cache_size=cache_size,
                    cache_dir=cache_dir,
                    always=True,
                )
            )

        return dataset

    def _get_dataset(
        self,
        shards: Union[str, List[str]],
        shuffle: bool = False,
        pipeline: Optional[pipelines.DataPipeline] = None,
        padded_size_per_worker: Optional[int] = None,
        cache_dir: Optional[str] = None,
        cache_size: int = -1,
    ):
        if shuffle:
            # For shuffling samples, we sample shards with replacement. This means that each node
            # and worker uses all shards from the dataset (instead of splitting shards).
            dataset = self._get_webdataset(
                shards, resampled=True, cache_dir=cache_dir, cache_size=cache_size
            )
        else:
            splitter = (
                wds.shardlists.split_by_node
                if self.num_nodes > 1
                else wds.shardlists.single_node_only
            )
            dataset = self._get_webdataset(
                shards, splitter=splitter, cache_dir=cache_dir, cache_size=cache_size
            )

        # Filter unneeded properties first to avoid decoding them. If pipeline defines no keys,
        # keep everything.
        if pipeline and pipeline.keys is not None:
            dataset = dataset.map(
                partial(WebdatasetDataModule._filter_properties, prefixes_to_keep=pipeline.keys)
            )

        dataset = dataset.decode("rgb").map(WebdatasetDataModule._remove_extensions)

        if padded_size_per_worker is not None:
            # Pad dataset stream to contain a certain number of samples. This is needed to balance
            # data between nodes and workers during validation. Note that `padded_size` here refers
            # to the number of samples PER WORKER, not to the total number of samples in the dataset.
            dataset = dataset.compose(
                partial(WebdatasetDataModule._pad, n_samples=padded_size_per_worker)
            )
            # Only add length if we can be sure about the exact number of samples. If we do not add
            # padding and only know the global dataset size, this is not the case, because samples
            # may be unevenly distributed over nodes and workers.
            dataset = dataset.with_length(padded_size_per_worker)

        if pipeline:
            dataset = pipeline.apply(dataset)

            if padded_size_per_worker and pipeline.get_num_samples(padded_size_per_worker):
                dataset = dataset.with_length(pipeline.get_num_samples(padded_size_per_worker))

        return dataset

    def _get_dataloader(
        self,
        dataset,
        batch_size: int,
        num_workers: int,
        num_samples_per_epoch: Optional[int],
        partial_batches: bool = False,
    ):
        assert num_samples_per_epoch is None or num_samples_per_epoch > 0

        # Do batching within each worker
        dataset_batched = dataset.batched(
            batch_size, partial=partial_batches, collation_fn=torch_collate.default_collate
        )

        dataloader = wds.WebLoader(
            dataset_batched,
            num_workers=num_workers,
            batch_size=None,
            worker_init_fn=worker_init_function,
            persistent_workers=num_workers > 0,
            # Heuristic to check whether GPUs are used. Misses the case where num_gpus = 1
            pin_memory=(self.num_nodes > 1 and torch.cuda.is_available()),
            prefetch_factor=2,
        )

        num_batches_per_epoch = None
        if num_samples_per_epoch is not None:
            if partial_batches:
                num_batches_per_epoch = int(
                    math.ceil(num_samples_per_epoch / (batch_size * self.num_nodes))
                )
                dataloader = dataloader.with_epoch(num_batches_per_epoch)
            else:
                num_batches_per_epoch = num_samples_per_epoch // (batch_size * self.num_nodes)
                # Equalize batches across nodes and workers for DDP. This may lead to dropping some
                # samples and repeating other samples across an epoch in case the shards are
                # unevenly distributed across nodes. This seems to be unavoidable with DDP.
                # See https://github.com/webdataset/webdataset/issues/225#issuecomment-1344642570
                dataloader = dataloader.repeat(2).with_epoch(num_batches_per_epoch)
        else:
            # Set dataset to loop indefinitely
            dataloader = dataloader.repeat()

        if num_batches_per_epoch:
            dataloader = dataloader.with_length(num_batches_per_epoch)

        return dataloader

    def train_dataset(self):
        self._verify_settings_lazy()
        if self.train_shards is None:
            raise ValueError("No training split.")

        return self._get_dataset(
            self.train_shards,
            shuffle=True,
            pipeline=self.train_pipeline,
            cache_dir=self.cache_dir if self.cache_train else None,
        )

    def val_dataset(self):
        self._verify_settings_lazy()
        if self.val_shards is None:
            raise ValueError("No validation split.")

        padded_size = self._get_max_samples_per_worker(
            self.val_size, self.num_val_shards, self.num_val_workers
        )
        return self._get_dataset(
            self.val_shards,
            shuffle=False,
            pipeline=self.val_pipeline,
            padded_size_per_worker=padded_size,
            cache_dir=self.cache_dir if self.cache_val else None,
        )

    def test_dataset(self):
        self._verify_settings_lazy()
        if self.test_shards is None:
            raise ValueError("No test split.")

        padded_size = self._get_max_samples_per_worker(
            self.test_size, self.num_test_shards, self.num_val_workers
        )
        return self._get_dataset(
            self.test_shards,
            shuffle=False,
            pipeline=self.test_pipeline,
            padded_size_per_worker=padded_size,
            cache_dir=self.cache_dir if self.cache_val else None,
        )

    def train_dataloader(self):
        return self._get_dataloader(
            self.train_dataset(),
            batch_size=self.batch_size,
            num_workers=self.num_train_workers,
            num_samples_per_epoch=self.samples_per_epoch,
            partial_batches=False,
        )

    def val_dataloader(self):
        dataset = self.val_dataset()

        try:
            num_samples_per_worker = len(dataset)
            num_samples_per_epoch = (
                num_samples_per_worker * self.num_nodes * max(self.num_val_workers, 1)
            )
        except TypeError:
            num_samples_per_epoch = None

        return self._get_dataloader(
            dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_val_workers,
            num_samples_per_epoch=num_samples_per_epoch,
            partial_batches=True,
        )

    def test_dataloader(self):
        dataset = self.val_dataset()

        try:
            num_samples_per_worker = len(dataset)
            num_samples_per_epoch = (
                num_samples_per_worker * self.num_nodes * max(self.num_val_workers, 1)
            )
        except TypeError:
            num_samples_per_epoch = None

        return self._get_dataloader(
            self.test_dataset(),
            batch_size=self.val_batch_size,
            num_workers=self.num_val_workers,
            num_samples_per_epoch=num_samples_per_epoch,
            partial_batches=True,
        )


def _to_abs_shard_path(
    shards: Union[str, List[str]], data_root_dir: Optional[str]
) -> Union[str, List[str]]:
    """Turn relative shard path to absolute by preprending the path to the data root directory."""
    if isinstance(shards, str):
        if os.path.isabs(shards):
            return shards  # Directly use absolute paths
        elif "://" in shards:
            return shards  # Directly use URI's like `s3://abc/xyz`
        else:
            if data_root_dir is not None:
                return os.path.join(data_root_dir, shards)
            else:
                raise ValueError(
                    f"Passed relative shard path {shards}, but data root path is missing."
                )
    else:
        assert isinstance(shards, Iterable), f"Expected iterable but found {type(shards)}"
        return [_to_abs_shard_path(shard, data_root_dir) for shard in shards]


class DummyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_size: int,
        val_size: int,
        batch_size: int,
        shapes: Dict[str, Tuple[int]],
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
    ):
        super().__init__()
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.shapes = shapes
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def __str__(self) -> str:
        res = ["DummyDataModule"]
        res.append(f"  - Number of train samples: {self.train_size}")
        res.append(f"  - Number of val samples: {self.val_size}")
        res.append(f"  - Number of test samples: {self.test_size}")
        res.append(f"  - Batch size: {self.batch_size}")
        return "\n".join(res)

    @staticmethod
    def _make_random_dataset(shapes, size: int, seed: int):
        rng = np.random.RandomState(seed)
        dataset = []
        for idx in range(size):
            data = {"__key__": str(idx)}
            for name, shape in shapes.items():
                data[name] = rng.randint(0, 255, size=shape, dtype=np.uint8)
            dataset.append(data)

        return dataset

    @staticmethod
    def _make_squares_dataset(shapes, size: int, seed: int, n_objects: int = 3):
        rng = np.random.RandomState(seed)
        dataset = []
        for idx in range(size):
            data = {"__key__": str(idx)}
            for name, shape in shapes.items():
                height, width = shape[-3:-1]
                # Place random black squares on white background
                if "mask" in name:
                    array = np.zeros(shape, dtype=np.uint8)
                else:
                    array = np.ones(shape, dtype=np.uint8) * 255
                for idx in range(n_objects):
                    x = rng.randint(0, width)
                    y = rng.randint(0, height)
                    size = rng.randint(0, int(0.3 * (height + width) / 2))
                    if "mask" in name:
                        array[..., y : y + size, x : x + size, :] = idx + 1
                    else:
                        array[..., y : y + size, x : x + size, :] = 0

                data[name] = array
            dataset.append(data)

        return dataset

    def setup(self, stage):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, data, transforms):
                super().__init__()
                self.data = data
                self.transforms = transforms

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                data = {**self.data[idx]}  # Copy dict
                if self.transforms:
                    for name, transform in self.transforms.items():
                        data[name] = transform(data[name])

                return data

        train_data = self._make_squares_dataset(self.shapes, self.train_size, 42)
        self.train_set = Dataset(train_data, self.train_transforms)
        val_data = self._make_squares_dataset(self.shapes, self.val_size, 42)
        self.val_set = Dataset(val_data, self.val_transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
