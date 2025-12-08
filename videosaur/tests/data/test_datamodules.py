import itertools

import pytest
import torch

from videosaur.data import datamodules


@pytest.fixture(scope="session")
def dummy_shards(tmp_path_factory):
    import webdataset as wds

    data_dir = tmp_path_factory.mktemp("data")

    def write_split(split, n_samples, n_samples_per_shard):
        pattern = str(data_dir / f"{split}-%06d.tar")
        with wds.ShardWriter(pattern, maxcount=n_samples_per_shard) as sink:
            sink.verbose = 0
            for idx in range(n_samples):
                sink.write({"__key__": str(idx), "tensor.pth": torch.randn(2, 2)})

        return sorted([str(p) for p in data_dir.glob(f"{split}-*.tar")])

    train_size, val_size = 10, 7
    train_shards = write_split("train", train_size, 4)
    val_shards = write_split("validation", val_size, 4)
    return {
        "train_shards": train_shards,
        "val_shards": val_shards,
        "train_size": train_size,
        "val_size": val_size,
        "tensor_key": "tensor",
    }


@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_webdataset_datamodule(dummy_shards, num_workers):
    batch_size, val_batch_size = 3, 2
    datamodule = datamodules.WebdatasetDataModule(
        train_shards=dummy_shards["train_shards"],
        val_shards=dummy_shards["val_shards"],
        val_size=dummy_shards["val_size"],
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        num_val_workers=num_workers,
    )

    # Check that training iterator contains an infinite stream of data, i.e. that dataloader does
    # not stop prematurely
    expected_n_batches = (dummy_shards["train_size"] // batch_size) + 3
    n_batches = 0
    for batch in itertools.islice(datamodule.train_dataloader(), expected_n_batches):
        assert batch[dummy_shards["tensor_key"]].shape[0] == batch_size
        n_batches += 1

    assert n_batches == expected_n_batches

    # Check that validation iterator contains exactly the validation data, modulo padding
    max_batches = dummy_shards["val_size"]  # Set upper bound to avoid infinite iteration
    keys = []
    for batch in itertools.islice(datamodule.val_dataloader(), max_batches):
        assert batch[dummy_shards["tensor_key"]].shape[0] == val_batch_size
        keys.extend([key for key in batch["__key__"] if key != "PADDING"])

    assert len(keys) == dummy_shards["val_size"]
    if num_workers <= 1:
        # For one worker, we expect the samples to be iterated in order
        assert sorted(list(set(keys))) == keys
    else:
        # For more than one worker, the samples are interleaved, so we need to sort to compare
        assert sorted(list(set(keys))) == sorted(keys)


@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_webdataset_datamodule_fixed_epoch_len(dummy_shards, num_workers):
    batch_size = 3
    samples_per_epoch = dummy_shards["train_size"]
    datamodule = datamodules.WebdatasetDataModule(
        train_shards=dummy_shards["train_shards"],
        samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Check that training iterator stops after having sampled roughly `samples_per_epoch` samples
    max_batches = dummy_shards["train_size"]  # Set upper bound to avoid infinite iteration
    n_batches = 0
    for batch in itertools.islice(datamodule.train_dataloader(), max_batches):
        assert batch[dummy_shards["tensor_key"]].shape[0] == batch_size
        n_batches += 1

    expected_n_batches = samples_per_epoch // batch_size
    assert n_batches == expected_n_batches
