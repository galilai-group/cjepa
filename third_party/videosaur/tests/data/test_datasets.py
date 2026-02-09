import os

import pytest
import torch

from third_party.videosaur.videosaur.data import datamodules


@pytest.mark.dataset
def test_movi_c(data_root_dir):
    val_shards = os.path.join(data_root_dir, "movi_c/movi_c-validation-{000000..000000}.tar")
    batch_size = 2
    data_module = datamodules.WebdatasetDataModule(
        val_shards=val_shards,
        val_size=250,
        batch_size=batch_size,
        num_workers=0,
    )
    for batch in data_module.val_dataloader():
        assert batch["video"].shape == (batch_size, 24, 128, 128, 3)
        assert batch["video"].dtype == torch.uint8
        assert batch["segmentations"].shape == (batch_size, 24, 128, 128, 1)
        assert batch["segmentations"].dtype == torch.uint8
        assert batch["segmentations"].max() <= 10
        break


@pytest.mark.dataset
def test_movi_e(data_root_dir):
    val_shards = os.path.join(data_root_dir, "movi_e/movi_e-validation-{000000..000000}.tar")
    batch_size = 2
    data_module = datamodules.WebdatasetDataModule(
        val_shards=val_shards,
        val_size=250,
        batch_size=batch_size,
        num_workers=0,
    )
    for batch in data_module.val_dataloader():
        assert batch["video"].shape == (batch_size, 24, 128, 128, 3)
        assert batch["video"].dtype == torch.uint8
        assert batch["segmentations"].shape == (batch_size, 24, 128, 128, 1)
        assert batch["segmentations"].dtype == torch.uint8
        assert batch["segmentations"].max() <= 23
        break


@pytest.mark.dataset
@pytest.mark.parametrize("year", [2019, 2021])
def test_ytvis(data_root_dir, year):
    val_shards = os.path.join(
        data_root_dir, f"ytvis{year}_resized/" + "ytvis-validation-{000000..000000}.tar"
    )
    batch_size = 1
    data_module = datamodules.WebdatasetDataModule(
        val_shards=val_shards,
        val_size=30,
        batch_size=batch_size,
        num_workers=0,
    )
    for batch in data_module.val_dataloader():
        assert batch["video"].shape[0] == batch_size
        assert batch["video"].shape[-1] == 3
        assert batch["video"].dtype == torch.uint8
        assert batch["segmentations"].shape[0] == batch_size
        assert batch["segmentations"].dtype == torch.uint8
        break


@pytest.mark.dataset
def test_davis(data_root_dir):
    val_shards = os.path.join(data_root_dir, "davis/davis-validation-{000000..000000}.tar")
    batch_size = 1
    data_module = datamodules.WebdatasetDataModule(
        val_shards=val_shards,
        val_size=30,
        batch_size=batch_size,
        num_workers=0,
    )
    for batch in data_module.val_dataloader():
        assert batch["video"].shape[0] == batch_size
        assert batch["video"].shape[-1] == 3
        assert batch["video"].dtype == torch.uint8
        assert batch["segmentations"].shape[0] == batch_size
        assert batch["segmentations"].shape[-1] == 1
        assert batch["segmentations"].dtype == torch.uint8
        assert batch["segmentations"].max() <= 10
        break


@pytest.mark.dataset
@pytest.mark.parametrize("split", ["train", "validation"])
def test_coco(data_root_dir, split):
    shards = os.path.join(data_root_dir, f"coco/coco-{split}" + "-{000000..000000}.tar")

    batch_size = 1
    data_module = datamodules.WebdatasetDataModule(
        val_shards=shards,
        val_size=30,
        batch_size=batch_size,
        num_workers=0,
    )
    for batch in data_module.val_dataloader():
        assert batch["image"].shape[0] == batch_size
        assert batch["image"].shape[-1] == 3
        assert batch["image"].dtype == torch.uint8
        if split == "validation":
            assert len(batch["segmentations"].shape) == 4
            assert batch["segmentations"].shape[0] == batch_size
            assert batch["segmentations"].dtype == torch.uint8
            assert batch["segmentations"].max() <= 91
