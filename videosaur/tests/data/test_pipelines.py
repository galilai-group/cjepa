import os

import omegaconf
import pytest
import torch

from videosaur.videosaur.data import datamodules, pipelines


@pytest.mark.dataset
def test_video_transforms_movi_e(data_root_dir):
    val_shards = os.path.join(data_root_dir, "movi_e/movi_e-validation-{000000..000000}.tar")
    batch_size = 2
    episode_length = 6
    input_size = 224
    num_classes = 24

    pipeline_config = omegaconf.DictConfig(
        {
            "name": "video",
            "keys": ("video", "segmentations"),
            "chunk_size": episode_length,
            "transforms": {
                "name": "movi_val",
                "type": "video",
                "input_size": input_size,
                "num_classes": num_classes,
            },
        }
    )
    val_pipeline = pipelines.build(pipeline_config)

    data_module = datamodules.WebdatasetDataModule(
        val_shards=val_shards,
        val_size=250,
        batch_size=batch_size,
        num_workers=0,
        val_pipeline=val_pipeline,
    )
    dataloader = data_module.val_dataloader()

    for batch in dataloader:
        assert batch["video"].shape == (batch_size, episode_length, 3, input_size, input_size)
        assert batch["segmentations"].shape == (
            batch_size,
            episode_length,
            num_classes,
            input_size,
            input_size,
        )
        assert batch["video"].dtype == torch.float32
        assert batch["segmentations"].dtype == torch.bool
        break


@pytest.mark.dataset
@pytest.mark.parametrize(
    "train_crop_type", ["central", "random", "short_side_resize_random", "short_side_resize_central"]
)
@pytest.mark.parametrize("val_crop_type", ["central", "short_side_resize_central"])
@pytest.mark.parametrize("train_h_flip_prob", [0.5, None])
@pytest.mark.parametrize("year", [2019, 2021])
def test_video_transforms_ytvis(
    data_root_dir, year, train_crop_type, val_crop_type, train_h_flip_prob
):
    val_shards = os.path.join(
        data_root_dir, f"ytvis{year}_resized/" + "ytvis-validation-{000000..000000}.tar"
    )
    train_shards = os.path.join(
        data_root_dir, f"ytvis{year}_resized/" + "ytvis-train-{000000..000000}.tar"
    )
    batch_size = 2
    episode_length = 6
    input_size = 224
    num_classes = 10

    pipeline_config = omegaconf.DictConfig(
        {
            "name": "video",
            "keys": ("video", "segmentations"),
            "chunk_size": episode_length,
            "transforms": {
                "name": "ytvis_val",
                "type": "video",
                "crop_type": val_crop_type,
                "input_size": input_size,
                "num_classes": num_classes,
            },
        }
    )
    val_pipeline = pipelines.build(pipeline_config)

    data_module = datamodules.WebdatasetDataModule(
        val_shards=val_shards,
        val_size=30,
        batch_size=batch_size,
        num_workers=0,
        val_pipeline=val_pipeline,
    )
    dataloader = data_module.val_dataloader()

    for batch in dataloader:
        assert batch["video"].shape == (batch_size, episode_length, 3, input_size, input_size)
        assert batch["segmentations"].shape == (
            batch_size,
            episode_length,
            num_classes,
            input_size,
            input_size,
        )
        # check that values are binary (0.0 or 1.0)
        assert torch.equal(
            sum(batch["segmentations"] == i for i in [0.0, 1.0]).bool(),
            torch.ones_like(batch["segmentations"]).bool(),
        )
        assert batch["video"].dtype == torch.float32
        assert batch["segmentations"].dtype == torch.bool
        break

    pipeline_config = omegaconf.DictConfig(
        {
            "name": "video",
            "keys": ("video",),
            "chunk_size": episode_length,
            "transforms": {
                "name": "ytvis_train",
                "type": "video",
                "crop_type": train_crop_type,
                "h_flip_prob": train_h_flip_prob,
                "input_size": input_size,
                "num_classes": num_classes,
            },
        }
    )
    train_pipeline = pipelines.build(pipeline_config)

    data_module = datamodules.WebdatasetDataModule(
        train_shards=train_shards,
        batch_size=batch_size,
        num_workers=0,
        train_pipeline=train_pipeline,
    )
    dataloader = data_module.train_dataloader()

    for batch in dataloader:
        assert batch["video"].shape == (batch_size, episode_length, 3, input_size, input_size)
        assert batch["video"].dtype == torch.float32
        break


@pytest.mark.dataset
@pytest.mark.parametrize(
    "train_crop_type", ["central", "random", "short_side_resize_random", "short_side_resize_central"]
)
@pytest.mark.parametrize("val_crop_type", ["central", "short_side_resize_central"])
@pytest.mark.parametrize("train_h_flip_prob", [0.5, None])
def test_video_transforms_davis(data_root_dir, train_crop_type, val_crop_type, train_h_flip_prob):
    val_shards = os.path.join(data_root_dir, "davis/davis-validation-{000000..000000}.tar")
    batch_size = 2
    episode_length = 6
    input_size = 224
    num_classes = 10

    pipeline_config = omegaconf.DictConfig(
        {
            "name": "video",
            "keys": ("video", "segmentations"),
            "chunk_size": episode_length,
            "transforms": {
                "name": "davis_val",
                "type": "video",
                "crop_type": val_crop_type,
                "input_size": input_size,
                "num_classes": num_classes,
            },
        }
    )
    val_pipeline = pipelines.build(pipeline_config)

    data_module = datamodules.WebdatasetDataModule(
        val_shards=val_shards,
        val_size=30,
        batch_size=batch_size,
        num_workers=0,
        val_pipeline=val_pipeline,
    )
    dataloader = data_module.val_dataloader()

    for batch in dataloader:
        assert batch["video"].shape == (batch_size, episode_length, 3, input_size, input_size)
        assert batch["segmentations"].shape == (
            batch_size,
            episode_length,
            num_classes,
            input_size,
            input_size,
        )
        # check that values are binary (0.0 or 1.0)
        assert torch.equal(
            sum(batch["segmentations"] == i for i in [0.0, 1.0]).bool(),
            torch.ones_like(batch["segmentations"]).bool(),
        )
        assert batch["video"].dtype == torch.float32
        assert batch["segmentations"].dtype == torch.bool
        break

    train_shards = os.path.join(data_root_dir, "davis/davis-train-{000000..000000}.tar")
    pipeline_config = omegaconf.DictConfig(
        {
            "name": "video",
            "keys": ("video", "segmentations"),
            "chunk_size": episode_length,
            "transforms": {
                "name": "davis_train",
                "type": "video",
                "crop_type": train_crop_type,
                "h_flip_prob": train_h_flip_prob,
                "input_size": input_size,
                "num_classes": num_classes,
            },
        }
    )
    train_pipeline = pipelines.build(pipeline_config)

    data_module = datamodules.WebdatasetDataModule(
        train_shards=train_shards,
        batch_size=batch_size,
        num_workers=0,
        train_pipeline=train_pipeline,
    )
    dataloader = data_module.train_dataloader()

    for batch in dataloader:
        assert batch["video"].shape == (batch_size, episode_length, 3, input_size, input_size)
        assert batch["video"].dtype == torch.float32
        break


@pytest.mark.dataset
def test_image_transforms_movi_e(data_root_dir):
    val_shards = os.path.join(data_root_dir, "movi_e/movi_e-validation-{000000..000000}.tar")
    batch_size = 2
    input_size = 224
    num_classes = 24

    pipeline_config = omegaconf.DictConfig(
        {
            "name": "image",
            "keys": ("video", "segmentations"),
            "transforms": {
                "name": "movi_val",
                "type": "image",
                "input_size": input_size,
                "num_classes": num_classes,
            },
        }
    )
    val_pipeline = pipelines.build(pipeline_config)

    data_module = datamodules.WebdatasetDataModule(
        val_shards=val_shards,
        val_size=250,
        batch_size=batch_size,
        num_workers=0,
        val_pipeline=val_pipeline,
    )
    dataloader = data_module.val_dataloader()

    for batch in dataloader:
        assert batch["image"].shape == (batch_size, 3, input_size, input_size)
        assert batch["segmentations"].shape == (
            batch_size,
            num_classes,
            input_size,
            input_size,
        )
        assert batch["image"].dtype == torch.float32
        assert batch["segmentations"].dtype == torch.bool
        break


@pytest.mark.dataset
@pytest.mark.parametrize(
    "train_crop_type", ["central", "random", "short_side_resize_random", "short_side_resize_central"]
)
@pytest.mark.parametrize("val_crop_type", ["central", "short_side_resize_central"])
@pytest.mark.parametrize("train_h_flip_prob", [0.5, None])
def test_image_transforms_davis(data_root_dir, val_crop_type, train_crop_type, train_h_flip_prob):
    val_shards = os.path.join(data_root_dir, "davis/davis-validation-{000000..000000}.tar")
    batch_size = 2
    input_size = 224
    num_classes = 10

    pipeline_config = omegaconf.DictConfig(
        {
            "name": "image",
            "keys": ("video", "segmentations"),
            "transforms": {
                "name": "davis_val",
                "type": "image",
                "crop_type": val_crop_type,
                "input_size": input_size,
                "num_classes": num_classes,
            },
        }
    )
    val_pipeline = pipelines.build(pipeline_config)

    data_module = datamodules.WebdatasetDataModule(
        val_shards=val_shards,
        val_size=30,
        batch_size=batch_size,
        num_workers=0,
        val_pipeline=val_pipeline,
    )
    dataloader = data_module.val_dataloader()

    for batch in dataloader:
        assert batch["image"].shape == (batch_size, 3, input_size, input_size)
        assert batch["segmentations"].shape == (
            batch_size,
            num_classes,
            input_size,
            input_size,
        )
        # check that values are binary (0.0 or 1.0)
        assert torch.equal(
            sum(batch["segmentations"] == i for i in [0.0, 1.0]).bool(),
            torch.ones_like(batch["segmentations"]).bool(),
        )
        assert batch["image"].dtype == torch.float32
        assert batch["segmentations"].dtype == torch.bool
        break

    train_shards = os.path.join(data_root_dir, "davis/davis-train-{000000..000000}.tar")
    pipeline_config = omegaconf.DictConfig(
        {
            "name": "image",
            "keys": ("video", "segmentations"),
            "transforms": {
                "name": "davis_train",
                "type": "image",
                "crop_type": train_crop_type,
                "input_size": input_size,
                "num_classes": num_classes,
                "h_flip_prob": train_h_flip_prob,
            },
        }
    )
    train_pipeline = pipelines.build(pipeline_config)

    data_module = datamodules.WebdatasetDataModule(
        train_shards=train_shards,
        batch_size=batch_size,
        num_workers=0,
        train_pipeline=train_pipeline,
    )
    dataloader = data_module.train_dataloader()

    for batch in dataloader:
        assert batch["image"].shape == (batch_size, 3, input_size, input_size)
        assert batch["image"].dtype == torch.float32
        break


@pytest.mark.dataset
@pytest.mark.parametrize("mask_size", [384, None])
def test_image_transforms_coco(data_root_dir, mask_size):
    batch_size = 2
    input_size = 224
    num_classes = 30
    val_shards = os.path.join(data_root_dir, "coco/coco-validation-{000000..000000}.tar")
    train_shards = os.path.join(data_root_dir, "coco/coco-train-{000000..000000}.tar")
    mask_size = mask_size if mask_size else input_size

    pipeline_config = omegaconf.DictConfig(
        {
            "name": "image",
            "keys": ("image", "segmentations"),
            "is_video_dataset": False,
            "transforms": {
                "name": "coco_val",
                "type": "image",
                "crop_type": "central",
                "input_size": input_size,
                "num_classes": num_classes,
                "mask_size": mask_size,
            },
        }
    )
    val_pipeline = pipelines.build(pipeline_config)

    data_module = datamodules.WebdatasetDataModule(
        val_shards=val_shards,
        val_size=30,
        batch_size=batch_size,
        num_workers=0,
        val_pipeline=val_pipeline,
    )
    dataloader = data_module.val_dataloader()

    for batch in dataloader:
        assert batch["image"].shape == (batch_size, 3, input_size, input_size)
        assert batch["segmentations"].shape == (
            batch_size,
            num_classes,
            mask_size,
            mask_size,
        )
        # check that values are binary (0.0 or 1.0)
        assert torch.equal(
            sum(batch["segmentations"] == i for i in [0.0, 1.0]).bool(),
            torch.ones_like(batch["segmentations"]).bool(),
        )
        assert batch["image"].dtype == torch.float32
        assert batch["segmentations"].dtype == torch.bool
        break

    pipeline_config = omegaconf.DictConfig(
        {
            "name": "image",
            "keys": ("image",),
            "is_video_dataset": False,
            "transforms": {
                "name": "coco_train",
                "type": "image",
                "crop_type": "random",
                "input_size": input_size,
                "num_classes": num_classes,
            },
        }
    )
    train_pipeline = pipelines.build(pipeline_config)

    data_module = datamodules.WebdatasetDataModule(
        train_shards=train_shards,
        batch_size=batch_size,
        num_workers=0,
        train_pipeline=train_pipeline,
    )
    dataloader = data_module.train_dataloader()

    for batch in dataloader:
        assert batch["image"].shape == (batch_size, 3, input_size, input_size)
        assert batch["image"].dtype == torch.float32
        break


@pytest.mark.dataset
def test_video_transforms_movi_c(data_root_dir):
    val_shards = os.path.join(data_root_dir, "movi_c/movi_c-validation-{000000..000000}.tar")
    batch_size = 2
    episode_length = 6
    input_size = 128
    target_size = 224
    num_classes = 24

    pipeline_config = omegaconf.DictConfig(
        {
            "name": "video",
            "keys": ("video", "segmentations"),
            "duplicate": {"video": "target_video"},
            "chunk_size": episode_length,
            "transforms": {
                "name": "movi_val",
                "type": "video",
                "target_size": target_size,
                "input_size": input_size,
                "num_classes": num_classes,
            },
        }
    )
    val_pipeline = pipelines.build(pipeline_config)

    data_module = datamodules.WebdatasetDataModule(
        val_shards=val_shards,
        val_size=250,
        batch_size=batch_size,
        num_workers=0,
        val_pipeline=val_pipeline,
    )
    dataloader = data_module.val_dataloader()

    for batch in dataloader:
        assert batch["target_video"].shape == (
            batch_size,
            episode_length,
            3,
            target_size,
            target_size,
        )
        assert batch["video"].shape == (batch_size, episode_length, 3, input_size, input_size)
        assert batch["segmentations"].shape == (
            batch_size,
            episode_length,
            num_classes,
            input_size,
            input_size,
        )
        assert batch["video"].dtype == torch.float32
        assert batch["segmentations"].dtype == torch.bool
        break
