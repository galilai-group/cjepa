import pathlib

import pytest

from videosaur.videosaur import configuration, models, train


def _should_filter_config(config: pathlib.Path) -> bool:
    if config.name.startswith("_"):
        return True

    if "configs/eval/" in str(config):
        return True

    return False


TEST_ROOT = pathlib.Path(__file__).parent
PROJ_ROOT = TEST_ROOT.parent
CONFIGS = sorted(p for p in PROJ_ROOT.glob("configs/**/*.yml") if not _should_filter_config(p))
TEST_CONFIGS = sorted(TEST_ROOT.glob("configs/test_*.yml"))

DATASETS_WITHOUT_BACKGROUND_MASK = ("coco",)
DATASETS_WITH_OVERLAPPING_MASKS = ("coco",)
DATASETS_WITHOUT_OVERLAPPING_MASKS = ("movi_c", "movi_e", "davis")


@pytest.mark.parametrize("config", CONFIGS)
def test_configs_loadable(config):
    """Test checking that configs load and dataset and model can be constructed from them."""
    args = train.parser.parse_args(
        [
            "--dry",
            "--quiet",
            str(config),
            "trainer.max_steps=0",
            "trainer.num_sanity_val_steps=0",
            "trainer.accelerator=cpu",
            # Do not load pre-trained weights because we can not access them in the CI
            "model.load_weights=null",
        ]
    )
    train.main(args)


@pytest.mark.dataset
@pytest.mark.parametrize("config", TEST_CONFIGS)
def test_test_configs(config, tmp_path):
    """Integration test checking that training and validation works for test configs."""
    args = train.parser.parse_args(
        [
            "--quiet",
            f"--log-dir={tmp_path}",
            str(config),
            "trainer.max_steps=2",
            "trainer.val_check_interval=2",
            "trainer.num_sanity_val_steps=0",
            "trainer.limit_train_batches=2",
            "trainer.limit_val_batches=2",
            "trainer.limit_test_batches=2",
            "trainer.accelerator=cpu",
        ]
    )
    train.main(args)

    checkpoints = list(tmp_path.glob(f"**/{train.CHECKPOINT_SUBDIR}/*.ckpt"))
    assert len(checkpoints) > 0, "Expected a final checkpoint to be saved"


@pytest.mark.slow
@pytest.mark.dataset
@pytest.mark.parametrize("config", CONFIGS)
def test_configs(config, tmp_path):
    """Integration test checking that training and validation works for configs."""
    args = train.parser.parse_args(
        [
            "--quiet",
            f"--log-dir={tmp_path}",
            str(config),
            "trainer.max_steps=2",
            "trainer.val_check_interval=2",
            "trainer.num_sanity_val_steps=0",
            "trainer.limit_train_batches=2",
            "trainer.limit_val_batches=2",
            "trainer.limit_test_batches=2",
            "trainer.accelerator=cpu",
        ]
    )
    train.main(args)

    checkpoints = list(tmp_path.glob(f"**/{train.CHECKPOINT_SUBDIR}/*.ckpt"))
    assert len(checkpoints) > 0, "Expected a final checkpoint to be saved"


def test_load_image_model_checkpoint(tmp_path):
    # First generate an image-model checkpoint
    args = train.parser.parse_args(
        [
            "--quiet",
            f"--log-dir={tmp_path}",
            str(TEST_ROOT / "configs/test_dummy_image.yml"),
            "trainer.max_steps=1",
            "trainer.val_check_interval=0",
            "trainer.num_sanity_val_steps=0",
            "trainer.limit_train_batches=1",
            "trainer.limit_val_batches=0",
            "trainer.limit_test_batches=0",
            "trainer.accelerator=cpu",
        ]
    )
    train.main(args)

    checkpoints = list(tmp_path.glob(f"**/{train.CHECKPOINT_SUBDIR}/*.ckpt"))
    assert len(checkpoints) > 0, "Expected a final checkpoint to be saved"

    # Then construct a video model and try to load the checkpoint
    config = configuration.load_config(TEST_ROOT / "configs/test_dummy_video.yml")

    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(
        checkpoints[0],
        {
            "initializer": "initializer",
            "processor.module.corrector": "processor.corrector",
            "decoder.module": "decoder",
        },
    )


@pytest.mark.parametrize("config_path", CONFIGS + TEST_CONFIGS)
def test_config_metrics_settings(config_path):
    config = configuration.load_config(config_path)

    def test_properties(split, dataset_name, must_exist=False, **kwargs):
        if split == "train":
            metric_configs = config.train_metrics if config.train_metrics else {}
        else:
            metric_configs = config.val_metrics if config.val_metrics else {}

        if f"{split}_shards" in config.dataset and dataset_name in config.dataset[f"{split}_shards"]:
            for metric_name, metric_config in metric_configs.items():
                for name, value in kwargs.items():
                    if must_exist:
                        msg = (
                            f"Dataset {dataset_name} must have property {name} for "
                            f"{split} metric {metric_name}"
                        )
                        assert name in metric_config, msg
                    if name in metric_config:
                        msg = (
                            f"Dataset {dataset_name} must have {name}={value} for "
                            f"{split} metric {metric_name}"
                        )
                        assert metric_config[name] == value, msg

    # On datasets which do not contain a background mask, we should not let the metric ignore the
    # background mask (because there is none to remove).
    for dataset in DATASETS_WITHOUT_BACKGROUND_MASK:
        test_properties("train", dataset, ignore_background=False)
        test_properties("val", dataset, ignore_background=False)

    # On datasets which have mask overlaps, the metric should ignore those overlaps
    for dataset in DATASETS_WITH_OVERLAPPING_MASKS:
        test_properties("train", dataset, must_exist=True, ignore_overlaps=True)
        test_properties("val", dataset, must_exist=True, ignore_overlaps=True)

    # On datasets which do not have mask overlaps, we do not need to set ignore_overlap to true
    for dataset in DATASETS_WITHOUT_OVERLAPPING_MASKS:
        test_properties("train", dataset, ignore_overlaps=False)
        test_properties("val", dataset, ignore_overlaps=False)
