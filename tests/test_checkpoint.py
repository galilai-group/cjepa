from pathlib import Path

import pytest

import checkpoint


def test_policy_name_records_mask_count():
    assert checkpoint.policy_spec("cjepa/pusht_m2") == ("pusht", 2)
    assert checkpoint.released_checkpoint_name("pusht", 2) == (
        "cjepa-ckpts/pusht_videosaur_2_epoch_30_object.ckpt"
    )


def test_invalid_download_policy_has_actionable_error():
    with pytest.raises(ValueError, match="pusht_m2"):
        checkpoint.policy_spec("cjepa/pusht")


def test_policy_checkpoint_path_matches_auto_cost_model(tmp_path):
    assert checkpoint.policy_checkpoint_path("cjepa/pusht_m1", tmp_path) == (
        tmp_path / "cjepa" / "pusht_m1_object.ckpt"
    )


def test_download_is_renamed_before_conversion(tmp_path, monkeypatch):
    downloaded = tmp_path / "download" / "legacy.ckpt"
    downloaded.parent.mkdir()
    downloaded.write_bytes(b"legacy")
    converted = []

    monkeypatch.setattr(
        checkpoint,
        "hf_hub_download",
        lambda **kwargs: str(downloaded),
    )
    monkeypatch.setattr(checkpoint, "_load_if_current", lambda path: None)
    monkeypatch.setattr(
        checkpoint,
        "convert_legacy_checkpoint",
        lambda path, **kwargs: converted.append((Path(path), kwargs)),
    )

    target = checkpoint.ensure_planning_checkpoint(
        "cjepa/pusht_m2", tmp_path / "checkpoints"
    )

    assert target == tmp_path / "checkpoints" / "cjepa" / "pusht_m2_object.ckpt"
    assert target.read_bytes() == b"legacy"
    assert not downloaded.exists()
    assert converted == [
        (
            target,
            {
                "dataset": "pusht",
                "num_masked_slots": 2,
                "repo_id": "HazelNam/CJEPA",
            },
        )
    ]
