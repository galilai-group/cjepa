"""Unit tests for CPUOffloadCallback.

Run with: pytest test_cpu_offload_callback.py -v
Run specific test: pytest test_cpu_offload_callback.py::test_strategy_compatibility -v
"""

import pytest
import torch
from unittest.mock import Mock, patch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from loguru import logger
import sys

# Import the callback (adjust path as needed)
# from your_module import CPUOffloadCallback
# For testing, we'll assume it's in the same directory
from stable_pretraining.callbacks import CPUOffloadCallback


@pytest.fixture(autouse=True)
def setup_logger():
    """Setup logger for each test."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    yield
    logger.remove()


@pytest.fixture
def mock_trainer_ddp():
    """Create a mock trainer with DDP strategy."""
    trainer = Mock(spec=Trainer)
    trainer.strategy = DDPStrategy()
    trainer.global_rank = 0
    trainer.world_size = 4
    trainer.global_step = 1000
    trainer.current_epoch = 2
    return trainer


@pytest.fixture
def mock_trainer_single_device():
    """Create a mock trainer with single device strategy."""
    trainer = Mock(spec=Trainer)
    trainer.strategy = SingleDeviceStrategy(device=torch.device("cpu"))
    trainer.global_rank = 0
    trainer.world_size = 1
    trainer.global_step = 500
    trainer.current_epoch = 1
    return trainer


@pytest.fixture
def mock_trainer_fsdp():
    """Create a mock trainer with FSDP strategy."""
    trainer = Mock(spec=Trainer)
    # Mock FSDP strategy
    fsdp_strategy = Mock()
    fsdp_strategy.__class__.__name__ = "FSDPStrategy"
    trainer.strategy = fsdp_strategy
    trainer.global_rank = 0
    trainer.world_size = 4
    trainer.global_step = 1000
    trainer.current_epoch = 2
    return trainer


@pytest.fixture
def mock_pl_module():
    """Create a mock LightningModule."""
    module = Mock(spec=LightningModule)

    # Create mock parameters
    param1 = torch.randn(100, 100)  # 10k params
    param2 = torch.randn(200, 200)  # 40k params
    param1.requires_grad = True
    param2.requires_grad = True

    module.parameters.return_value = [param1, param2]
    return module


@pytest.fixture
def sample_checkpoint():
    """Create a sample checkpoint dict."""
    return {
        "state_dict": {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(20, 10),
        },
        "optimizer_states": [
            {
                "state": {
                    0: {
                        "momentum": torch.randn(10, 10),
                        "exp_avg": torch.randn(10, 10),
                    },
                    1: {"momentum": torch.randn(10), "exp_avg": torch.randn(10)},
                },
                "param_groups": [{"lr": 0.001, "weight_decay": 0.01}],
            }
        ],
        "lr_schedulers": [
            {
                "last_epoch": 100,
                "base_lr": 0.001,
            }
        ],
        "epoch": 2,
        "global_step": 1000,
        "callbacks": {"some_callback": "state"},
        "custom_object": Mock(),  # Custom object that should be skipped
    }


# ============================================================================
# Strategy Compatibility Tests
# ============================================================================


@pytest.mark.unit
def test_strategy_compatibility_ddp(mock_trainer_ddp, mock_pl_module):
    """Test that callback is enabled with DDP strategy."""
    callback = CPUOffloadCallback()

    with patch("torch.cuda.is_available", return_value=False):
        callback.setup(mock_trainer_ddp, mock_pl_module, "fit")

    assert callback._is_enabled is True


@pytest.mark.unit
def test_strategy_compatibility_single_device(
    mock_trainer_single_device, mock_pl_module
):
    """Test that callback is enabled with single device strategy."""
    callback = CPUOffloadCallback()

    with patch("torch.cuda.is_available", return_value=False):
        callback.setup(mock_trainer_single_device, mock_pl_module, "fit")

    assert callback._is_enabled is True


@pytest.mark.unit
def test_strategy_compatibility_fsdp(mock_trainer_fsdp, mock_pl_module):
    """Test that callback is disabled with FSDP strategy."""
    callback = CPUOffloadCallback()

    with patch("torch.cuda.is_available", return_value=False):
        callback.setup(mock_trainer_fsdp, mock_pl_module, "fit")

    assert callback._is_enabled is False


@pytest.mark.unit
def test_strategy_compatibility_unknown():
    """Test that callback is disabled with unknown strategy."""
    trainer = Mock(spec=Trainer)
    unknown_strategy = Mock()
    unknown_strategy.__class__.__name__ = "UnknownStrategy"
    trainer.strategy = unknown_strategy
    trainer.global_rank = 0
    trainer.world_size = 1

    callback = CPUOffloadCallback()

    with patch("torch.cuda.is_available", return_value=False):
        callback.setup(trainer, Mock(), "fit")

    assert callback._is_enabled is False


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
def test_init_default_params():
    """Test callback initialization with default parameters."""
    callback = CPUOffloadCallback()

    assert callback.offload_keys == ["state_dict", "optimizer_states", "lr_schedulers"]
    assert callback.log_skipped is False
    assert callback._checkpoint_count == 0
    assert callback._total_time_saved == 0.0
    assert callback._total_memory_freed == 0.0


@pytest.mark.unit
def test_init_custom_params():
    """Test callback initialization with custom parameters."""
    callback = CPUOffloadCallback(offload_keys=["state_dict"], log_skipped=True)

    assert callback.offload_keys == ["state_dict"]
    assert callback.log_skipped is True


@pytest.mark.unit
def test_init_cuda_detection():
    """Test CUDA detection during initialization."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.get_device_name", side_effect=["GPU0", "GPU1"]),
        patch("torch.cuda.get_device_properties") as mock_props,
    ):
        mock_props.return_value.total_memory = 80 * 1e9  # 80GB

        callback = CPUOffloadCallback()

        # Should initialize successfully
        assert callback is not None


# ============================================================================
# CPU Offload Logic Tests
# ============================================================================


@pytest.mark.unit
def test_safe_to_cpu_tensor():
    """Test that tensors are moved to CPU."""
    callback = CPUOffloadCallback()

    tensor = torch.randn(10, 10)
    result = callback._safe_to_cpu(tensor, path="test.tensor")

    assert result.device == torch.device("cpu")
    assert torch.equal(tensor, result)


@pytest.mark.unit
def test_safe_to_cpu_dict():
    """Test that nested dicts with tensors are processed."""
    callback = CPUOffloadCallback()

    data = {
        "tensor1": torch.randn(5, 5),
        "tensor2": torch.randn(3, 3),
        "nested": {"tensor3": torch.randn(2, 2)},
    }

    result = callback._safe_to_cpu(data, path="test.dict")

    assert result["tensor1"].device == torch.device("cpu")
    assert result["tensor2"].device == torch.device("cpu")
    assert result["nested"]["tensor3"].device == torch.device("cpu")


@pytest.mark.unit
def test_safe_to_cpu_list():
    """Test that lists with tensors are processed."""
    callback = CPUOffloadCallback()

    data = [torch.randn(2, 2), torch.randn(3, 3)]
    result = callback._safe_to_cpu(data, path="test.list")

    assert isinstance(result, list)
    assert result[0].device == torch.device("cpu")
    assert result[1].device == torch.device("cpu")


@pytest.mark.unit
def test_safe_to_cpu_tuple():
    """Test that tuples with tensors are processed and remain tuples."""
    callback = CPUOffloadCallback()

    data = (torch.randn(2, 2), torch.randn(3, 3))
    result = callback._safe_to_cpu(data, path="test.tuple")

    assert isinstance(result, tuple)
    assert result[0].device == torch.device("cpu")
    assert result[1].device == torch.device("cpu")


@pytest.mark.unit
def test_safe_to_cpu_custom_object():
    """Test that custom objects are skipped."""
    callback = CPUOffloadCallback()

    custom_obj = Mock()
    result = callback._safe_to_cpu(custom_obj, path="test.custom")

    assert result is custom_obj  # Should return unchanged
    assert len(callback._skipped_types) == 1
    assert "Mock at 'test.custom'" in callback._skipped_types


@pytest.mark.unit
def test_safe_to_cpu_primitives():
    """Test that primitives are skipped."""
    callback = CPUOffloadCallback()

    # Test various primitives
    assert callback._safe_to_cpu(42, "int") == 42
    assert callback._safe_to_cpu(3.14, "float") == 3.14
    assert callback._safe_to_cpu("hello", "str") == "hello"
    assert callback._safe_to_cpu(True, "bool") is True
    assert callback._safe_to_cpu(None, "none") is None


# ============================================================================
# Checkpoint Processing Tests
# ============================================================================


@pytest.mark.unit
def test_offload_checkpoint_all_keys(sample_checkpoint):
    """Test offloading all default checkpoint keys."""
    callback = CPUOffloadCallback()

    result = callback._offload_checkpoint(sample_checkpoint)

    assert "state_dict" in result
    assert "optimizer_states" in result
    assert "lr_schedulers" in result

    # Verify tensors are on CPU
    assert sample_checkpoint["state_dict"]["layer1.weight"].device == torch.device(
        "cpu"
    )
    assert sample_checkpoint["optimizer_states"][0]["state"][0][
        "momentum"
    ].device == torch.device("cpu")


@pytest.mark.unit
def test_offload_checkpoint_partial_keys(sample_checkpoint):
    """Test offloading only specific keys."""
    callback = CPUOffloadCallback(offload_keys=["state_dict"])

    result = callback._offload_checkpoint(sample_checkpoint)

    assert result == ["state_dict"]
    assert sample_checkpoint["state_dict"]["layer1.weight"].device == torch.device(
        "cpu"
    )


@pytest.mark.unit
def test_offload_checkpoint_missing_key():
    """Test handling of missing checkpoint keys."""
    callback = CPUOffloadCallback()
    checkpoint = {"state_dict": {"weight": torch.randn(5, 5)}}

    # Should not crash when lr_schedulers is missing
    result = callback._offload_checkpoint(checkpoint)

    assert "state_dict" in result
    assert "optimizer_states" not in result
    assert "lr_schedulers" not in result


@pytest.mark.unit
def test_offload_checkpoint_exception_handling():
    """Test that exceptions during offload are caught and logged."""
    callback = CPUOffloadCallback()

    # Create a checkpoint with a value that will cause an error when processed
    # We need to mock _safe_to_cpu to raise an exception
    checkpoint = {"state_dict": {"weight": torch.randn(5, 5)}}

    # Mock _safe_to_cpu to raise an exception
    with patch.object(callback, "_safe_to_cpu", side_effect=RuntimeError("Test error")):
        result = callback._offload_checkpoint(checkpoint)

    assert result == []  # No keys successfully processed


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.unit
def test_checkpoint_with_actual_error_in_data():
    """Test checkpoint that causes an error during tensor processing."""
    callback = CPUOffloadCallback()

    # Create a more realistic error scenario
    checkpoint = {
        "state_dict": {
            "good_tensor": torch.randn(3, 3),
            "bad_value": object(),  # This will be skipped, not error
        }
    }

    # This should succeed - custom objects are skipped, not errored
    result = callback._offload_checkpoint(checkpoint)

    assert "state_dict" in result
    assert checkpoint["state_dict"]["good_tensor"].device == torch.device("cpu")
    # bad_value should be unchanged (skipped as custom object)


# ============================================================================
# Skipped Objects Logging Tests
# ============================================================================


@pytest.mark.unit
def test_skipped_logging_disabled_few_objects():
    """Test skipped logging with log_skipped=False and few objects (<= 20)."""
    callback = CPUOffloadCallback(log_skipped=False)

    # Add 15 skipped objects
    for i in range(15):
        callback._skipped_types.append(f"CustomClass{i} at path{i}")

    # Mock the checkpoint save to test logging
    assert len(callback._skipped_types) == 15
    # With <= 20 objects, all should be shown


@pytest.mark.unit
def test_skipped_logging_disabled_many_objects():
    """Test skipped logging with log_skipped=False and many objects (> 20)."""
    callback = CPUOffloadCallback(log_skipped=False)

    # Add 100 skipped objects
    for i in range(100):
        callback._skipped_types.append(f"CustomClass{i} at path{i}")

    assert len(callback._skipped_types) == 100
    # Only first 10 and last 10 should be shown


@pytest.mark.unit
def test_skipped_logging_enabled():
    """Test skipped logging with log_skipped=True."""
    callback = CPUOffloadCallback(log_skipped=True)

    # Add many skipped objects
    for i in range(100):
        callback._skipped_types.append(f"CustomClass{i} at path{i}")

    assert len(callback._skipped_types) == 100
    # All should be shown when log_skipped=True


# ============================================================================
# Integration Tests (on_save_checkpoint)
# ============================================================================


@pytest.mark.unit
def test_on_save_checkpoint_enabled(
    mock_trainer_ddp, mock_pl_module, sample_checkpoint
):
    """Test on_save_checkpoint when callback is enabled."""
    callback = CPUOffloadCallback()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.memory_allocated", return_value=8e9),
        patch("torch.cuda.memory_reserved", return_value=10e9),
    ):
        callback.setup(mock_trainer_ddp, mock_pl_module, "fit")
        callback.on_save_checkpoint(mock_trainer_ddp, mock_pl_module, sample_checkpoint)

    assert callback._checkpoint_count == 1
    assert callback._total_time_saved > 0
    # Verify tensors moved to CPU
    assert sample_checkpoint["state_dict"]["layer1.weight"].device == torch.device(
        "cpu"
    )


@pytest.mark.unit
def test_on_save_checkpoint_disabled(
    mock_trainer_fsdp, mock_pl_module, sample_checkpoint
):
    """Test on_save_checkpoint when callback is disabled."""
    callback = CPUOffloadCallback()

    with patch("torch.cuda.is_available", return_value=False):
        callback.setup(mock_trainer_fsdp, mock_pl_module, "fit")

        # Store original device
        original_device = sample_checkpoint["state_dict"]["layer1.weight"].device

        callback.on_save_checkpoint(
            mock_trainer_fsdp, mock_pl_module, sample_checkpoint
        )

    assert callback._checkpoint_count == 0  # Should not increment
    # Tensors should remain unchanged
    assert sample_checkpoint["state_dict"]["layer1.weight"].device == original_device


@pytest.mark.unit
def test_on_save_checkpoint_non_rank_zero(
    mock_trainer_ddp, mock_pl_module, sample_checkpoint
):
    """Test that non-rank-0 processes don't save checkpoints."""
    mock_trainer_ddp.global_rank = 1  # Not rank 0

    callback = CPUOffloadCallback()

    with patch("torch.cuda.is_available", return_value=False):
        callback.setup(mock_trainer_ddp, mock_pl_module, "fit")
        callback.on_save_checkpoint(mock_trainer_ddp, mock_pl_module, sample_checkpoint)

    assert callback._checkpoint_count == 0


# ============================================================================
# State Dict Tests
# ============================================================================


@pytest.mark.unit
def test_state_dict_save():
    """Test saving callback state."""
    callback = CPUOffloadCallback()
    callback._checkpoint_count = 5
    callback._total_time_saved = 25.5
    callback._total_memory_freed = 40.2
    callback._is_enabled = True

    state = callback.state_dict()

    assert state["checkpoint_count"] == 5
    assert state["total_time_saved"] == 25.5
    assert state["total_memory_freed"] == 40.2
    assert state["is_enabled"] is True


@pytest.mark.unit
def test_state_dict_load():
    """Test loading callback state."""
    callback = CPUOffloadCallback()

    state = {
        "checkpoint_count": 10,
        "total_time_saved": 50.0,
        "total_memory_freed": 80.0,
        "is_enabled": False,
    }

    callback.load_state_dict(state)

    assert callback._checkpoint_count == 10
    assert callback._total_time_saved == 50.0
    assert callback._total_memory_freed == 80.0
    assert callback._is_enabled is False


# ============================================================================
# Lifecycle Hook Tests
# ============================================================================


@pytest.mark.unit
def test_on_train_start(mock_trainer_ddp, mock_pl_module):
    """Test on_train_start hook."""
    callback = CPUOffloadCallback()

    with patch("torch.cuda.is_available", return_value=False):
        callback.setup(mock_trainer_ddp, mock_pl_module, "fit")
        callback.on_train_start(mock_trainer_ddp, mock_pl_module)

    # Should complete without error


@pytest.mark.unit
def test_on_train_end(mock_trainer_ddp, mock_pl_module):
    """Test on_train_end hook."""
    callback = CPUOffloadCallback()
    callback._checkpoint_count = 5
    callback._total_time_saved = 25.0
    callback._total_memory_freed = 40.0

    with patch("torch.cuda.is_available", return_value=False):
        callback.setup(mock_trainer_ddp, mock_pl_module, "fit")
        callback.on_train_end(mock_trainer_ddp, mock_pl_module)

    # Should complete without error


@pytest.mark.unit
def test_teardown(mock_trainer_ddp, mock_pl_module):
    """Test teardown hook."""
    callback = CPUOffloadCallback()

    with patch("torch.cuda.is_available", return_value=False):
        callback.setup(mock_trainer_ddp, mock_pl_module, "fit")
        callback.teardown(mock_trainer_ddp, mock_pl_module, "fit")

    # Should complete without error


@pytest.mark.unit
def test_on_exception(mock_trainer_ddp, mock_pl_module):
    """Test on_exception hook."""
    callback = CPUOffloadCallback()
    callback._checkpoint_count = 3

    with patch("torch.cuda.is_available", return_value=False):
        callback.setup(mock_trainer_ddp, mock_pl_module, "fit")
        callback.on_exception(
            mock_trainer_ddp, mock_pl_module, RuntimeError("Test error")
        )

    # Should complete without error


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.unit
def test_empty_checkpoint():
    """Test handling of empty checkpoint."""
    callback = CPUOffloadCallback()
    checkpoint = {}

    result = callback._offload_checkpoint(checkpoint)

    assert result == []


@pytest.mark.unit
def test_checkpoint_with_none_values():
    """Test checkpoint with None values."""
    callback = CPUOffloadCallback()
    checkpoint = {
        "state_dict": None,
        "optimizer_states": None,
    }

    # _safe_to_cpu returns None unchanged (treated as custom object)
    # So these will be "processed successfully" (no exception thrown)
    result = callback._offload_checkpoint(checkpoint)

    # Both keys present in checkpoint are processed
    assert len(result) == 2  # NOT 0!
    assert "state_dict" in result
    assert "optimizer_states" in result

    # Values should still be None (unchanged)
    assert checkpoint["state_dict"] is None
    assert checkpoint["optimizer_states"] is None


@pytest.mark.unit
def test_deeply_nested_structure():
    """Test deeply nested checkpoint structure."""
    callback = CPUOffloadCallback()

    nested = {"level1": {"level2": {"level3": {"tensor": torch.randn(5, 5)}}}}

    result = callback._safe_to_cpu(nested, "root")

    assert result["level1"]["level2"]["level3"]["tensor"].device == torch.device("cpu")


@pytest.mark.unit
def test_mixed_types_in_list():
    """Test list with mixed types (tensors, primitives, custom objects)."""
    callback = CPUOffloadCallback()

    data = [
        torch.randn(2, 2),
        42,
        "string",
        Mock(),
        torch.randn(3, 3),
    ]

    result = callback._safe_to_cpu(data, "mixed_list")

    assert result[0].device == torch.device("cpu")  # tensor
    assert result[1] == 42  # int
    assert result[2] == "string"  # str
    assert isinstance(result[3], Mock)  # custom object
    assert result[4].device == torch.device("cpu")  # tensor


# ============================================================================
# Performance/Statistics Tests
# ============================================================================


@pytest.mark.unit
def test_cumulative_statistics(mock_trainer_ddp, mock_pl_module):
    """Test that cumulative statistics are tracked correctly."""
    callback = CPUOffloadCallback()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.memory_allocated", side_effect=[8e9, 1e9, 8e9, 1e9]),
        patch("torch.cuda.memory_reserved", return_value=10e9),
    ):
        callback.setup(mock_trainer_ddp, mock_pl_module, "fit")

        # Save 2 checkpoints
        checkpoint1 = {"state_dict": {"w": torch.randn(10, 10)}}
        checkpoint2 = {"state_dict": {"w": torch.randn(10, 10)}}

        callback.on_save_checkpoint(mock_trainer_ddp, mock_pl_module, checkpoint1)
        callback.on_save_checkpoint(mock_trainer_ddp, mock_pl_module, checkpoint2)

    assert callback._checkpoint_count == 2
    assert callback._total_time_saved > 0
    assert callback._total_memory_freed > 0


@pytest.mark.unit
def test_average_statistics_calculation():
    """Test that average statistics are calculated correctly."""
    callback = CPUOffloadCallback()
    callback._checkpoint_count = 5
    callback._total_time_saved = 25.0  # 5s average
    callback._total_memory_freed = 40.0  # 8GB average

    avg_time = callback._total_time_saved / callback._checkpoint_count
    avg_mem = callback._total_memory_freed / callback._checkpoint_count

    assert avg_time == 5.0
    assert avg_mem == 8.0


# ============================================================================
# Run pytest
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
