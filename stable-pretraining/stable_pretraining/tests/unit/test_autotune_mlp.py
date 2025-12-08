"""Unit tests for AutoTuneMLP module."""

import pytest
import torch
import torch.nn as nn
from stable_pretraining.backbone import AutoTuneMLP

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return {
        "in_features": 10,
        "out_features": 3,
        "hidden_features": [20, 15],
        "name": "test_mlp",
        "loss_fn": nn.CrossEntropyLoss(),
    }


@pytest.fixture
def simple_data():
    """Simple data for forward pass testing."""
    batch_size = 4
    in_features = 10
    out_features = 3
    x = torch.randn(batch_size, in_features)
    y = torch.randint(0, out_features, (batch_size,))
    return x, y


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
class TestInitialization:
    """Test AutoTuneMLP initialization."""

    def test_basic_initialization(self, basic_config):
        """Test basic initialization with default parameters."""
        model = AutoTuneMLP(**basic_config)

        assert model.in_features == 10
        assert model.out_features == 3
        assert model.name == "test_mlp"
        assert len(model.mlp) > 0

    def test_single_architecture(self, basic_config):
        """Test initialization with single architecture."""
        model = AutoTuneMLP(**basic_config)

        # Should normalize to list of lists
        assert isinstance(model.hidden_features, list)
        assert isinstance(model.hidden_features[0], list)
        assert model.hidden_features == [[20, 15]]

    def test_multiple_architectures(self, basic_config):
        """Test initialization with multiple architectures."""
        basic_config["hidden_features"] = [[10], [20, 10], [30, 20, 10]]
        model = AutoTuneMLP(**basic_config)

        assert len(model.hidden_features) == 3
        assert model.hidden_features[0] == [10]
        assert model.hidden_features[1] == [20, 10]
        assert model.hidden_features[2] == [30, 20, 10]

    def test_linear_model_empty_list(self, basic_config):
        """Test initialization with empty list for linear model."""
        basic_config["hidden_features"] = []
        model = AutoTuneMLP(**basic_config)

        assert model.hidden_features == [[]]
        assert len(model.mlp) > 0

    def test_multiple_hyperparameters(self, basic_config):
        """Test initialization with multiple hyperparameter values."""
        basic_config.update(
            {
                "dropout": [0.0, 0.1, 0.2],
                "lr_scaling": [0.1, 1.0],
                "additional_weight_decay": [0.0, 0.01],
                "normalization": ["none", "bn"],
                "activation": ["relu", "tanh"],
            }
        )
        model = AutoTuneMLP(**basic_config)

        # Should create: 1 arch * 3 dropout * 2 lr * 2 wd * 2 norm * 2 act = 48 variants
        assert len(model.mlp) == 48

    def test_single_hyperparameter_values(self, basic_config):
        """Test that single values are properly converted to lists."""
        basic_config.update(
            {
                "dropout": 0.5,
                "lr_scaling": 0.1,
                "additional_weight_decay": 0.01,
            }
        )
        model = AutoTuneMLP(**basic_config)

        # Should still create variants
        assert len(model.mlp) > 0


# ============================================================================
# Architecture Tests
# ============================================================================


@pytest.mark.unit
class TestArchitectures:
    """Test different architecture configurations."""

    def test_linear_architecture(self, basic_config):
        """Test linear model (no hidden layers)."""
        basic_config["hidden_features"] = []
        model = AutoTuneMLP(**basic_config)

        # Get one variant
        key = model.keys()[0]
        variant = model.get_variant(key)

        # Should have only one Linear layer
        linear_layers = [m for m in variant if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 1
        assert linear_layers[0].in_features == 10
        assert linear_layers[0].out_features == 3

    def test_single_hidden_layer(self, basic_config):
        """Test model with single hidden layer."""
        basic_config["hidden_features"] = [50]
        model = AutoTuneMLP(**basic_config)

        key = model.keys()[0]
        variant = model.get_variant(key)

        # Should have 2 Linear layers
        linear_layers = [m for m in variant if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 2

    def test_deep_architecture(self, basic_config):
        """Test model with multiple hidden layers."""
        basic_config["hidden_features"] = [100, 50, 25, 10]
        model = AutoTuneMLP(**basic_config)

        key = model.keys()[0]
        variant = model.get_variant(key)

        # Should have 5 Linear layers (4 hidden + 1 output)
        linear_layers = [m for m in variant if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 5

    def test_architecture_naming(self, basic_config):
        """Test that architectures are properly named in variant IDs."""
        basic_config["hidden_features"] = [[], [50], [100, 50]]
        model = AutoTuneMLP(**basic_config)

        keys = model.keys()

        # Check that linear architecture is named
        linear_keys = [k for k in keys if "linear" in k]
        assert len(linear_keys) > 0

        # Check that other architectures are named
        arch_keys = [k for k in keys if "arch" in k]
        assert len(arch_keys) > 0


# ============================================================================
# Normalization Tests
# ============================================================================


@pytest.mark.unit
class TestNormalization:
    """Test normalization layer configurations."""

    def test_no_normalization(self, basic_config):
        """Test model without normalization."""
        basic_config["normalization"] = "none"
        model = AutoTuneMLP(**basic_config)

        key = model.keys()[0]
        variant = model.get_variant(key)

        # Should not have any normalization layers
        bn_layers = [m for m in variant if isinstance(m, nn.BatchNorm1d)]
        ln_layers = [m for m in variant if isinstance(m, nn.LayerNorm)]
        assert len(bn_layers) == 0
        assert len(ln_layers) == 0

    def test_batch_normalization(self, basic_config):
        """Test model with batch normalization."""
        basic_config["normalization"] = "bn"
        basic_config["hidden_features"] = [20, 15]
        model = AutoTuneMLP(**basic_config)

        key = model.keys()[0]
        variant = model.get_variant(key)

        # Should have BatchNorm layers
        bn_layers = [m for m in variant if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) > 0

    def test_layer_normalization(self, basic_config):
        """Test model with layer normalization."""
        basic_config["normalization"] = "norm"
        basic_config["hidden_features"] = [20, 15]
        model = AutoTuneMLP(**basic_config)

        key = model.keys()[0]
        variant = model.get_variant(key)

        # Should have LayerNorm layers
        ln_layers = [m for m in variant if isinstance(m, nn.LayerNorm)]
        assert len(ln_layers) > 0

    def test_multiple_normalizations(self, basic_config):
        """Test creating variants with different normalizations."""
        basic_config["normalization"] = ["none", "bn", "norm"]
        basic_config["hidden_features"] = [20]
        model = AutoTuneMLP(**basic_config)

        # Should create 3 variants (one for each normalization)
        assert len(model.mlp) >= 3


# ============================================================================
# Activation Tests
# ============================================================================


@pytest.mark.unit
class TestActivations:
    """Test activation function configurations."""

    @pytest.mark.parametrize(
        "activation,expected_class",
        [
            ("relu", nn.ReLU),
            ("leaky_relu", nn.LeakyReLU),
            ("tanh", nn.Tanh),
        ],
    )
    def test_activation_types(self, basic_config, activation, expected_class):
        """Test different activation functions."""
        basic_config["activation"] = activation
        basic_config["hidden_features"] = [20]
        model = AutoTuneMLP(**basic_config)

        key = model.keys()[0]
        variant = model.get_variant(key)

        # Should have activation layers of the expected type
        act_layers = [m for m in variant if isinstance(m, expected_class)]
        assert len(act_layers) > 0

    def test_multiple_activations(self, basic_config):
        """Test creating variants with different activations."""
        basic_config["activation"] = ["relu", "tanh"]
        basic_config["hidden_features"] = [20]
        model = AutoTuneMLP(**basic_config)

        # Should create at least 2 variants
        assert len(model.mlp) >= 2


# ============================================================================
# Dropout Tests
# ============================================================================


@pytest.mark.unit
class TestDropout:
    """Test dropout configurations."""

    def test_zero_dropout(self, basic_config):
        """Test model with zero dropout."""
        basic_config["dropout"] = 0.0
        basic_config["hidden_features"] = [20]
        model = AutoTuneMLP(**basic_config)

        key = model.keys()[0]
        variant = model.get_variant(key)

        # Should still have Dropout layers (with p=0)
        dropout_layers = [m for m in variant if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) > 0

    def test_nonzero_dropout(self, basic_config):
        """Test model with non-zero dropout."""
        basic_config["dropout"] = 0.5
        basic_config["hidden_features"] = [20, 15]
        model = AutoTuneMLP(**basic_config)

        key = model.keys()[0]
        variant = model.get_variant(key)

        # Should have Dropout layers
        dropout_layers = [m for m in variant if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) > 0
        # Verify dropout rate
        assert dropout_layers[0].p == 0.5

    def test_multiple_dropout_rates(self, basic_config):
        """Test creating variants with different dropout rates."""
        basic_config["dropout"] = [0.0, 0.3, 0.5]
        model = AutoTuneMLP(**basic_config)

        # Should create 3 variants
        assert len(model.mlp) >= 3


# ============================================================================
# Forward Pass Tests
# ============================================================================


@pytest.mark.unit
class TestForward:
    """Test forward pass functionality."""

    def test_forward_without_targets(self, basic_config, simple_data):
        """Test forward pass without target labels."""
        x, _ = simple_data
        model = AutoTuneMLP(**basic_config)

        outputs = model(x)

        # Should return predictions for all variants
        pred_keys = [k for k in outputs.keys() if k.startswith("pred/")]
        assert len(pred_keys) == len(model.mlp)

        # Should not return losses
        loss_keys = [k for k in outputs.keys() if k.startswith("loss/")]
        assert len(loss_keys) == 0

    def test_forward_with_targets(self, basic_config, simple_data):
        """Test forward pass with target labels."""
        x, y = simple_data
        model = AutoTuneMLP(**basic_config)

        outputs = model(x, y)

        # Should return predictions and losses
        pred_keys = [k for k in outputs.keys() if k.startswith("pred/")]
        loss_keys = [k for k in outputs.keys() if k.startswith("loss/")]

        assert len(pred_keys) == len(model.mlp)
        assert len(loss_keys) == len(model.mlp)

    def test_output_shapes(self, basic_config, simple_data):
        """Test output tensor shapes."""
        x, y = simple_data
        batch_size = x.shape[0]

        model = AutoTuneMLP(**basic_config)
        outputs = model(x, y)

        # Check prediction shapes
        for key, pred in outputs.items():
            if key.startswith("pred/"):
                assert pred.shape == (batch_size, basic_config["out_features"])
            elif key.startswith("loss/"):
                assert pred.dim() == 0  # Scalar loss

    def test_linear_model_forward(self, basic_config, simple_data):
        """Test forward pass with linear model."""
        basic_config["hidden_features"] = []
        x, y = simple_data

        model = AutoTuneMLP(**basic_config)
        outputs = model(x, y)

        # Should produce valid outputs
        pred_keys = [k for k in outputs.keys() if k.startswith("pred/")]
        assert len(pred_keys) > 0

        # Check output shape
        first_pred = outputs[pred_keys[0]]
        assert first_pred.shape[1] == basic_config["out_features"]

    def test_batch_independence(self, basic_config):
        """Test that different batch sizes work correctly."""
        model = AutoTuneMLP(**basic_config)

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, basic_config["in_features"])
            outputs = model(x)

            pred_key = list(outputs.keys())[0]
            assert outputs[pred_key].shape[0] == batch_size


# ============================================================================
# Keys and Variant Access Tests
# ============================================================================


@pytest.mark.unit
class TestKeysAndAccess:
    """Test keys() method and variant access."""

    def test_keys_method(self, basic_config):
        """Test keys() method returns list of strings."""
        model = AutoTuneMLP(**basic_config)

        keys = model.keys()

        assert isinstance(keys, list)
        assert len(keys) > 0
        assert all(isinstance(k, str) for k in keys)

    def test_keys_match_mlp_dict(self, basic_config):
        """Test that keys() matches internal ModuleDict."""
        model = AutoTuneMLP(**basic_config)

        keys = model.keys()
        dict_keys = list(model.mlp.keys())

        assert set(keys) == set(dict_keys)

    def test_get_variant(self, basic_config):
        """Test get_variant() method."""
        model = AutoTuneMLP(**basic_config)

        key = model.keys()[0]
        variant = model.get_variant(key)

        assert isinstance(variant, nn.Sequential)

    def test_get_variant_invalid_key(self, basic_config):
        """Test get_variant() with invalid key raises error."""
        model = AutoTuneMLP(**basic_config)

        with pytest.raises(KeyError):
            model.get_variant("nonexistent_key")

    def test_len_method(self, basic_config):
        """Test __len__() method."""
        model = AutoTuneMLP(**basic_config)

        assert len(model) == len(model.mlp)
        assert len(model) == model.num_variants()


# ============================================================================
# Best Variant Tests
# ============================================================================


@pytest.mark.unit
class TestBestVariant:
    """Test get_best_variant() functionality."""

    def test_get_best_variant_lower_is_better(self):
        """Test get_best_variant with lower_is_better=True (e.g., loss)."""
        model = AutoTuneMLP(
            in_features=10,
            out_features=3,
            hidden_features=[20],
            name="test",
            loss_fn=nn.CrossEntropyLoss(),
            dropout=[0.0, 0.1, 0.2],  # Ensure at least 3 variants
        )

        keys = model.keys()
        assert len(keys) >= 3

        # Create fake metrics with known values
        metrics = {keys[0]: 0.5, keys[1]: 0.3, keys[2]: 0.8}

        best = model.get_best_variant(metrics, lower_is_better=True)
        assert best == keys[1]  # Should pick the one with 0.3 (lowest)

    def test_get_best_variant_higher_is_better(self):
        """Test get_best_variant with lower_is_better=False (e.g., accuracy)."""
        model = AutoTuneMLP(
            in_features=10,
            out_features=3,
            hidden_features=[20],
            name="test",
            loss_fn=nn.CrossEntropyLoss(),
            dropout=[0.0, 0.1, 0.2],  # Ensure at least 3 variants
        )

        keys = model.keys()
        assert len(keys) >= 3

        # Create fake metrics with known values
        metrics = {keys[0]: 0.85, keys[1]: 0.92, keys[2]: 0.78}

        best = model.get_best_variant(metrics, lower_is_better=False)
        assert best == keys[1]  # Should pick the one with 0.92 (highest)

    def test_get_best_variant_from_outputs(self, basic_config, simple_data):
        """Test finding best variant from actual forward pass."""
        x, y = simple_data

        # Create model with multiple variants
        basic_config["dropout"] = [0.0, 0.1]
        model = AutoTuneMLP(**basic_config)

        outputs = model(x, y)

        # Extract losses
        losses = {
            k.replace("loss/", ""): v.item()
            for k, v in outputs.items()
            if k.startswith("loss/")
        }

        assert len(losses) > 0

        best = model.get_best_variant(losses)
        assert best in model.keys()

        # Verify it's actually the minimum loss
        assert losses[best] == min(losses.values())

    def test_get_best_variant_single_variant(self, basic_config):
        """Test get_best_variant works with only one variant."""
        # Default config creates 1 variant
        model = AutoTuneMLP(**basic_config)

        keys = model.keys()
        assert len(keys) >= 1

        metrics = {keys[0]: 0.5}

        best = model.get_best_variant(metrics, lower_is_better=True)
        assert best == keys[0]

    def test_get_best_variant_all_equal(self):
        """Test get_best_variant when all metrics are equal."""
        model = AutoTuneMLP(
            in_features=10,
            out_features=3,
            hidden_features=[20],
            name="test",
            loss_fn=nn.CrossEntropyLoss(),
            dropout=[0.0, 0.1],
        )

        keys = model.keys()
        # All have same value
        metrics = {key: 0.5 for key in keys}

        best = model.get_best_variant(metrics, lower_is_better=True)
        assert best in keys  # Should return one of them


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_deep_network(self, basic_config):
        """Test with very deep network."""
        basic_config["hidden_features"] = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        model = AutoTuneMLP(**basic_config)

        assert len(model.mlp) > 0

        # Test forward pass
        x = torch.randn(2, basic_config["in_features"])
        outputs = model(x)
        assert len(outputs) > 0

    def test_wide_network(self, basic_config):
        """Test with very wide network."""
        basic_config["hidden_features"] = [1000, 1000]
        model = AutoTuneMLP(**basic_config)

        x = torch.randn(2, basic_config["in_features"])
        outputs = model(x)
        assert len(outputs) > 0

    def test_single_unit_hidden_layer(self, basic_config):
        """Test with single unit hidden layer."""
        basic_config["hidden_features"] = [1]
        model = AutoTuneMLP(**basic_config)

        x = torch.randn(2, basic_config["in_features"])
        outputs = model(x)
        assert len(outputs) > 0

    def test_many_architectures(self, basic_config):
        """Test with many different architectures."""
        basic_config["hidden_features"] = [
            [],
            [10],
            [20, 10],
            [30, 20, 10],
            [40, 30, 20, 10],
        ]
        model = AutoTuneMLP(**basic_config)

        # Should create variants for all architectures
        keys = model.keys()
        assert "linear" in " ".join(keys)
        assert len(keys) >= 5

    def test_repr_method(self, basic_config):
        """Test __repr__() method."""
        model = AutoTuneMLP(**basic_config)

        repr_str = repr(model)

        assert "AutoTuneMLP" in repr_str
        assert basic_config["name"] in repr_str
        assert str(len(model.mlp)) in repr_str


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.unit
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, simple_data):
        """Test complete workflow: create, forward, find best."""
        x, y = simple_data

        # Create model with multiple configurations
        model = AutoTuneMLP(
            in_features=10,
            out_features=3,
            hidden_features=[[], [20], [20, 10]],
            name="test",
            loss_fn=nn.CrossEntropyLoss(),
            dropout=[0.0, 0.1],
            normalization=["none", "bn"],
            activation=["relu", "tanh"],
        )

        # Forward pass
        outputs = model(x, y)

        # Get all variant keys
        keys = model.keys()
        assert len(keys) > 0

        # Extract losses
        losses = {
            k.replace("loss/", ""): v.item()
            for k, v in outputs.items()
            if k.startswith("loss/")
        }

        # Find best variant
        best = model.get_best_variant(losses)
        assert best in keys

        # Get best model
        best_model = model.get_variant(best)
        assert isinstance(best_model, nn.Sequential)

    def test_gradient_flow(self, basic_config, simple_data):
        """Test that gradients flow correctly through all variants."""
        x, y = simple_data
        model = AutoTuneMLP(**basic_config)

        outputs = model(x, y)

        # Take first loss and backprop
        loss_key = [k for k in outputs.keys() if k.startswith("loss/")][0]
        loss = outputs[loss_key]
        loss.backward()

        # Check that gradients exist
        variant_key = loss_key.replace("loss/", "")
        variant = model.get_variant(variant_key)

        has_grad = False
        for param in variant.parameters():
            if param.grad is not None:
                has_grad = True
                break

        assert has_grad, "No gradients found in variant"

    def test_multiple_forward_passes(self, basic_config):
        """Test multiple forward passes with different batch sizes."""
        model = AutoTuneMLP(**basic_config)

        for _ in range(3):
            batch_size = torch.randint(1, 10, (1,)).item()
            x = torch.randn(batch_size, basic_config["in_features"])
            y = torch.randint(0, basic_config["out_features"], (batch_size,))

            outputs = model(x, y)

            # Verify all outputs
            assert len(outputs) == 2 * len(model.mlp)


# ============================================================================
# Parameterized Tests
# ============================================================================


@pytest.mark.unit
class TestParameterized:
    """Parameterized tests for various configurations."""

    @pytest.mark.parametrize(
        "in_features,out_features",
        [
            (10, 2),
            (50, 10),
            (100, 50),
            (1, 1),
        ],
    )
    def test_various_input_output_dims(self, in_features, out_features):
        """Test with various input/output dimensions."""
        model = AutoTuneMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_features=[20],
            name="test",
            loss_fn=nn.MSELoss(),
        )

        x = torch.randn(4, in_features)
        outputs = model(x)

        pred_key = list(outputs.keys())[0]
        assert outputs[pred_key].shape == (4, out_features)

    @pytest.mark.parametrize(
        "hidden_features",
        [
            [],
            [10],
            [50, 25],
            [100, 75, 50, 25],
        ],
    )
    def test_various_architectures(self, hidden_features):
        """Test with various architecture depths."""
        model = AutoTuneMLP(
            in_features=20,
            out_features=5,
            hidden_features=hidden_features,
            name="test",
            loss_fn=nn.CrossEntropyLoss(),
        )

        x = torch.randn(4, 20)
        outputs = model(x)

        assert len(outputs) > 0

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 32, 64])
    def test_various_batch_sizes(self, basic_config, batch_size):
        """Test with various batch sizes."""
        model = AutoTuneMLP(**basic_config)

        x = torch.randn(batch_size, basic_config["in_features"])
        outputs = model(x)

        for key, value in outputs.items():
            if key.startswith("pred/"):
                assert value.shape[0] == batch_size
