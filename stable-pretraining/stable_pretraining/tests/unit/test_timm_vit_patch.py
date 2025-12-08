"""Unit tests for EfficientMaskedTimmViT with register token support."""

import pytest
import torch
import torch.nn as nn
import timm

from stable_pretraining.backbone import EfficientMaskedTimmViT


# ============================================================================
# Test Configuration
# ============================================================================

# List of timm models to test
TIMM_MODELS = [
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
]

# Smaller set for quick tests
QUICK_MODELS = [
    "vit_tiny_patch16_224",
    "deit_tiny_patch16_224",
]

BATCH_SIZE = 4
IMAGE_SIZE = 224
CHANNELS = 3


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device():
    """Get CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture(params=QUICK_MODELS)
def model_name(request):
    """Fixture providing model names for parametrized tests."""
    return request.param


@pytest.fixture
def basic_vit_model(model_name, device):
    """Fixture providing a fresh timm ViT model without register tokens."""
    model = timm.create_model(model_name, pretrained=False, num_classes=1000)
    return model.to(device)


@pytest.fixture
def basic_masked_vit(basic_vit_model):
    """Fixture providing a wrapped masked ViT model without register tokens."""
    return EfficientMaskedTimmViT(basic_vit_model)


@pytest.fixture
def sample_input(device):
    """Fixture providing sample input images."""
    return torch.randn(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)


@pytest.fixture
def sample_patches(basic_vit_model, device):
    """Fixture providing sample patch embeddings."""
    x = torch.randn(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():
        patches = basic_vit_model.patch_embed(x)
        if patches.ndim == 4:
            patches = patches.flatten(1, 2)
    return patches


# Register token specific fixtures
@pytest.fixture(params=[0, 1, 4])
def num_register_tokens(request):
    """Test with different numbers of register tokens."""
    return request.param


@pytest.fixture
def reg_vit_model(num_register_tokens, device):
    """Create a ViT model with register tokens."""
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=100,
        reg_tokens=num_register_tokens,
        dynamic_img_size=True,
    )
    return model.to(device).eval()


@pytest.fixture
def reg_masked_vit(reg_vit_model):
    """Create wrapped masked ViT with register tokens."""
    return EfficientMaskedTimmViT(reg_vit_model)


# ============================================================================
# Basic Functionality Tests
# ============================================================================


@pytest.mark.unit
class TestBasicFunctionality:
    """Test basic operations of EfficientMaskedTimmViT."""

    def test_initialization(self, basic_vit_model):
        """Test that model initializes correctly."""
        masked_vit = EfficientMaskedTimmViT(basic_vit_model)
        assert masked_vit.vit is basic_vit_model
        assert hasattr(masked_vit, "forward")

    def test_initialization_invalid_model(self):
        """Test that initialization fails with invalid model."""
        invalid_model = nn.Linear(10, 10)
        with pytest.raises(RuntimeError, match="patch_embed"):
            EfficientMaskedTimmViT(invalid_model)

    def test_forward_no_nans(self, basic_masked_vit, sample_input):
        """Test forward pass with no NaN patches."""
        output = basic_masked_vit(sample_input)
        assert output.shape[0] == BATCH_SIZE
        assert not torch.isnan(output).any(), "Output should not contain NaNs"

    def test_output_shape(self, basic_masked_vit, basic_vit_model, sample_input):
        """Test that output shape matches original model."""
        basic_masked_vit.eval()
        basic_vit_model.eval()

        with torch.no_grad():
            masked_output = basic_masked_vit(sample_input)
            original_output = basic_vit_model(sample_input)

        assert masked_output.shape == original_output.shape, (
            f"Shape mismatch: {masked_output.shape} vs {original_output.shape}"
        )

    def test_deterministic_output(self, basic_masked_vit, sample_input):
        """Test that output is deterministic for same input."""
        basic_masked_vit.eval()
        with torch.no_grad():
            output1 = basic_masked_vit(sample_input)
            output2 = basic_masked_vit(sample_input)

        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)


# ============================================================================
# NaN Handling Tests
# ============================================================================


@pytest.mark.unit
class TestNaNHandling:
    """Test NaN patch handling functionality."""

    def test_with_nan_patches(self, basic_masked_vit, basic_vit_model, sample_input):
        """Test forward pass with NaN patches."""
        with torch.no_grad():
            patches = basic_vit_model.patch_embed(sample_input)
            if patches.ndim == 4:
                patches = patches.flatten(1, 2)

            # Add same number of NaN patches to each sample (but different locations)
            patches[0, 10:12, :] = float("nan")  # Patches 10-11 for sample 0
            patches[1, 20:22, :] = float("nan")  # Patches 20-21 for sample 1
            patches[2, 5:7, :] = float("nan")  # Patches 5-6 for sample 2
            patches[3, 15:17, :] = float("nan")  # Patches 15-16 for sample 3

        output = basic_masked_vit(patches)

        assert output.shape[0] == BATCH_SIZE
        assert not torch.isnan(output).any(), "Output should not contain NaNs"

    def test_all_nan_patches_raises_error(self, basic_masked_vit, sample_patches):
        """Test that all-NaN input raises appropriate error."""
        sample_patches[:] = float("nan")

        with pytest.raises(ValueError, match="All patches are NaN"):
            basic_masked_vit(sample_patches)

    def test_mismatched_nan_counts_raises_error(self, basic_masked_vit, sample_patches):
        """Test that mismatched NaN counts raise error."""
        # Different number of NaN patches per sample
        sample_patches[0, 10:12, :] = float("nan")  # 2 NaN patches
        sample_patches[1, 20:24, :] = float("nan")  # 4 NaN patches (different!)

        with pytest.raises(ValueError, match="same number of NaN patches"):
            basic_masked_vit(sample_patches)

    def test_single_nan_patch(self, basic_masked_vit, sample_patches):
        """Test with single NaN patch per sample."""
        sample_patches[0, 10, :] = float("nan")
        sample_patches[1, 20, :] = float("nan")
        sample_patches[2, 5, :] = float("nan")
        sample_patches[3, 15, :] = float("nan")

        output = basic_masked_vit(sample_patches)
        assert not torch.isnan(output).any()

    def test_many_nan_patches(self, basic_masked_vit, basic_vit_model, sample_input):
        """Test with many NaN patches (keep only few patches)."""
        with torch.no_grad():
            patches = basic_vit_model.patch_embed(sample_input)
            if patches.ndim == 4:
                patches = patches.flatten(1, 2)

            num_patches = patches.shape[1]
            keep_patches = 10  # Keep only 10 patches

            # Set all but 10 patches to NaN (different 10 for each sample)
            for i in range(BATCH_SIZE):
                mask = torch.ones(num_patches, dtype=torch.bool)
                keep_idx = torch.randperm(num_patches)[:keep_patches]
                mask[keep_idx] = False
                patches[i, mask, :] = float("nan")

        output = basic_masked_vit(patches)
        assert not torch.isnan(output).any()


# ============================================================================
# Output Correctness Tests
# ============================================================================


@pytest.mark.unit
class TestOutputCorrectness:
    """Test that outputs are mathematically correct."""

    def test_no_nan_matches_original(
        self, basic_masked_vit, basic_vit_model, sample_input
    ):
        """Test that output matches original model when no NaNs."""
        basic_masked_vit.eval()
        basic_vit_model.eval()

        with torch.no_grad():
            masked_output = basic_masked_vit(sample_input)
            original_output = basic_vit_model(sample_input)

        # Should be very close (minor floating point differences acceptable)
        torch.testing.assert_close(
            masked_output,
            original_output,
            rtol=1e-4,
            atol=1e-4,
            msg="Output should match original model when no NaN patches",
        )

    def test_consistent_output_for_same_nan_pattern(
        self, basic_masked_vit, sample_patches
    ):
        """Test that same NaN pattern produces same output."""
        basic_masked_vit.eval()

        # Create NaN pattern
        nan_patches = sample_patches.clone()
        for i in range(BATCH_SIZE):
            nan_patches[i, 10:12, :] = float("nan")

        with torch.no_grad():
            output1 = basic_masked_vit(nan_patches.clone())
            output2 = basic_masked_vit(nan_patches.clone())

        torch.testing.assert_close(output1, output2)


# ============================================================================
# Multi-Model Tests
# ============================================================================


@pytest.mark.parametrize("test_model_name", TIMM_MODELS)
@pytest.mark.unit
class TestMultipleModels:
    """Test across multiple timm models."""

    def test_model_compatibility(self, test_model_name, device):
        """Test that model works with various timm architectures."""
        vit = timm.create_model(test_model_name, pretrained=False)
        vit = vit.to(device)
        masked_vit = EfficientMaskedTimmViT(vit)

        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            output = masked_vit(x)

        assert output.shape[0] == 2, f"Failed for {test_model_name}"
        assert not torch.isnan(output).any(), f"NaN in output for {test_model_name}"

    def test_model_with_nans(self, test_model_name, device):
        """Test each model handles NaN patches correctly."""
        vit = timm.create_model(test_model_name, pretrained=False)
        vit = vit.to(device)
        masked_vit = EfficientMaskedTimmViT(vit)

        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            patches = vit.patch_embed(x)
            if patches.ndim == 4:
                patches = patches.flatten(1, 2)
            patches[0, 5:8, :] = float("nan")
            patches[1, 10:13, :] = float("nan")

        with torch.no_grad():
            output = masked_vit(patches)
        assert not torch.isnan(output).any(), (
            f"Failed NaN handling for {test_model_name}"
        )


# ============================================================================
# Gradient and Training Tests
# ============================================================================


@pytest.mark.unit
class TestGradientsAndTraining:
    """Test gradient flow and training compatibility."""

    def test_gradient_flow(self, basic_masked_vit, sample_input):
        """Test that gradients flow properly."""
        basic_masked_vit.train()

        # Forward pass
        output = basic_masked_vit(sample_input)
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check that some parameters have gradients
        has_grad = False
        for param in basic_masked_vit.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients found in model parameters"

    def test_gradient_flow_with_nans(self, basic_masked_vit, basic_vit_model, device):
        """Test gradient flow with NaN patches."""
        basic_masked_vit.train()

        x = torch.randn(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)

        with torch.no_grad():
            patches = basic_vit_model.patch_embed(x)
            if patches.ndim == 4:
                patches = patches.flatten(1, 2)
            patches[:, 10:12, :] = float("nan")

        patches.requires_grad = False  # Patches don't need grad, model params do
        output = basic_masked_vit(patches)
        loss = output.mean()
        loss.backward()

        # Check gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in basic_masked_vit.parameters()
        )
        assert has_grad

    def test_training_step(self, basic_masked_vit, sample_input):
        """Test a full training step."""
        basic_masked_vit.train()
        optimizer = torch.optim.Adam(basic_masked_vit.parameters(), lr=1e-4)

        # Forward
        output = basic_masked_vit(sample_input)
        target = torch.randint(0, 1000, (BATCH_SIZE,))
        loss = nn.functional.cross_entropy(output, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_sample_batch(self, basic_masked_vit, device):
        """Test with batch size of 1."""
        single_input = torch.randn(1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
        output = basic_masked_vit(single_input)
        assert output.shape[0] == 1

    def test_large_batch(self, basic_masked_vit, device):
        """Test with larger batch size."""
        large_input = torch.randn(32, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
        with torch.no_grad():
            output = basic_masked_vit(large_input)
        assert output.shape[0] == 32

    def test_partial_nans_in_patch(self, basic_masked_vit, sample_patches):
        """Test when only some elements in a patch are NaN."""
        # Set only some dimensions to NaN (not all)
        sample_patches[0, 10, :128] = float("nan")  # Half the dimensions
        sample_patches[1, 20, :128] = float("nan")
        sample_patches[2, 5, :128] = float("nan")
        sample_patches[3, 15, :128] = float("nan")

        output = basic_masked_vit(sample_patches)
        assert not torch.isnan(output).any()

    def test_eval_mode(self, basic_masked_vit, sample_input):
        """Test model in eval mode."""
        basic_masked_vit.eval()
        with torch.no_grad():
            output = basic_masked_vit(sample_input)
        assert not torch.isnan(output).any()


# ============================================================================
# Helper Function Tests
# ============================================================================


@pytest.mark.unit
class TestHelperMethods:
    """Test internal helper methods."""

    def test_get_num_extra_tokens(self, basic_masked_vit):
        """Test _get_num_extra_tokens method."""
        num_extra = basic_masked_vit._get_num_extra_tokens()
        assert isinstance(num_extra, int)
        assert num_extra >= 0
        assert num_extra <= 10  # Reasonable upper bound

    def test_get_num_pos_tokens(self, basic_masked_vit):
        """Test _get_num_pos_tokens method."""
        num_pos = basic_masked_vit._get_num_pos_tokens()
        assert isinstance(num_pos, int)
        assert num_pos >= 0
        assert num_pos <= 2  # Max is cls + dist

    def test_add_extra_tokens(self, basic_masked_vit, device):
        """Test _add_extra_tokens method."""
        embed_dim = basic_masked_vit.vit.embed_dim
        x = torch.randn(2, 10, embed_dim, device=device)
        x_with_tokens = basic_masked_vit._add_extra_tokens(x)

        num_extra = basic_masked_vit._get_num_extra_tokens()
        assert x_with_tokens.shape[1] == x.shape[1] + num_extra


# ============================================================================
# Register Token Specific Tests
# ============================================================================
@pytest.mark.unit
class TestRegisterTokens:
    """Test register token functionality."""

    def test_model_creation_with_register_tokens(self, num_register_tokens, device):
        """Test that model can be created with register tokens."""
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            reg_tokens=num_register_tokens,
            dynamic_img_size=True,
        )
        model = model.to(device)

        # Check register tokens exist
        if num_register_tokens > 0:
            assert hasattr(model, "reg_token")
            assert model.reg_token.shape[1] == num_register_tokens

        # Wrap it
        wrapped = EfficientMaskedTimmViT(model)
        assert wrapped._get_num_extra_tokens() == 1 + num_register_tokens  # 1 for CLS

    def test_num_extra_tokens_counting(self, reg_vit_model, num_register_tokens):
        """Test that extra tokens are counted correctly."""
        wrapped = EfficientMaskedTimmViT(reg_vit_model)

        # Should be: 1 (CLS) + num_register_tokens
        expected = 1 + num_register_tokens
        assert wrapped._get_num_extra_tokens() == expected

    def test_num_pos_tokens_counting(self, reg_vit_model):
        """Test that tokens with positional embeddings are counted correctly."""
        wrapped = EfficientMaskedTimmViT(reg_vit_model)

        # Only CLS token (and dist if present) - not including register
        # This returns base tokens, actual pos_embed may include register with dynamic_img_size
        expected = 1  # Just CLS
        assert wrapped._get_num_pos_tokens() == expected

    def test_forward_no_nan_with_register_tokens(self, reg_masked_vit, device):
        """Test forward pass without NaN (fast path) with register tokens."""
        x = torch.randn(2, 3, 224, 224, device=device)

        with torch.no_grad():
            output = reg_masked_vit(x)

        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == 100  # num_classes
        assert not torch.isnan(output).any()

    def test_forward_with_nan_same_pattern(self, reg_masked_vit, device):
        """Test forward pass with NaN patches (same pattern across batch)."""
        B, C, H, W = 2, 3, 224, 224
        x = torch.randn(B, C, H, W, device=device)

        x_patches = reg_masked_vit.vit.patch_embed(x)
        if x_patches.ndim == 4:
            x_patches = x_patches.flatten(1, 2)

        # Mask same patches across batch
        x_patches[:, :50, :] = float("nan")

        with torch.no_grad():
            output = reg_masked_vit(x_patches)

        assert output.shape[0] == B
        assert output.shape[1] == 100
        assert not torch.isnan(output).any()

    def test_forward_with_nan_different_patterns(self, reg_masked_vit, device):
        """Test forward pass with NaN patches (different patterns per sample)."""
        B, C, H, W = 2, 3, 224, 224
        x = torch.randn(B, C, H, W, device=device)

        x_patches = reg_masked_vit.vit.patch_embed(x)
        if x_patches.ndim == 4:
            x_patches = x_patches.flatten(1, 2)

        # Use fewer NaN patches - different patterns but same count
        x_patches[0, :30, :] = float("nan")
        x_patches[1, 50:80, :] = float("nan")

        with torch.no_grad():
            output = reg_masked_vit(x_patches)

        assert output.shape[0] == B
        assert output.shape[1] == 100
        assert not torch.isnan(output).any()

    def test_add_extra_tokens_order(self, reg_vit_model, num_register_tokens, device):
        """Test that tokens are added in correct order: [CLS, REG, PATCHES]."""
        wrapped = EfficientMaskedTimmViT(reg_vit_model)

        B, N, D = 2, 196, reg_vit_model.embed_dim
        patches = torch.randn(B, N, D, device=device)

        result = wrapped._add_extra_tokens(patches)

        # Expected sequence length: 1 (CLS) + num_register_tokens + N (patches)
        expected_len = 1 + num_register_tokens + N
        assert result.shape[1] == expected_len

        # Verify dimensions
        assert result.shape[0] == B
        assert result.shape[2] == D

    def test_positional_embeddings_structure(self, reg_vit_model, num_register_tokens):
        """Test that positional embeddings have correct structure with dynamic_img_size."""
        # With dynamic_img_size=True, timm INCLUDES register tokens in pos_embed
        # Structure: [CLS_pos, REG_pos, PATCH_pos]
        expected_pos_embed_len = (
            1 + num_register_tokens + 196
        )  # CLS + REG + patches (14x14)

        assert reg_vit_model.pos_embed.shape[1] == expected_pos_embed_len, (
            f"Expected pos_embed length {expected_pos_embed_len}, got {reg_vit_model.pos_embed.shape[1]}"
        )

    def test_pos_embed_application(self, reg_masked_vit, device, num_register_tokens):
        """Test that positional embeddings are applied correctly."""
        B, N, D = 2, 196, reg_masked_vit.vit.embed_dim
        patches = torch.randn(B, N, D, device=device)

        # Add extra tokens
        x = reg_masked_vit._add_extra_tokens(patches)

        # Get positional embeddings
        pos_embed = reg_masked_vit._subsample_pos_embed_same_pattern(
            torch.arange(N, device=device), B, N
        )

        # With dynamic_img_size=True, pos_embed includes register tokens
        # So pos_embed and x should have same length
        assert pos_embed.shape[1] == x.shape[1], (
            f"pos_embed length {pos_embed.shape[1]} should match x length {x.shape[1]}"
        )

    def test_cache_functionality(self, reg_masked_vit, device):
        """Test that batch indices cache works correctly."""
        B, num_keep = 4, 100

        # First call - cache miss
        idx1 = reg_masked_vit._get_batch_indices(B, num_keep, device)
        assert (B, num_keep, device) in reg_masked_vit._batch_indices_cache

        # Second call - cache hit
        idx2 = reg_masked_vit._get_batch_indices(B, num_keep, device)
        assert torch.equal(idx1, idx2)
        assert idx1 is idx2  # Should be same object

        # Clear cache
        reg_masked_vit.clear_cache()
        assert len(reg_masked_vit._batch_indices_cache) == 0

    def test_interpolate_pos_embed_with_register_tokens(self, device):
        """Test positional embedding interpolation includes register tokens."""
        # Create model with register tokens
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            reg_tokens=4,
            dynamic_img_size=True,
        ).to(device)

        wrapped = EfficientMaskedTimmViT(model)

        # With dynamic_img_size=True and reg_tokens=4:
        # pos_embed includes register tokens: [1, 201, D] (1 CLS + 4 REG + 196 patches)
        assert model.pos_embed.shape[1] == 201

        # Interpolate to different size
        N_new = 256  # 16x16 patches
        pos_embed_new = wrapped._interpolate_pos_embed(model.pos_embed, N_new)

        # Should be: 1 (CLS) + 4 (REG) + 256 (new patches) = 261
        assert pos_embed_new.shape[1] == 261

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(self, reg_vit_model, batch_size, device):
        """Test model works with different batch sizes."""
        wrapped = EfficientMaskedTimmViT(reg_vit_model)
        x = torch.randn(batch_size, 3, 224, 224, device=device)

        with torch.no_grad():
            output = wrapped(x)

        assert output.shape[0] == batch_size
        assert not torch.isnan(output).any()

    def test_output_shape_consistency(self, device):
        """Test that output shape is consistent regardless of register tokens."""
        outputs = []

        for n_reg in [0, 1, 4]:
            model = timm.create_model(
                "vit_tiny_patch16_224",
                pretrained=False,
                num_classes=100,
                reg_tokens=n_reg,
                dynamic_img_size=True,
            ).to(device)

            wrapped = EfficientMaskedTimmViT(model)
            x = torch.randn(2, 3, 224, 224, device=device)

            with torch.no_grad():
                output = wrapped(x)

            outputs.append(output.shape)

        # All outputs should have same shape (register tokens are internal)
        assert all(shape == outputs[0] for shape in outputs)

    def test_gradient_flow_with_register_tokens(self, reg_vit_model, device):
        """Test that gradients flow correctly with register tokens."""
        wrapped = EfficientMaskedTimmViT(reg_vit_model)
        wrapped.train()

        x = torch.randn(2, 3, 224, 224, device=device, requires_grad=True)

        output = wrapped(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check model parameters have gradients
        for name, param in wrapped.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ============================================================================
# Regression Tests
# ============================================================================


@pytest.mark.unit
class TestRegression:
    """Tests to prevent regressions in specific scenarios."""

    def test_deit_distillation_token(self, device):
        """Test DeiT models with distillation token specifically."""
        deit = timm.create_model("deit_tiny_patch16_224", pretrained=False)
        deit = deit.to(device)
        masked_deit = EfficientMaskedTimmViT(deit)

        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            output = masked_deit(x)

        # Check actual model structure instead of assuming
        num_extra = masked_deit._get_num_extra_tokens()
        has_cls = hasattr(deit, "cls_token") and deit.cls_token is not None
        has_dist = hasattr(deit, "dist_token") and deit.dist_token is not None
        expected_num = int(has_cls) + int(has_dist)

        assert num_extra == expected_num, (
            f"Expected {expected_num} tokens (cls={has_cls}, dist={has_dist}), got {num_extra}"
        )
        assert not torch.isnan(output).any()

    def test_very_small_keep_set(self, basic_vit_model, device):
        """Test with very few patches kept (edge case)."""
        masked_vit = EfficientMaskedTimmViT(basic_vit_model)

        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            patches = basic_vit_model.patch_embed(x)
            if patches.ndim == 4:
                patches = patches.flatten(1, 2)

            # Keep only 3 patches
            patches[:, 3:, :] = float("nan")

        with torch.no_grad():
            output = masked_vit(patches)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
