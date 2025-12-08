"""Modular tensor aggregation module for feeding multi-scale/multi-layer features to MLPs.

Commonly used for:
- SSL linear probes using multiple transformer layers
- Multi-scale feature fusion
- Combining features from different network stages
"""

from typing import Union, List, Dict, Optional, Literal
import torch
import torch.nn as nn
from loguru import logger


AggregationMode = Literal["mean", "max", "cls", "flatten", "adaptive"]
TensorInput = Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]


class TensorAggregator(nn.Module):
    """Aggregates multi-dimensional tensors into 2D format for MLP input.

    Pure aggregation module with NO trainable parameters.
    Handles various input formats and aggregation strategies.

    Args:
        input_spec: Specification of input format and aggregation modes:
            - str: Single aggregation mode for all tensors (e.g., "mean")
            - List[str]: Per-tensor aggregation modes for list inputs
            - Dict[str, str]: Per-key aggregation modes for dict inputs
        adaptive_pool_size: Output size for adaptive pooling (default: 1)

    Aggregation Modes:
        - "mean": Spatial/temporal mean pooling
        - "max": Spatial/temporal max pooling
        - "cls": Take first token (for transformers with [CLS] token)
        - "flatten": Flatten all dimensions after batch
        - "adaptive": Adaptive average pooling to fixed size

    Examples:
        >>> # Single tensor with mean pooling
        >>> agg = TensorAggregator("mean")
        >>> x = torch.randn(4, 768, 14, 14)
        >>> out = agg(x)  # Shape: (4, 768)

        >>> # SSL: Last 4 transformer layers with CLS token
        >>> agg = TensorAggregator(["cls", "cls", "cls", "cls"])
        >>> layers = [torch.randn(4, 197, 768) for _ in range(4)]
        >>> out = agg(layers)  # Shape: (4, 3072)  # 768 * 4

        >>> # Multi-scale features
        >>> agg = TensorAggregator({"layer1": "cls", "layer2": "mean", "conv": "mean"})
        >>> out = agg(
        ...     {
        ...         "layer1": torch.randn(4, 197, 768),
        ...         "layer2": torch.randn(4, 197, 768),
        ...         "conv": torch.randn(4, 512, 14, 14),
        ...     }
        ... )  # Shape: (4, 2048)
    """

    def __init__(
        self,
        input_spec: Union[str, List[str], Dict[str, str]],
        adaptive_pool_size: int = 1,
    ):
        super().__init__()

        self.input_spec = input_spec
        self.adaptive_pool_size = adaptive_pool_size

        # Determine input type
        if isinstance(input_spec, str):
            self.input_type = "single"
            self.agg_modes = {"default": input_spec}
        elif isinstance(input_spec, list):
            self.input_type = "list"
            self.agg_modes = {i: mode for i, mode in enumerate(input_spec)}
        elif isinstance(input_spec, dict):
            self.input_type = "dict"
            self.agg_modes = input_spec
        else:
            raise ValueError(f"Invalid input_spec type: {type(input_spec)}")

        # Validate aggregation modes
        valid_modes = {"mean", "max", "cls", "flatten", "adaptive"}
        for mode in self.agg_modes.values():
            if mode not in valid_modes:
                raise ValueError(
                    f"Invalid aggregation mode: {mode}. Valid modes: {valid_modes}"
                )

        logger.info(f"Initialized TensorAggregator with {self.input_type} input")
        logger.debug(f"Aggregation modes: {self.agg_modes}")

    def _aggregate_single_tensor(
        self, x: torch.Tensor, mode: str, key: Optional[Union[str, int]] = None
    ) -> torch.Tensor:
        """Aggregate a single tensor to 2D based on aggregation mode.

        Args:
            x: Input tensor of shape (B, ..., D) or (B, D, H, W)
            mode: Aggregation mode
            key: Optional key for logging

        Returns:
            2D tensor of shape (B, features)
        """
        batch_size = x.shape[0]
        original_shape = x.shape

        logger.trace(f"Aggregating {key or 'tensor'}: {x.shape} using '{mode}'")

        # Already 2D - nothing to do!
        if x.ndim == 2:
            logger.trace(f"Already 2D: {x.shape}")
            return x

        # 3D: (B, L, D) - sequence data
        elif x.ndim == 3:
            result = self._aggregate_3d(x, mode, key)

        # 4D: (B, C, H, W) - image/feature maps
        elif x.ndim == 4:
            result = self._aggregate_4d(x, mode, key)

        # 5D: (B, C, T, H, W) - video/3D data
        elif x.ndim == 5:
            result = self._aggregate_5d(x, mode, key)

        else:
            raise ValueError(
                f"Unsupported tensor dimension: {x.ndim}. Supported: 2 (no-op), 3, 4, 5"
            )

        # Ensure output is 2D
        if result.ndim != 2:
            result = result.reshape(batch_size, -1)

        logger.trace(f"Aggregated {original_shape} -> {result.shape}")
        return result

    def _aggregate_3d(
        self, x: torch.Tensor, mode: str, key: Optional[Union[str, int]]
    ) -> torch.Tensor:
        """Aggregate 3D tensor (B, L, D) to 2D."""
        if mode == "mean":
            return x.mean(dim=1)  # (B, D)

        elif mode == "max":
            return x.max(dim=1)[0]  # (B, D)

        elif mode == "cls":
            return x[:, 0]  # (B, D) - first token

        elif mode == "flatten":
            return x.reshape(x.shape[0], -1)  # (B, L*D)

        elif mode == "adaptive":
            # Pool sequence dimension to fixed size
            return nn.functional.adaptive_avg_pool1d(
                x.transpose(1, 2),
                self.adaptive_pool_size,  # (B, D, L)
            ).squeeze(-1)  # (B, D)

        else:
            raise ValueError(f"Mode '{mode}' not supported for 3D tensors")

    def _aggregate_4d(
        self, x: torch.Tensor, mode: str, key: Optional[Union[str, int]]
    ) -> torch.Tensor:
        """Aggregate 4D tensor (B, C, H, W) to 2D."""
        batch_size = x.shape[0]

        if mode == "mean":
            return x.mean(dim=(2, 3))  # (B, C)

        elif mode == "max":
            return x.amax(dim=(2, 3))  # (B, C)

        elif mode == "adaptive":
            return nn.functional.adaptive_avg_pool2d(
                x, (self.adaptive_pool_size, self.adaptive_pool_size)
            ).reshape(batch_size, -1)  # (B, C * pool_size^2)

        elif mode == "flatten":
            return x.reshape(batch_size, -1)  # (B, C*H*W)

        elif mode == "cls":
            logger.warning(
                f"Using 'cls' on 4D tensor, taking [0,0] spatial position. "
                f"Consider 'mean' or 'adaptive' instead. Shape: {x.shape}"
            )
            return x[:, :, 0, 0]  # (B, C)

        else:
            raise ValueError(f"Mode '{mode}' not supported for 4D tensors")

    def _aggregate_5d(
        self, x: torch.Tensor, mode: str, key: Optional[Union[str, int]]
    ) -> torch.Tensor:
        """Aggregate 5D tensor (B, C, T, H, W) to 2D."""
        batch_size = x.shape[0]

        if mode == "mean":
            return x.mean(dim=(2, 3, 4))  # (B, C)

        elif mode == "max":
            return x.amax(dim=(2, 3, 4))  # (B, C)

        elif mode == "adaptive":
            pool_size = self.adaptive_pool_size
            return nn.functional.adaptive_avg_pool3d(
                x, (pool_size, pool_size, pool_size)
            ).reshape(batch_size, -1)  # (B, C * pool_size^3)

        elif mode == "flatten":
            return x.reshape(batch_size, -1)  # (B, C*T*H*W)

        else:
            raise ValueError(
                f"Mode '{mode}' not supported for 5D tensors. "
                f"Use: mean, max, adaptive, flatten"
            )

    def forward(self, x: TensorInput) -> torch.Tensor:
        """Aggregate input tensor(s) to 2D format.

        Args:
            x: Input tensor, list of tensors, or dict of tensors

        Returns:
            Aggregated 2D tensor of shape (B, total_features)
        """
        # Single tensor
        if isinstance(x, torch.Tensor):
            if self.input_type != "single":
                logger.warning(
                    f"Expected {self.input_type} input but got single tensor"
                )
            mode = self.agg_modes.get("default", "mean")
            return self._aggregate_single_tensor(x, mode)

        # List of tensors
        elif isinstance(x, list):
            if self.input_type == "single":
                mode = self.agg_modes["default"]
                aggregated = [
                    self._aggregate_single_tensor(tensor, mode, i)
                    for i, tensor in enumerate(x)
                ]
            else:
                if len(x) != len(self.agg_modes):
                    logger.warning(
                        f"Number of tensors ({len(x)}) != number of modes "
                        f"({len(self.agg_modes)})"
                    )

                aggregated = []
                for i, tensor in enumerate(x):
                    mode = self.agg_modes.get(i, list(self.agg_modes.values())[0])
                    agg = self._aggregate_single_tensor(tensor, mode, i)
                    aggregated.append(agg)

            result = torch.cat(aggregated, dim=1)
            logger.debug(f"Concatenated {len(aggregated)} tensors -> {result.shape}")
            return result

        # Dict of tensors (sorted for determinism)
        elif isinstance(x, dict):
            if self.input_type == "single":
                mode = self.agg_modes["default"]
                aggregated = [
                    self._aggregate_single_tensor(tensor, mode, key)
                    for key, tensor in sorted(x.items())
                ]
            else:
                aggregated = []
                for key, tensor in sorted(x.items()):
                    mode = self.agg_modes.get(key, "mean")
                    if key not in self.agg_modes:
                        logger.warning(f"No mode specified for '{key}', using 'mean'")
                    agg = self._aggregate_single_tensor(tensor, mode, key)
                    aggregated.append(agg)

            result = torch.cat(aggregated, dim=1)
            logger.debug(
                f"Concatenated {len(aggregated)} dict entries -> {result.shape}"
            )
            return result

        else:
            raise TypeError(
                f"Unsupported input type: {type(x)}. "
                f"Expected Tensor, List[Tensor], or Dict[str, Tensor]"
            )

    def compute_output_dim(
        self, input_shapes: Union[tuple, List[tuple], Dict[str, tuple]]
    ) -> int:
        """Compute the output dimension given input shapes.

        Args:
            input_shapes: Shape(s) of input tensor(s) (excluding batch dim)

        Returns:
            Total output features

        Examples:
            >>> agg = TensorAggregator(["cls", "mean"])
            >>> agg.compute_output_dim([(197, 768), (197, 768)])
            1536

            >>> agg = TensorAggregator({"l1": "cls", "conv": "mean"})
            >>> agg.compute_output_dim({"l1": (197, 768), "conv": (512, 14, 14)})
            1280
        """

        def _compute_single_dim(shape: tuple, mode: str) -> int:
            """Compute output dim for a single tensor."""
            ndim = len(shape)

            # Already 2D
            if ndim == 1:
                return shape[0]

            # 3D tensor (seq_len, features)
            elif ndim == 2:
                if mode in ["cls", "mean", "max"]:
                    return shape[1]
                elif mode == "flatten":
                    return shape[0] * shape[1]
                elif mode == "adaptive":
                    return shape[1] * self.adaptive_pool_size

            # 4D tensor (channels, height, width)
            elif ndim == 3:
                if mode in ["mean", "max", "cls"]:
                    return shape[0]
                elif mode == "flatten":
                    return shape[0] * shape[1] * shape[2]
                elif mode == "adaptive":
                    return shape[0] * (self.adaptive_pool_size**2)

            # 5D tensor (channels, time, height, width)
            elif ndim == 4:
                if mode in ["mean", "max"]:
                    return shape[0]
                elif mode == "flatten":
                    return shape[0] * shape[1] * shape[2] * shape[3]
                elif mode == "adaptive":
                    return shape[0] * (self.adaptive_pool_size**3)

            raise ValueError(f"Cannot compute dim for shape {shape} with mode {mode}")

        # Single input
        if isinstance(input_shapes, tuple):
            mode = self.agg_modes.get("default", "mean")
            return _compute_single_dim(input_shapes, mode)

        # List of inputs
        elif isinstance(input_shapes, list):
            total = 0
            for i, shape in enumerate(input_shapes):
                mode = self.agg_modes.get(i, list(self.agg_modes.values())[0])
                total += _compute_single_dim(shape, mode)
            return total

        # Dict of inputs
        elif isinstance(input_shapes, dict):
            total = 0
            for key, shape in input_shapes.items():
                mode = self.agg_modes.get(key, "mean")
                total += _compute_single_dim(shape, mode)
            return total

        else:
            raise TypeError(f"Unsupported input_shapes type: {type(input_shapes)}")

    def __repr__(self) -> str:
        return (
            f"TensorAggregator(type={self.input_type}, "
            f"modes={self.agg_modes}, "
            f"adaptive_pool_size={self.adaptive_pool_size})"
        )
