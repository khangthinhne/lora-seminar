"""Analytical memory estimates for one adapted layer.

This is an estimate, not a measurement. Activations are deliberately excluded:
that omission is exactly why trainable parameter count is not the same as GPU
memory. No GPU is used or required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BYTES_PER_ELEMENT: dict[str, int] = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
}


@dataclass(frozen=True)
class MemoryEstimate:
    """Analytical byte counts for one layer (activations excluded).

    Attributes:
        weight_bytes: Storage for the frozen base weight (dense or quantized).
        gradient_bytes: Storage for gradients of trainable parameters.
        optimizer_bytes: Storage for optimizer state of trainable parameters.
    """

    weight_bytes: int
    gradient_bytes: int
    optimizer_bytes: int

    @property
    def total_bytes(self) -> int:
        """Sum of weights, gradients, and optimizer state; activations excluded."""
        return self.weight_bytes + self.gradient_bytes + self.optimizer_bytes

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary including ``total_bytes``."""
        result = asdict(self)
        result["total_bytes"] = self.total_bytes
        return result


def dense_weight_bytes(d: int, k: int, dtype_bytes: int = 4) -> int:
    """Bytes for a dense ``(d, k)`` weight in ``dtype_bytes``-byte precision."""
    if d <= 0 or k <= 0:
        raise ValueError(f"d and k must be > 0, got d={d}, k={k}")
    if dtype_bytes <= 0:
        raise ValueError(f"dtype_bytes must be > 0, got {dtype_bytes}")
    return d * k * dtype_bytes


def estimate_training_memory(
    weight_bytes: int,
    trainable_params: int,
    *,
    gradient_bytes_per_param: int = 4,
    optimizer_states: int = 2,
    optimizer_bytes_per_state: int = 4,
) -> MemoryEstimate:
    """Estimate weights, gradients, and optimiser state for one layer.

    Args:
        weight_bytes: Bytes used by the (possibly quantized) frozen base weight.
        trainable_params: Number of trainable adapter parameters.
        gradient_bytes_per_param: Gradient precision (Adam default: float32).
        optimizer_states: Adam uses two moments per trainable parameter.
        optimizer_bytes_per_state: Precision of each optimizer moment.

    Returns:
        A :class:`MemoryEstimate`. Activations are excluded by design.
    """
    if weight_bytes < 0:
        raise ValueError(f"weight_bytes must be >= 0, got {weight_bytes}")
    if trainable_params < 0:
        raise ValueError(f"trainable_params must be >= 0, got {trainable_params}")
    if gradient_bytes_per_param <= 0:
        raise ValueError("gradient_bytes_per_param must be > 0")
    if optimizer_states < 0:
        raise ValueError("optimizer_states must be >= 0")
    if optimizer_bytes_per_state <= 0:
        raise ValueError("optimizer_bytes_per_state must be > 0")
    return MemoryEstimate(
        weight_bytes=int(weight_bytes),
        gradient_bytes=int(trainable_params * gradient_bytes_per_param),
        optimizer_bytes=int(trainable_params * optimizer_states * optimizer_bytes_per_state),
    )
