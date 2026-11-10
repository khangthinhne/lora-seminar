"""Quantizer interface and the container returned by ``quantize``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class QuantizedTensor:
    """Storage for a quantized tensor.

    Attributes:
        values: Integer (or floating, for identity) codes; flattened.
        scale: Per-block scale factors of shape ``(n_blocks,)``.
        shape: Original tensor shape.
        block_size: Number of elements per scale; ``0`` for identity.
        bits: Nominal bits per stored value.
        dtype: Data type used for dequantization.
    """

    values: Tensor
    scale: Tensor
    shape: tuple[int, ...]
    block_size: int
    bits: int
    dtype: torch.dtype

    def numel(self) -> int:
        """Number of original elements."""
        count = 1
        for size in self.shape:
            count *= size
        return count


class Quantizer(ABC):
    """Turns a floating-point weight into a storage-efficient representation."""

    @abstractmethod
    def quantize(self, w: Tensor) -> QuantizedTensor:
        """Quantize ``w`` (any shape)."""

    @abstractmethod
    def dequantize(self, q: QuantizedTensor) -> Tensor:
        """Reconstruct a tensor with the original shape."""

    @abstractmethod
    def storage_bytes(self, q: QuantizedTensor) -> int:
        """Return bytes used by codes, scales, and metadata."""
