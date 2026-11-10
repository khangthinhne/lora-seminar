"""Simple symmetric block absmax quantization (used to simulate QLoRA).

This is NOT NF4. It uses signed ``bits``-bit codes in ``[-2**(bits-1), 2**(bits-1) - 1]``
and one floating-point scale per block of ``block_size`` values.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lora_lab.quant.base import QuantizedTensor, Quantizer
from lora_lab.registry import QUANTIZERS, register


@register(QUANTIZERS, "absmax_block4")
class AbsmaxBlockQuantizer(Quantizer):
    """Per-block symmetric quantization with ``2**bits`` codes.

    Args:
        block_size: Elements sharing one scale (default ``64``).
        bits: Nominal bits per code (default ``4``).
    """

    def __init__(self, block_size: int = 64, bits: int = 4) -> None:
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError(f"block_size must be a positive integer, got {block_size}")
        if not isinstance(bits, int) or not 1 <= bits <= 8:
            raise ValueError(f"bits must satisfy 1 <= bits <= 8, got {bits}")
        self.block_size = block_size
        self.bits = bits

    def quantize(self, w: Tensor) -> QuantizedTensor:
        """Quantize ``w`` block-wise using ``scale = block_absmax / 2**(bits-1)``."""
        flat = w.detach().reshape(-1)
        numel = flat.numel()
        block_size = self.block_size
        padded = math.ceil(numel / block_size) * block_size
        if padded > numel:
            padding = torch.zeros(padded - numel, dtype=flat.dtype, device=flat.device)
            flat = torch.cat([flat, padding])

        blocks = flat.reshape(-1, block_size)
        absmax = blocks.abs().amax(dim=1)
        qmax = 2 ** (self.bits - 1)
        scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
        codes = torch.round(blocks / scale[:, None]).clamp(-qmax, qmax - 1).to(torch.int8)
        return QuantizedTensor(
            values=codes.reshape(-1),
            scale=scale,
            shape=tuple(w.shape),
            block_size=block_size,
            bits=self.bits,
            dtype=w.dtype,
        )

    def dequantize(self, q: QuantizedTensor) -> Tensor:
        """Reconstruct the original shape from codes and per-block scales."""
        blocks = q.values.to(dtype=q.dtype).reshape(-1, q.block_size)
        flat = (blocks * q.scale[:, None]).reshape(-1)
        return flat[: q.numel()].reshape(q.shape)

    def storage_bytes(self, q: QuantizedTensor) -> int:
        """Bytes for packed codes plus float32 scales (metadata included)."""
        code_bytes = math.ceil(q.numel() * self.bits / 8)
        scale_bytes = int(q.scale.numel() * q.scale.element_size())
        return code_bytes + scale_bytes
