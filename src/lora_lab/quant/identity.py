"""No-op quantizer: keeps the base weight in its original precision."""

from __future__ import annotations

import torch
from torch import Tensor

from lora_lab.quant.base import QuantizedTensor, Quantizer
from lora_lab.registry import QUANTIZERS, register


@register(QUANTIZERS, "none")
class IdentityQuantizer(Quantizer):
    """Exact identity quantizer (round-trip error is zero)."""

    def quantize(self, w: Tensor) -> QuantizedTensor:
        """Store ``w`` unchanged."""
        values = w.detach().clone().reshape(-1)
        scale = torch.empty(0, dtype=torch.float32, device=w.device)
        return QuantizedTensor(
            values=values,
            scale=scale,
            shape=tuple(w.shape),
            block_size=0,
            bits=values.element_size() * 8,
            dtype=w.dtype,
        )

    def dequantize(self, q: QuantizedTensor) -> Tensor:
        """Return the stored tensor unchanged."""
        return q.values.reshape(q.shape).to(dtype=q.dtype)

    def storage_bytes(self, q: QuantizedTensor) -> int:
        """Bytes for the dense values; no scale metadata."""
        return int(q.values.numel() * q.values.element_size())
