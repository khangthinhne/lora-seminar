"""Full fine-tuning baseline expressed as a dense delta.

This is equivalent to directly training ``W``: it has the same parameter count
and the same optimization problem, but keeps ``W0`` frozen so that both the
baseline and LoRA represent the same ``W' = W0 + delta``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lora_lab.methods.base import AdaptedLayer
from lora_lab.registry import METHODS, register


@register(METHODS, "full_ft")
class FullFTDelta(AdaptedLayer):
    """Frozen ``W0`` plus a trainable dense ``delta`` of shape ``(d, k)``.

    The update starts at zero, so the initial model equals the base model.
    """

    zero_init = True
    default_hparams: dict[str, Any] = {}

    def __init__(self, w0: Tensor) -> None:
        super().__init__()
        if w0.ndim != 2:
            raise ValueError(f"w0 must be 2-D (d, k), got shape {tuple(w0.shape)}")
        self.register_buffer("w0", w0.detach().clone())
        self.delta = nn.Parameter(torch.zeros_like(self.w0))

    @classmethod
    def from_base(cls, w0: Tensor, **hparams: Any) -> FullFTDelta:
        """Build the baseline; only the optional ``init_generator`` is accepted."""
        init_generator = hparams.pop("init_generator", None)
        if hparams:
            raise ValueError(f"full_ft received unknown hyperparameters: {sorted(hparams)}")
        del init_generator
        return cls(w0)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(..., k) -> (..., d)`` using ``x @ (W0 + delta).T``."""
        base = F.linear(x, self.w0)
        return base + F.linear(x, self.delta)

    def effective_delta(self) -> Tensor:
        """Return the dense update with shape ``(d, k)``."""
        return self.delta

    def merge(self) -> nn.Linear:
        """Return an ``nn.Linear`` with weight ``W0 + delta``."""
        d, k = self.w0.shape
        linear = nn.Linear(k, d, bias=False, device=self.w0.device, dtype=self.w0.dtype)
        with torch.no_grad():
            linear.weight.copy_(self.w0 + self.effective_delta())
        return linear
