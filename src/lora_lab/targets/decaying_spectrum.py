"""Full-rank synthetic target with a decaying singular-value spectrum."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lora_lab.registry import TARGETS, register
from lora_lab.targets.base import Target, normalize_delta, required_float


@register(TARGETS, "decaying_spectrum")
class DecayingSpectrumTarget(Target):
    """``delta_star = Q_d diag(s) Q_k.T`` with ``s_i proportional to i ** -gamma``."""

    def make_delta(self, d: int, k: int, rng: torch.Generator, **cfg: Any) -> Tensor:
        """Return a full-rank ``(d, k)`` target.

        Required cfg keys: ``gamma``, ``reference_norm``, ``delta_scale``.
        """
        gamma = required_float(cfg, "gamma", strictly_positive=True)
        reference_norm = required_float(cfg, "reference_norm", strictly_positive=True)
        delta_scale = required_float(cfg, "delta_scale", strictly_positive=True)

        rank = min(d, k)
        q_d, _ = torch.linalg.qr(torch.randn(d, rank, generator=rng))
        q_k, _ = torch.linalg.qr(torch.randn(k, rank, generator=rng))
        indices = torch.arange(1, rank + 1, dtype=torch.float32)
        singular_values = indices.pow(-gamma)
        delta = (q_d * singular_values) @ q_k.T
        return normalize_delta(delta, reference_norm, delta_scale)
