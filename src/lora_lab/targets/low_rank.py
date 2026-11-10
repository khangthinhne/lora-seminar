"""Low-rank synthetic target: ``delta_star = U @ V.T`` plus optional small noise."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lora_lab.registry import TARGETS, register
from lora_lab.targets.base import Target, normalize_delta, required_float, required_int


@register(TARGETS, "low_rank")
class LowRankTarget(Target):
    """Rank-``true_rank`` target, normalized to ``delta_scale * ||W0||_F``."""

    def make_delta(self, d: int, k: int, rng: torch.Generator, **cfg: Any) -> Tensor:
        """Return ``delta_star`` of shape ``(d, k)``.

        Required cfg keys: ``true_rank``, ``reference_norm``, ``delta_scale``.
        Optional: ``noise_std`` (default ``0.0``).
        """
        true_rank = required_int(cfg, "true_rank")
        reference_norm = required_float(cfg, "reference_norm", strictly_positive=True)
        delta_scale = required_float(cfg, "delta_scale", strictly_positive=True)
        noise_std = required_float(cfg, "noise_std") if "noise_std" in cfg else 0.0
        if noise_std < 0:
            raise ValueError(f"target config 'noise_std' must be >= 0, got {noise_std}")
        if not 1 <= true_rank <= min(d, k):
            raise ValueError(
                f"true_rank must satisfy 1 <= true_rank <= {min(d, k)}, got {true_rank}"
            )

        u = torch.randn(d, true_rank, generator=rng)
        v = torch.randn(k, true_rank, generator=rng)
        delta = u @ v.T
        if noise_std > 0:
            delta = delta + noise_std * torch.randn(d, k, generator=rng)
        return normalize_delta(delta, reference_norm, delta_scale)
