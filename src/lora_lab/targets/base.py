"""Target interface and small configuration helpers for synthetic targets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class Target(ABC):
    """Produces the ground-truth update ``delta_star`` with shape ``(d, k)``."""

    @abstractmethod
    def make_delta(self, d: int, k: int, rng: torch.Generator, **cfg: Any) -> Tensor:
        """Return ``delta_star`` of shape ``(d, k)``, already scaled per ``cfg``."""


def required_float(cfg: dict[str, Any], key: str, *, strictly_positive: bool = False) -> float:
    """Read a required float from a target config with a clear error message."""
    if key not in cfg:
        raise ValueError(f"target config is missing required key {key!r}")
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"target config {key!r} must be a number, got {type(value).__name__}")
    number = float(value)
    if strictly_positive and number <= 0:
        raise ValueError(f"target config {key!r} must be > 0, got {number}")
    return number


def required_int(cfg: dict[str, Any], key: str) -> int:
    """Read a required integer from a target config with a clear error message."""
    if key not in cfg:
        raise ValueError(f"target config is missing required key {key!r}")
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"target config {key!r} must be an integer, got {type(value).__name__}")
    return value


def normalize_delta(delta: Tensor, reference_norm: float, delta_scale: float) -> Tensor:
    """Scale ``delta`` so that ``||delta||_F = delta_scale * reference_norm``.

    Args:
        delta: Tensor of shape ``(d, k)``.
        reference_norm: Frobenius norm of the base weight ``W0``.
        delta_scale: Desired ``||delta||_F / ||W0||_F`` ratio.
    """
    if reference_norm <= 0:
        raise ValueError(f"reference_norm must be > 0, got {reference_norm}")
    if delta_scale <= 0:
        raise ValueError(f"delta_scale must be > 0, got {delta_scale}")
    current = torch.linalg.matrix_norm(delta)
    if float(current) <= 0:
        raise ValueError("cannot normalize a zero target delta")
    return delta * (delta_scale * reference_norm / current)
