"""Mandatory interface implemented by every adaptation method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from torch import Tensor, nn


class AdaptedLayer(nn.Module, ABC):
    """A frozen base weight ``W0`` of shape ``(d, k)`` plus a trainable update.

    ``forward`` maps ``(..., k) -> (..., d)``. The generic training loop,
    metrics, experiments, tables, and figures only use this interface, so a new
    method automatically participates in every experiment once registered.

    Attributes:
        zero_init: Whether fresh initialization yields a zero update. If true,
            the contract test asserts ``effective_delta() == 0`` before training.
        default_hparams: Hyperparameters used by the generic contract test.
    """

    zero_init: ClassVar[bool] = False
    default_hparams: ClassVar[dict[str, Any]] = {}

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Map ``x`` of shape ``(..., k)`` to ``(..., d)``."""

    @abstractmethod
    def effective_delta(self) -> Tensor:
        """Return the dense update ``W_eff - W0`` with shape ``(d, k)``."""

    @abstractmethod
    def merge(self) -> nn.Linear:
        """Return an equivalent plain ``nn.Linear`` (inference form, no extra latency)."""

    def trainable_param_count(self) -> int:
        """Number of parameters with ``requires_grad=True``."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @classmethod
    @abstractmethod
    def from_base(cls, w0: Tensor, **hparams: Any) -> AdaptedLayer:
        """Build the adapted layer around a frozen base weight ``w0`` of shape ``(d, k)``."""
