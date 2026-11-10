"""LoRA: ``W' = W0 + (alpha / r) * B @ A`` with ``A`` of shape ``(r, k)``."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lora_lab.methods.base import AdaptedLayer
from lora_lab.registry import METHODS, register


@register(METHODS, "lora")
class LoRALinear(AdaptedLayer):
    """Low-rank adaptation of a frozen ``w0`` of shape ``(d, k)``.

    Shapes:
        ``A``: ``(r, k)``, initialized from a Gaussian with standard deviation
            ``init_std``.
        ``B``: ``(d, r)``, initialized to zero so the update starts at zero.
        ``effective_delta``: ``(alpha / r) * B @ A`` with shape ``(d, k)``.

    The forward pass uses the rank-``r`` path and never materializes a dense
    ``(d, k)`` update.
    """

    zero_init = True
    default_hparams: dict[str, Any] = {"r": 2, "alpha": 4.0, "init_std": 0.01}

    def __init__(
        self,
        w0: Tensor,
        r: int,
        alpha: float,
        init_std: float,
        init_generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if w0.ndim != 2:
            raise ValueError(f"w0 must be 2-D (d, k), got shape {tuple(w0.shape)}")
        d, k = w0.shape
        if not isinstance(r, int) or isinstance(r, bool):
            raise ValueError(f"r must be an integer, got {type(r).__name__}")
        if r < 1 or r > min(d, k):
            raise ValueError(f"r must satisfy 1 <= r <= min(d, k) = {min(d, k)}, got {r}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {init_std}")

        self.register_buffer("w0", w0.detach().clone())
        self.r = r
        self.alpha = float(alpha)
        self.init_std = float(init_std)
        self.scaling = self.alpha / self.r

        dtype = self.w0.dtype
        device = self.w0.device
        self.A = nn.Parameter(torch.empty(r, k, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.zeros(d, r, dtype=dtype, device=device))
        with torch.no_grad():
            self.A.normal_(mean=0.0, std=self.init_std, generator=init_generator)

    @classmethod
    def from_base(cls, w0: Tensor, **hparams: Any) -> LoRALinear:
        """Build LoRA from ``w0`` and the ``r``/``alpha``/``init_std`` hyperparameters."""
        init_generator = hparams.pop("init_generator", None)
        try:
            r = hparams.pop("r")
            alpha = hparams.pop("alpha")
        except KeyError as exc:
            raise ValueError(f"lora requires hyperparameter {exc.args[0]!r}") from exc
        init_std = hparams.pop("init_std", cls.default_hparams["init_std"])
        if hparams:
            raise ValueError(f"lora received unknown hyperparameters: {sorted(hparams)}")
        return cls(w0, r=r, alpha=alpha, init_std=init_std, init_generator=init_generator)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(..., k) -> (..., d)`` via ``x @ W0.T + s * (x @ A.T) @ B.T``."""
        base = F.linear(x, self.w0)
        low_rank = F.linear(F.linear(x, self.A), self.B)
        return base + self.scaling * low_rank

    def effective_delta(self) -> Tensor:
        """Return ``(alpha / r) * B @ A`` with shape ``(d, k)``."""
        return self.scaling * (self.B @ self.A)

    def merge(self) -> nn.Linear:
        """Return an ``nn.Linear`` whose weight is ``W0 + effective_delta()``."""
        d, k = self.w0.shape
        linear = nn.Linear(k, d, bias=False, device=self.w0.device, dtype=self.w0.dtype)
        with torch.no_grad():
            linear.weight.copy_(self.w0 + self.effective_delta())
        return linear
