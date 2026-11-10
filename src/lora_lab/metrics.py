"""Pure metric functions. Metrics do not know which method produced a tensor."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def mse(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error over all elements of two identically shaped tensors."""
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )
    return torch.mean((pred - target) ** 2)


def relative_delta_error(delta_hat: Tensor, delta_star: Tensor) -> float:
    """Return ``||delta_hat - delta_star||_F / ||delta_star||_F``.

    Both tensors must have shape ``(d, k)``. The denominator must be nonzero.
    """
    if delta_hat.shape != delta_star.shape:
        raise ValueError(
            f"shape mismatch: delta_hat {tuple(delta_hat.shape)} vs "
            f"delta_star {tuple(delta_star.shape)}"
        )
    denominator = torch.linalg.matrix_norm(delta_star)
    if float(denominator) == 0:
        raise ValueError("delta_star has zero Frobenius norm; relative error is undefined")
    return float(torch.linalg.matrix_norm(delta_hat - delta_star) / denominator)


def trainable_params(module: nn.Module) -> int:
    """Number of parameters with ``requires_grad=True``."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def total_params(module: nn.Module) -> int:
    """Number of parameters in ``module`` (buffers are not parameters)."""
    return sum(parameter.numel() for parameter in module.parameters())


def trainable_ratio(full_params: int, trainable: int) -> float:
    """Return ``trainable / full_params`` (fraction of a full fine-tune)."""
    if full_params <= 0:
        raise ValueError(f"full_params must be > 0, got {full_params}")
    if trainable < 0:
        raise ValueError(f"trainable must be >= 0, got {trainable}")
    return trainable / full_params


def reduction_ratio(full_params: int, peft_params: int) -> float:
    """Return ``full_params / peft_params`` (how many times smaller)."""
    if peft_params <= 0:
        raise ValueError(f"peft_params must be > 0, got {peft_params}")
    return full_params / peft_params
