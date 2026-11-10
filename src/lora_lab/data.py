"""Synthetic data generation with independent, explicit random streams.

One seed controls ``W0``, the target ``delta_star``, the inputs ``X``, and the
observation noise. Each task uses a separate ``torch.Generator`` so that
changing one part of the setup does not perturb another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

import lora_lab.targets  # noqa: F401  (populates TARGETS)
from lora_lab.config import DataConfig
from lora_lab.registry import TARGETS

STREAM_W0 = 0
STREAM_TARGET = 1
STREAM_X = 2
STREAM_NOISE = 3
STREAM_SHUFFLE = 4
STREAM_INIT = 5

_MASK_63 = (1 << 63) - 1


def make_generator(seed: int, stream: int) -> torch.Generator:
    """Create a CPU generator from ``(seed, stream)`` using a stable mix.

    Args:
        seed: Non-negative experiment seed.
        stream: Small integer identifying the task (see ``STREAM_*`` constants).
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    mixed = (seed * 0x9E3779B97F4A7C15 + stream * 0xBF58476D1CE4E5B9) & _MASK_63
    generator = torch.Generator(device="cpu")
    generator.manual_seed(mixed)
    return generator


@dataclass(frozen=True)
class Dataset:
    """Synthetic regression data for one (target, seed) pair.

    Shapes: ``w0``, ``delta_star``, ``w_star`` are ``(d, k)``; ``x_*`` are
    ``(n, k)``; ``y_*`` are ``(n, d)``.
    """

    w0: Tensor
    delta_star: Tensor
    w_star: Tensor
    x_train: Tensor
    y_train: Tensor
    x_val: Tensor
    y_val: Tensor
    x_test: Tensor
    y_test: Tensor
    seed: int
    target_name: str
    target_cfg: dict[str, Any]


def make_dataset(
    cfg: DataConfig,
    target_name: str,
    target_cfg: dict[str, Any],
    seed: int,
) -> Dataset:
    """Generate ``W0``, ``delta_star``, ``W*``, and the train/val/test split.

    ``Y = X @ W*.T`` plus optional observation noise. The layer bias is zero.
    """
    if target_name not in TARGETS:
        raise ValueError(f"unknown target {target_name!r}; available: {sorted(TARGETS)}")
    d, k = cfg.d, cfg.k

    w0_generator = make_generator(seed, STREAM_W0)
    w0 = torch.randn(d, k, generator=w0_generator) * cfg.w0_std

    target_generator = make_generator(seed, STREAM_TARGET)
    delta_star = TARGETS[target_name]().make_delta(
        d,
        k,
        target_generator,
        reference_norm=float(torch.linalg.matrix_norm(w0)),
        **dict(target_cfg),
    )
    w_star = w0 + delta_star

    n_total = cfg.n_train + cfg.n_val + cfg.n_test
    x_generator = make_generator(seed, STREAM_X)
    x_all = torch.randn(n_total, k, generator=x_generator)
    y_all = x_all @ w_star.T
    if cfg.obs_noise > 0:
        noise_generator = make_generator(seed, STREAM_NOISE)
        y_all = y_all + cfg.obs_noise * torch.randn(n_total, d, generator=noise_generator)

    train_end = cfg.n_train
    val_end = cfg.n_train + cfg.n_val
    return Dataset(
        w0=w0,
        delta_star=delta_star,
        w_star=w_star,
        x_train=x_all[:train_end],
        y_train=y_all[:train_end],
        x_val=x_all[train_end:val_end],
        y_val=y_all[train_end:val_end],
        x_test=x_all[val_end:],
        y_test=y_all[val_end:],
        seed=seed,
        target_name=target_name,
        target_cfg=dict(target_cfg),
    )
