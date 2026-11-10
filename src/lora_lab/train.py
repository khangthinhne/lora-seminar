"""Generic training loop shared by every method.

There is deliberately no ``if method == ...`` branch: the loop only sees an
:class:`~lora_lab.methods.base.AdaptedLayer`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn

from lora_lab.config import TrainConfig
from lora_lab.data import STREAM_SHUFFLE, Dataset, make_generator
from lora_lab.metrics import mse


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one training run.

    Attributes:
        train_mse: Full training-set MSE after the last epoch.
        val_mse: Validation-set MSE after the last epoch.
        time_s: Wall-clock training time measured with ``time.perf_counter()``.
        lr: Learning rate used.
    """

    train_mse: float
    val_mse: float
    time_s: float
    lr: float


def train(
    layer: nn.Module,
    dataset: Dataset,
    cfg: TrainConfig,
    *,
    lr: float | None = None,
    seed: int = 0,
) -> TrainResult:
    """Train ``layer`` with Adam and MSE on the synthetic training split.

    Only parameters with ``requires_grad=True`` are passed to the optimizer.
    Batch order is controlled by a generator derived from ``(seed, STREAM_SHUFFLE)``
    so every method, rank, and learning-rate trial sees the same batches.
    """
    learning_rate = cfg.lr_grid[0] if lr is None else lr
    if learning_rate <= 0:
        raise ValueError(f"lr must be > 0, got {learning_rate}")

    parameters = [parameter for parameter in layer.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("layer has no trainable parameters")
    optimizer = torch.optim.Adam(
        parameters,
        lr=learning_rate,
        weight_decay=cfg.weight_decay,
    )
    shuffle_generator = make_generator(seed, STREAM_SHUFFLE)
    n_train = dataset.x_train.shape[0]

    started = time.perf_counter()
    layer.train()
    for _epoch in range(cfg.epochs):
        order = torch.randperm(n_train, generator=shuffle_generator)
        for start in range(0, n_train, cfg.batch_size):
            indices = order[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = layer(dataset.x_train[indices])
            loss = mse(prediction, dataset.y_train[indices])
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - started

    layer.eval()
    with torch.no_grad():
        train_mse = float(mse(layer(dataset.x_train), dataset.y_train))
        val_mse = float(mse(layer(dataset.x_val), dataset.y_val))
    return TrainResult(train_mse=train_mse, val_mse=val_mse, time_s=elapsed, lr=learning_rate)
