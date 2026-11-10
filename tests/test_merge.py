from __future__ import annotations

import pytest
import torch

from lora_lab.config import DataConfig, TrainConfig
from lora_lab.data import make_dataset
from lora_lab.experiments.common import build_layer
from lora_lab.train import train

DATA = DataConfig(d=12, k=9, n_train=32, n_val=8, n_test=8, w0_std=0.1, obs_noise=0.0)
TARGET_CFG = {"true_rank": 3, "noise_std": 0.0, "delta_scale": 0.5}
TRAIN = TrainConfig(epochs=2, batch_size=8, optimizer="adam", lr_grid=(0.03,), weight_decay=0.0)


@pytest.mark.parametrize(
    "method_name,hparams",
    [
        ("full_ft", {}),
        ("lora", {"r": 1, "alpha": 2, "init_std": 0.01}),
        ("lora", {"r": 4, "alpha": 8, "init_std": 0.01}),
    ],
)
def test_merge_matches_forward_after_training(method_name, hparams) -> None:
    dataset = make_dataset(DATA, "low_rank", TARGET_CFG, seed=11)
    layer = build_layer(method_name, dataset.w0, hparams, 11)
    train(layer, dataset, TRAIN, lr=0.03, seed=11)
    x = dataset.x_test
    layer.eval()
    with torch.no_grad():
        error = float((layer(x) - layer.merge()(x)).abs().max())
    assert error < 1e-5
