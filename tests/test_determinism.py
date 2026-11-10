from __future__ import annotations

import torch

from lora_lab.config import DataConfig, TrainConfig
from lora_lab.data import make_dataset
from lora_lab.experiments.common import build_layer
from lora_lab.train import train

DATA = DataConfig(
    d=16,
    k=12,
    n_train=64,
    n_val=16,
    n_test=16,
    w0_std=0.1,
    obs_noise=0.0,
)
TARGET_CFG = {"true_rank": 4, "noise_std": 0.0, "delta_scale": 0.5}
TRAIN = TrainConfig(
    epochs=4,
    batch_size=16,
    optimizer="adam",
    lr_grid=(0.03,),
    weight_decay=0.0,
)


def test_same_seed_same_dataset() -> None:
    first = make_dataset(DATA, "low_rank", TARGET_CFG, seed=3)
    second = make_dataset(DATA, "low_rank", TARGET_CFG, seed=3)
    for field in ("w0", "delta_star", "w_star", "x_train", "y_train", "x_test", "y_test"):
        assert torch.equal(getattr(first, field), getattr(second, field))


def test_random_streams_are_independent() -> None:
    base = make_dataset(DATA, "low_rank", TARGET_CFG, seed=5)
    changed_target = make_dataset(
        DATA,
        "low_rank",
        {"true_rank": 2, "noise_std": 0.2, "delta_scale": 0.5},
        seed=5,
    )
    assert torch.equal(base.w0, changed_target.w0)
    assert torch.equal(base.x_train, changed_target.x_train)
    assert not torch.equal(base.delta_star, changed_target.delta_star)


def test_training_is_deterministic() -> None:
    dataset = make_dataset(DATA, "low_rank", TARGET_CFG, seed=9)
    first_layer = build_layer("lora", dataset.w0, {"r": 2, "alpha": 4, "init_std": 0.01}, 9)
    second_layer = build_layer("lora", dataset.w0, {"r": 2, "alpha": 4, "init_std": 0.01}, 9)
    first = train(first_layer, dataset, TRAIN, lr=0.03, seed=9)
    second = train(second_layer, dataset, TRAIN, lr=0.03, seed=9)
    assert first.train_mse == second.train_mse
    assert first.val_mse == second.val_mse
    torch.testing.assert_close(
        first_layer.effective_delta(),
        second_layer.effective_delta(),
        rtol=0,
        atol=1e-7,
    )
