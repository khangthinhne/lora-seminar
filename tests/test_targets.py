from __future__ import annotations

import pytest
import torch

from lora_lab.config import DataConfig
from lora_lab.data import make_dataset, make_generator
from lora_lab.targets.decaying_spectrum import DecayingSpectrumTarget
from lora_lab.targets.low_rank import LowRankTarget


def test_low_rank_norm_and_rank() -> None:
    rng = make_generator(0, 1)
    delta = LowRankTarget().make_delta(
        d=20,
        k=15,
        rng=rng,
        true_rank=4,
        noise_std=0.0,
        delta_scale=0.5,
        reference_norm=10.0,
    )
    assert delta.shape == (20, 15)
    assert float(torch.linalg.matrix_norm(delta)) == pytest.approx(5.0, rel=1e-5)
    assert torch.linalg.matrix_rank(delta).item() == 4


def test_decaying_spectrum_is_full_rank_and_decreasing() -> None:
    rng = make_generator(1, 1)
    delta = DecayingSpectrumTarget().make_delta(
        d=12,
        k=8,
        rng=rng,
        gamma=1.0,
        delta_scale=0.5,
        reference_norm=4.0,
    )
    assert delta.shape == (12, 8)
    assert float(torch.linalg.matrix_norm(delta)) == pytest.approx(2.0, rel=1e-5)
    singular_values = torch.linalg.svdvals(delta)
    assert torch.all(singular_values[:-1] >= singular_values[1:] - 1e-6)


def test_dataset_target_norm_ratio() -> None:
    cfg = DataConfig(d=24, k=20, n_train=8, n_val=4, n_test=4, w0_std=0.05, obs_noise=0.0)
    dataset = make_dataset(
        cfg,
        "low_rank",
        {"true_rank": 3, "noise_std": 0.0, "delta_scale": 0.5},
        seed=2,
    )
    ratio = torch.linalg.matrix_norm(dataset.delta_star) / torch.linalg.matrix_norm(dataset.w0)
    assert float(ratio) == pytest.approx(0.5, rel=1e-5)


def test_unknown_target_raises() -> None:
    cfg = DataConfig(d=4, k=4, n_train=2, n_val=2, n_test=2, w0_std=0.1, obs_noise=0.0)
    with pytest.raises(ValueError, match="unknown target"):
        make_dataset(cfg, "does_not_exist", {}, seed=0)
