from __future__ import annotations

import pytest

from lora_lab.config import load_config
from lora_lab.experiments.exp01_param_table import ParamTableExperiment
from lora_lab.metrics import reduction_ratio, trainable_ratio


@pytest.fixture(scope="module")
def records():
    cfg = load_config("configs/exp01_param_table.yaml")
    return ParamTableExperiment().run(cfg)


def test_full_ft_has_exactly_one_million_and_48k(records) -> None:
    full = next(record for record in records if record.method == "full_ft")
    assert full.trainable_params == 1_048_576
    assert full.total_params == 1_048_576
    assert full.trainable_ratio == 1.0


@pytest.mark.parametrize("rank,expected", [(4, 8_192), (8, 16_384), (16, 32_768)])
def test_lora_parameter_counts(records, rank: int, expected: int) -> None:
    lora = next(
        record
        for record in records
        if record.method == "lora" and record.method_hparams.get("r") == rank
    )
    assert lora.trainable_params == expected
    assert lora.total_params == 1_048_576
    assert lora.trainable_ratio == pytest.approx(expected / 1_048_576)


def test_reduction_factors(records) -> None:
    full = next(record for record in records if record.method == "full_ft")
    lora16 = next(
        record
        for record in records
        if record.method == "lora" and record.method_hparams.get("r") == 16
    )
    assert reduction_ratio(full.trainable_params, lora16.trainable_params) == 32.0
    assert trainable_ratio(full.trainable_params, lora16.trainable_params) == 1 / 32


def test_memory_components_sum(records) -> None:
    for record in records:
        extra = record.extra
        assert extra["memory_total_bytes"] == (
            extra["memory_weight_bytes"]
            + extra["memory_gradient_bytes"]
            + extra["memory_optimizer_bytes"]
        )
        assert extra["activations_included"] is False
