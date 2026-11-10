from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace

import pytest

from lora_lab.config import ConfigError, load_config, load_resolved_dict, parse_config


def test_load_base_and_experiment_merge() -> None:
    cfg = load_config("configs/exp02_rank_sweep.yaml")
    assert cfg.name == "exp02_rank_sweep"
    assert cfg.data.d == 256 and cfg.data.k == 256
    assert cfg.seeds == (0, 1, 2, 3, 4)
    assert cfg.targets[0].name == "low_rank"
    assert cfg.targets[0].cfg["true_rank"] == 8
    assert cfg.quantizer.name == "none"
    assert [method.name for method in cfg.methods] == ["full_ft", "lora"]
    assert cfg.include_no_finetune is True


def test_targets_list_replaces_single_target() -> None:
    cfg = load_config("configs/exp03_fullrank_target.yaml")
    assert len(cfg.targets) == 3
    assert {target.cfg["gamma"] for target in cfg.targets} == {0.5, 1.0, 2.0}


def test_grid_expansion_is_deterministic() -> None:
    cfg = load_config("configs/exp02_rank_sweep.yaml")
    lora_spec = next(method for method in cfg.methods if method.name == "lora")
    combos = lora_spec.expand()
    assert len(combos) == 5
    assert combos[0] == {"r": 1, "alpha": 16, "init_std": 0.01}
    assert combos[-1] == {"r": 16, "alpha": 16, "init_std": 0.01}


def test_empty_grid_yields_single_run() -> None:
    cfg = load_config("configs/exp02_rank_sweep.yaml")
    full_spec = next(method for method in cfg.methods if method.name == "full_ft")
    assert full_spec.expand() == [{}]


def test_unknown_key_is_reported() -> None:
    raw = load_resolved_dict("configs/exp02_rank_sweep.yaml")
    broken = copy.deepcopy(raw)
    broken["unexpected"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(broken, source="test")


def test_invalid_value_identifies_key() -> None:
    raw = load_resolved_dict("configs/exp02_rank_sweep.yaml")
    broken = copy.deepcopy(raw)
    broken["data"]["d"] = 0
    with pytest.raises(ConfigError, match=r"data\.d"):
        parse_config(broken, source="test")


def test_duplicate_seeds_rejected() -> None:
    raw = load_resolved_dict("configs/exp02_rank_sweep.yaml")
    broken = copy.deepcopy(raw)
    broken["seeds"] = [1, 1]
    with pytest.raises(ConfigError, match="unique"):
        parse_config(broken, source="test")


def test_frozen_config_cannot_be_mutated() -> None:
    cfg = load_config("configs/exp01_param_table.yaml")
    with pytest.raises(FrozenInstanceError):
        cfg.name = "other"  # type: ignore[misc]
    assert replace(cfg, name="other").name == "other"
