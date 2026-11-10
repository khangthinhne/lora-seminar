"""YAML configuration loading, merging, and validation.

Every experiment config is merged on top of ``configs/base.yaml`` (or a file
named by the optional ``extends`` key). Validation happens once, at load time,
so the rest of the code receives typed objects instead of raw dictionaries.

Typical usage::

    cfg = load_config("configs/exp02_rank_sweep.yaml")
    cfg.data.d, cfg.train.lr_grid, cfg.methods[0].name
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a YAML configuration is missing or contains invalid values."""


@dataclass(frozen=True)
class DataConfig:
    """Synthetic data settings.

    Attributes:
        d: Output dimension of the adapted layer (rows of ``W0``).
        k: Input dimension of the adapted layer (columns of ``W0``).
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        w0_std: Standard deviation of ``W0 ~ N(0, w0_std**2)``.
        obs_noise: Standard deviation of additive observation noise on ``Y``.
    """

    d: int
    k: int
    n_train: int
    n_val: int
    n_test: int
    w0_std: float
    obs_noise: float


@dataclass(frozen=True)
class TrainConfig:
    """Training settings shared by all methods.

    ``lr_grid`` is used to select a learning rate on the validation split; the
    selected value is then evaluated on the test split.
    """

    epochs: int
    batch_size: int
    optimizer: str
    lr_grid: tuple[float, ...]
    weight_decay: float


@dataclass(frozen=True)
class ComponentConfig:
    """A registered component (target or quantizer) plus its free-form config."""

    name: str
    cfg: Mapping[str, Any]


@dataclass(frozen=True)
class MethodSpec:
    """A method name and the Cartesian grid of hyperparameters to sweep."""

    name: str
    grid: Mapping[str, tuple[Any, ...]]

    def expand(self) -> list[dict[str, Any]]:
        """Expand the grid into concrete hyperparameter dictionaries.

        An empty grid yields a single empty dictionary, i.e. one run.
        """
        if not self.grid:
            return [{}]
        keys = list(self.grid)
        value_columns = (self.grid[key] for key in keys)
        return [dict(zip(keys, values, strict=True)) for values in product(*value_columns)]


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully validated experiment configuration."""

    name: str
    description: str
    data: DataConfig
    train: TrainConfig
    seeds: tuple[int, ...]
    targets: tuple[ComponentConfig, ...]
    quantizer: ComponentConfig
    methods: tuple[MethodSpec, ...]
    include_no_finetune: bool
    extra: Mapping[str, Any]
    source: str
    resolved: Mapping[str, Any]

    def get_extra(self, key: str, default: Any = None) -> Any:
        """Return an experiment-specific extra value or ``default``."""
        return self.extra.get(key, default)

    def require_extra(self, key: str, expected_type: type | tuple[type, ...]) -> Any:
        """Return an extra value and validate its type with a clear error."""
        if key not in self.extra:
            raise ConfigError(f"{self.source}: extra.{key} is required but missing")
        value = self.extra[key]
        if not isinstance(value, expected_type):
            raise ConfigError(
                f"{self.source}: extra.{key} must be {expected_type}, got {type(value).__name__}"
            )
        return value


_TOP_LEVEL_KEYS = {
    "name",
    "description",
    "data",
    "train",
    "seeds",
    "target",
    "targets",
    "quantizer",
    "methods",
    "include_no_finetune",
    "extra",
}

_DATA_KEYS = {"d", "k", "n_train", "n_val", "n_test", "w0_std", "obs_noise"}
_TRAIN_KEYS = {"epochs", "batch_size", "optimizer", "lr_grid", "weight_decay"}
_COMPONENT_KEYS = {"name", "cfg"}
_METHOD_KEYS = {"name", "grid"}


def load_config(path: str | Path) -> ExperimentConfig:
    """Load, merge with the base config, validate, and return an ``ExperimentConfig``."""
    resolved = load_resolved_dict(path)
    return parse_config(resolved, source=str(path))


def load_resolved_dict(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and deep-merge it on top of its base config.

    The base config is ``extends`` when present, otherwise ``base.yaml`` in the
    same directory (if that file exists and is not the file being loaded).
    """
    config_path = Path(path)
    experiment_raw = _load_yaml_mapping(config_path)
    if "extends" in experiment_raw:
        base_path = Path(experiment_raw["extends"])
        if not base_path.is_absolute():
            base_path = (config_path.parent / base_path).resolve()
        base_raw = _load_yaml_mapping(base_path)
        experiment_raw = {k: v for k, v in experiment_raw.items() if k != "extends"}
    else:
        candidate = config_path.parent / "base.yaml"
        base_raw = _load_yaml_mapping(candidate) if candidate.exists() else {}
        if candidate.resolve() == config_path.resolve():
            base_raw = {}

    if "targets" in experiment_raw:
        base_raw.pop("target", None)
    if "target" in experiment_raw:
        base_raw.pop("targets", None)

    merged = _deep_merge(base_raw, experiment_raw)
    return merged


def parse_component_config(value: Any, path: str = "component") -> ComponentConfig:
    """Validate one ``{name, cfg}`` mapping (used by experiments with custom lists)."""
    return _parse_component(value, path)


def parse_config(raw: Mapping[str, Any], source: str = "<config>") -> ExperimentConfig:
    """Validate a resolved config mapping and convert it to dataclasses."""
    mapping = _as_mapping(raw, source)
    _check_allowed(mapping, _TOP_LEVEL_KEYS, source)

    name = _get_str(mapping, "name", source)
    description = _get_optional_str(mapping, "description", "")

    data = _parse_data(_as_mapping(_get_required(mapping, "data", source), f"{source}: data"))
    train = _parse_train(_as_mapping(_get_required(mapping, "train", source), f"{source}: train"))
    seeds = _parse_seeds(_get_required(mapping, "seeds", source), source)

    if "targets" in mapping:
        target_values = _get_required(mapping, "targets", source)
        if not isinstance(target_values, Sequence) or isinstance(target_values, (str, bytes)):
            raise ConfigError(f"{source}: targets must be a non-empty list")
        if not target_values:
            raise ConfigError(f"{source}: targets must be a non-empty list")
        targets = tuple(
            _parse_component(value, f"{source}: targets[{index}]")
            for index, value in enumerate(target_values)
        )
    elif "target" in mapping:
        targets = (_parse_component(_get_required(mapping, "target", source), f"{source}: target"),)
    else:
        raise ConfigError(f"{source}: either target or targets is required")

    quantizer = _parse_component(
        _get_required(mapping, "quantizer", source), f"{source}: quantizer"
    )
    methods = _parse_methods(_get_required(mapping, "methods", source), source)
    include_no_finetune = _get_optional_bool(mapping, "include_no_finetune", False, source)
    extra = _as_mapping(mapping.get("extra", {}), f"{source}: extra")

    return ExperimentConfig(
        name=name,
        description=description,
        data=data,
        train=train,
        seeds=seeds,
        targets=targets,
        quantizer=quantizer,
        methods=methods,
        include_no_finetune=include_no_finetune,
        extra=copy.deepcopy(dict(extra)),
        source=source,
        resolved=copy.deepcopy(dict(mapping)),
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:  # pragma: no cover - depends on broken YAML
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"config file is empty: {path}")
    return dict(_as_mapping(raw, str(path)))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_data(mapping: Mapping[str, Any]) -> DataConfig:
    _check_allowed(mapping, _DATA_KEYS, "data")
    return DataConfig(
        d=_get_int(mapping, "d", "data", minimum=1),
        k=_get_int(mapping, "k", "data", minimum=1),
        n_train=_get_int(mapping, "n_train", "data", minimum=1),
        n_val=_get_int(mapping, "n_val", "data", minimum=1),
        n_test=_get_int(mapping, "n_test", "data", minimum=1),
        w0_std=_get_float(mapping, "w0_std", "data", minimum=0.0, strictly_positive=True),
        obs_noise=_get_float(mapping, "obs_noise", "data", minimum=0.0),
    )


def _parse_train(mapping: Mapping[str, Any]) -> TrainConfig:
    _check_allowed(mapping, _TRAIN_KEYS, "train")
    optimizer = _get_str(mapping, "optimizer", "train").lower()
    if optimizer != "adam":
        raise ConfigError(f"train.optimizer: only 'adam' is supported, got {optimizer!r}")
    raw_grid = _get_required(mapping, "lr_grid", "train")
    if not isinstance(raw_grid, Sequence) or isinstance(raw_grid, (str, bytes)) or not raw_grid:
        raise ConfigError("train.lr_grid must be a non-empty list of positive numbers")
    lr_grid = tuple(
        _check_number(value, f"train.lr_grid[{index}]", minimum=0.0, strictly_positive=True)
        for index, value in enumerate(raw_grid)
    )
    return TrainConfig(
        epochs=_get_int(mapping, "epochs", "train", minimum=1),
        batch_size=_get_int(mapping, "batch_size", "train", minimum=1),
        optimizer=optimizer,
        lr_grid=lr_grid,
        weight_decay=_get_float(mapping, "weight_decay", "train", minimum=0.0),
    )


def _parse_seeds(value: Any, source: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ConfigError(f"{source}: seeds must be a non-empty list of non-negative integers")
    seeds: list[int] = []
    for index, item in enumerate(value):
        seeds.append(_check_int(item, f"{source}: seeds[{index}]", minimum=0))
    if len(set(seeds)) != len(seeds):
        raise ConfigError(f"{source}: seeds must be unique")
    return tuple(seeds)


def _parse_component(value: Any, path: str) -> ComponentConfig:
    mapping = _as_mapping(value, path)
    _check_allowed(mapping, _COMPONENT_KEYS, path)
    name = _get_str(mapping, "name", path)
    cfg = _as_mapping(mapping.get("cfg", {}), f"{path}.cfg")
    return ComponentConfig(name=name, cfg=copy.deepcopy(dict(cfg)))


def _parse_methods(value: Any, source: str) -> tuple[MethodSpec, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ConfigError(f"{source}: methods must be a non-empty list")
    methods: list[MethodSpec] = []
    for index, item in enumerate(value):
        path = f"{source}: methods[{index}]"
        mapping = _as_mapping(item, path)
        _check_allowed(mapping, _METHOD_KEYS, path)
        name = _get_str(mapping, "name", path)
        grid_raw = _as_mapping(mapping.get("grid", {}), f"{path}.grid")
        grid: dict[str, tuple[Any, ...]] = {}
        for key, values in grid_raw.items():
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
                raise ConfigError(f"{path}.grid.{key} must be a non-empty list of values")
            grid[str(key)] = tuple(copy.deepcopy(value) for value in values)
        methods.append(MethodSpec(name=name, grid=grid))
    return tuple(methods)


def _check_allowed(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}; allowed keys are {sorted(allowed)}")


def _as_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping, got {type(value).__name__}")
    return value


def _get_required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{path}: missing required key {key!r}")
    return mapping[key]


def _get_str(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = _get_required(mapping, key, path)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}.{key} must be a non-empty string")
    return value


def _get_optional_str(mapping: Mapping[str, Any], key: str, default: str) -> str:
    if key not in mapping:
        return default
    value = mapping[key]
    if not isinstance(value, str):
        raise ConfigError("description must be a string")
    return value


def _get_bool(mapping: Mapping[str, Any], key: str, path: str) -> bool:
    value = _get_required(mapping, key, path)
    if not isinstance(value, bool):
        raise ConfigError(f"{path}.{key} must be a boolean")
    return value


def _get_optional_bool(mapping: Mapping[str, Any], key: str, default: bool, path: str) -> bool:
    if key not in mapping:
        return default
    value = mapping[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{path}.{key} must be a boolean")
    return value


def _get_int(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    *,
    minimum: int | None = None,
) -> int:
    return _check_int(_get_required(mapping, key, path), f"{path}.{key}", minimum=minimum)


def _check_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be >= {minimum}, got {value}")
    return value


def _get_float(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    return _check_number(
        _get_required(mapping, key, path),
        f"{path}.{key}",
        minimum=minimum,
        strictly_positive=strictly_positive,
    )


def _check_number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number, got {type(value).__name__}")
    number = float(value)
    if strictly_positive and number <= 0:
        raise ConfigError(f"{path} must be > 0, got {number}")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{path} must be >= {minimum}, got {number}")
    return number
