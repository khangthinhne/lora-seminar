"""Shared experiment helpers.

Experiments use the registries by name; they never import concrete method,
target, or quantizer classes. This keeps new plugins compatible with every
existing experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

import lora_lab.methods  # noqa: F401  (populates METHODS)
import lora_lab.quant  # noqa: F401  (populates QUANTIZERS)
import lora_lab.targets  # noqa: F401  (populates TARGETS)
from lora_lab.config import ComponentConfig, ExperimentConfig
from lora_lab.data import STREAM_INIT, Dataset, make_generator
from lora_lab.metrics import mse, relative_delta_error, trainable_ratio
from lora_lab.records import RunRecord
from lora_lab.registry import METHODS, QUANTIZERS
from lora_lab.train import TrainResult, train


@dataclass(frozen=True)
class QuantizedBase:
    """Base weight after optional quantization, plus storage metadata."""

    w0: Tensor
    quantizer_name: str
    quantizer_cfg: dict[str, Any]
    quant_error: float
    storage_bytes: int


@dataclass(frozen=True)
class TrainedRun:
    """The validation-selected layer and its learning-rate trials."""

    layer: Any
    lr: float
    result: TrainResult
    lr_trials: list[dict[str, float]]


def prepare_base(w0: Tensor, quantizer: ComponentConfig) -> QuantizedBase:
    """Quantize ``w0`` (if requested), dequantize it, and measure storage/error."""
    if quantizer.name not in QUANTIZERS:
        raise ValueError(f"unknown quantizer {quantizer.name!r}; available: {sorted(QUANTIZERS)}")
    instance = QUANTIZERS[quantizer.name](**dict(quantizer.cfg))
    quantized = instance.quantize(w0)
    restored = instance.dequantize(quantized)
    reference = float(torch.linalg.matrix_norm(w0))
    error = 0.0 if reference == 0 else float(torch.linalg.matrix_norm(restored - w0) / reference)
    return QuantizedBase(
        w0=restored,
        quantizer_name=quantizer.name,
        quantizer_cfg=dict(quantizer.cfg),
        quant_error=error,
        storage_bytes=instance.storage_bytes(quantized),
    )


def build_layer(
    method_name: str,
    w0: Tensor,
    hparams: dict[str, Any],
    seed: int,
) -> Any:
    """Instantiate a registered method with a deterministic A/init stream."""
    if method_name not in METHODS:
        raise ValueError(f"unknown method {method_name!r}; available: {sorted(METHODS)}")
    generator = make_generator(seed, STREAM_INIT)
    return METHODS[method_name].from_base(w0, init_generator=generator, **dict(hparams))


def train_and_select(
    method_name: str,
    hparams: dict[str, Any],
    dataset: Dataset,
    cfg: ExperimentConfig,
    seed: int,
    lr_grid: tuple[float, ...] | None = None,
) -> TrainedRun:
    """Select a learning rate on the validation split (never on test)."""
    grid = cfg.train.lr_grid if lr_grid is None else lr_grid
    if not grid:
        raise ValueError("learning-rate grid must not be empty")

    best_layer = None
    best_lr = float(grid[0])
    best_result = None
    trials: list[dict[str, float]] = []
    for lr in grid:
        layer = build_layer(method_name, dataset.w0, hparams, seed)
        result = train(layer, dataset, cfg.train, lr=float(lr), seed=seed)
        trials.append({"lr": float(lr), "val_mse": result.val_mse, "train_time_s": result.time_s})
        if best_result is None or result.val_mse < best_result.val_mse:
            best_layer, best_lr, best_result = layer, float(lr), result
    if best_layer is None or best_result is None:
        raise RuntimeError("learning-rate selection failed")
    return TrainedRun(layer=best_layer, lr=best_lr, result=best_result, lr_trials=trials)


def make_run_record(
    *,
    experiment: str,
    cfg: ExperimentConfig,
    target: ComponentConfig,
    method_name: str,
    hparams: dict[str, Any],
    seed: int,
    dataset: Dataset,
    layer: Any,
    lr: float | None,
    result: TrainResult,
    quantized_base: QuantizedBase,
    lr_trials: list[dict[str, float]] | None = None,
    extra: dict[str, Any] | None = None,
) -> RunRecord:
    """Evaluate one trained layer and package it as a generic run record."""
    d, k = cfg.data.d, cfg.data.k
    trainable = layer.trainable_param_count()
    delta = layer.effective_delta().detach()
    delta_error = relative_delta_error(delta, dataset.delta_star)
    effective = quantized_base.w0 + delta
    delta_star_norm = float(torch.linalg.matrix_norm(dataset.delta_star))
    effective_error = float(torch.linalg.matrix_norm(effective - dataset.w_star) / delta_star_norm)

    layer.eval()
    with torch.no_grad():
        test_mse = float(mse(layer(dataset.x_test), dataset.y_test))

    record_extra: dict[str, Any] = {
        "lr_trials": lr_trials or [],
        "lr_selection_total_s": sum(trial["train_time_s"] for trial in (lr_trials or [])),
        "quantizer_cfg": dict(quantized_base.quantizer_cfg),
        "quant_error": quantized_base.quant_error,
        "weight_storage_bytes": quantized_base.storage_bytes,
        "effective_weight_rel_err": effective_error,
        "activations_included": False,
        "timing_note": "CPU toy timing; not representative of LLM training.",
    }
    if extra:
        record_extra.update(extra)

    return RunRecord(
        experiment=experiment,
        method=method_name,
        method_hparams=dict(hparams),
        target=target.name,
        target_cfg=dict(target.cfg),
        quantizer=quantized_base.quantizer_name,
        d=d,
        k=k,
        seed=seed,
        lr=lr,
        epochs=cfg.train.epochs,
        trainable_params=trainable,
        total_params=d * k,
        trainable_ratio=trainable_ratio(d * k, trainable),
        train_mse=result.train_mse,
        val_mse=result.val_mse,
        test_mse=test_mse,
        rel_delta_err=delta_error,
        train_time_s=result.time_s,
        extra=record_extra,
    )


def make_no_finetune_record(
    *,
    experiment: str,
    cfg: ExperimentConfig,
    target: ComponentConfig,
    seed: int,
    dataset: Dataset,
    quantized_base: QuantizedBase,
) -> RunRecord:
    """Record the ``delta = 0`` baseline (no optimization at all)."""
    layer = build_layer("full_ft", quantized_base.w0, {}, seed)
    layer.eval()
    with torch.no_grad():
        train_mse = float(mse(layer(dataset.x_train), dataset.y_train))
        val_mse = float(mse(layer(dataset.x_val), dataset.y_val))
    result = TrainResult(train_mse=train_mse, val_mse=val_mse, time_s=0.0, lr=0.0)
    return make_run_record(
        experiment=experiment,
        cfg=cfg,
        target=target,
        method_name="full_ft",
        hparams={"trained": False},
        seed=seed,
        dataset=dataset,
        layer=layer,
        lr=None,
        result=result,
        quantized_base=quantized_base,
        extra={"no_finetune": True},
    )


def run_rank_sweep(cfg: ExperimentConfig, experiment: str) -> list[RunRecord]:
    """Shared implementation of exp02 and exp03 (rank sweep across targets)."""
    return run_sweep(cfg, experiment, [cfg.quantizer])


def run_sweep(
    cfg: ExperimentConfig,
    experiment: str,
    quantizers: list[ComponentConfig],
) -> list[RunRecord]:
    """Run every method/target/seed for each supplied quantizer."""
    records: list[RunRecord] = []
    for quantizer in quantizers:
        for target in cfg.targets:
            for seed in cfg.seeds:
                dataset = make_dataset(cfg, target, seed)
                quantized_base = prepare_base(dataset.w0, quantizer)
                if cfg.include_no_finetune:
                    records.append(
                        make_no_finetune_record(
                            experiment=experiment,
                            cfg=cfg,
                            target=target,
                            seed=seed,
                            dataset=dataset,
                            quantized_base=quantized_base,
                        )
                    )
                for method_spec in cfg.methods:
                    for hparams in method_spec.expand():
                        trained = train_and_select(
                            method_spec.name,
                            hparams,
                            with_base(dataset, quantized_base.w0),
                            cfg,
                            seed,
                        )
                        records.append(
                            make_run_record(
                                experiment=experiment,
                                cfg=cfg,
                                target=target,
                                method_name=method_spec.name,
                                hparams=hparams,
                                seed=seed,
                                dataset=dataset,
                                layer=trained.layer,
                                lr=trained.lr,
                                result=trained.result,
                                quantized_base=quantized_base,
                                lr_trials=trained.lr_trials,
                                extra=_quantization_extra(quantized_base),
                            )
                        )
    return records


def _quantization_extra(quantized_base: QuantizedBase) -> dict[str, Any] | None:
    if quantized_base.quantizer_name == "none":
        return None
    return {
        "quantization_note": "Simple absmax block quantization; this is NOT NF4.",
        "simulated_qlora": True,
    }


def make_dataset(cfg: ExperimentConfig, target: ComponentConfig, seed: int) -> Dataset:
    """Build a dataset from an experiment config (thin wrapper for readability)."""
    from lora_lab.data import make_dataset as _make_dataset

    return _make_dataset(cfg.data, target.name, dict(target.cfg), seed)


def with_base(dataset: Dataset, w0: Tensor) -> Dataset:
    """Return a dataset copy whose ``w0`` is the (possibly dequantized) base."""
    return Dataset(
        w0=w0,
        delta_star=dataset.delta_star,
        w_star=dataset.w_star,
        x_train=dataset.x_train,
        y_train=dataset.y_train,
        x_val=dataset.x_val,
        y_val=dataset.y_val,
        x_test=dataset.x_test,
        y_test=dataset.y_test,
        seed=dataset.seed,
        target_name=dataset.target_name,
        target_cfg=dataset.target_cfg,
    )
