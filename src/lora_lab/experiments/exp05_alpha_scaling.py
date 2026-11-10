"""exp05 - optional alpha/r scaling experiment.

Hypothesis: alpha and the learning rate interact, so comparing scaling policies
with a fixed untuned learning rate is misleading. Setup: LoRA ranks {4, 8} and
alpha values {4, 8, 16}, with and without validation-based learning-rate
selection. Expected outcome: the tuned comparison is the fair one; the untuned
comparison can reverse the ranking. Results are reported in this toy setup only.
"""

from __future__ import annotations

from typing import Any

from lora_lab.config import ExperimentConfig
from lora_lab.experiments.base import Experiment
from lora_lab.experiments.common import (
    TrainedRun,
    build_layer,
    make_dataset,
    make_run_record,
    prepare_base,
    train_and_select,
    with_base,
)
from lora_lab.records import RunRecord
from lora_lab.registry import EXPERIMENTS, register
from lora_lab.train import train


@register(EXPERIMENTS, "exp05_alpha_scaling")
class AlphaScalingExperiment(Experiment):
    """Compare alpha policies with tuned and untuned learning rates."""

    name = "exp05_alpha_scaling"

    def run(self, cfg: ExperimentConfig) -> list[RunRecord]:
        modes = cfg.require_extra("lr_selection_modes", (list, tuple))
        for mode in modes:
            if mode not in {"tuned", "untuned"}:
                raise ValueError(f"unknown lr selection mode {mode!r}; expected tuned/untuned")
        untuned_lr = float(cfg.require_extra("untuned_lr", (int, float)))

        records: list[RunRecord] = []
        for target in cfg.targets:
            for seed in cfg.seeds:
                dataset = make_dataset(cfg, target, seed)
                quantized_base = prepare_base(dataset.w0, cfg.quantizer)
                train_dataset = with_base(dataset, quantized_base.w0)
                for method_spec in cfg.methods:
                    for hparams in method_spec.expand():
                        r = int(hparams["r"])
                        alpha = float(hparams["alpha"])
                        scheme = alpha_scheme(r, alpha)
                        for mode in modes:
                            trained = self._fit(
                                method_spec.name,
                                hparams,
                                train_dataset,
                                cfg,
                                seed,
                                mode,
                                untuned_lr,
                            )
                            record_hparams = dict(hparams)
                            record_hparams["alpha_scheme"] = scheme
                            record_hparams["lr_selection"] = mode
                            records.append(
                                make_run_record(
                                    experiment=self.name,
                                    cfg=cfg,
                                    target=target,
                                    method_name=method_spec.name,
                                    hparams=record_hparams,
                                    seed=seed,
                                    dataset=dataset,
                                    layer=trained.layer,
                                    lr=trained.lr,
                                    result=trained.result,
                                    quantized_base=quantized_base,
                                    lr_trials=trained.lr_trials,
                                    extra={
                                        "alpha_scheme": scheme,
                                        "lr_selection_mode": mode,
                                        "untuned_lr": untuned_lr,
                                    },
                                )
                            )
        return records

    @staticmethod
    def _fit(
        method_name: str,
        hparams: dict[str, Any],
        dataset: Any,
        cfg: ExperimentConfig,
        seed: int,
        mode: str,
        untuned_lr: float,
    ) -> TrainedRun:
        if mode == "tuned":
            return train_and_select(method_name, hparams, dataset, cfg, seed)
        layer = build_layer(method_name, dataset.w0, hparams, seed)
        result = train(layer, dataset, cfg.train, lr=untuned_lr, seed=seed)
        trials = [
            {
                "lr": untuned_lr,
                "val_mse": result.val_mse,
                "train_time_s": result.time_s,
            }
        ]
        return TrainedRun(layer=layer, lr=untuned_lr, result=result, lr_trials=trials)


def alpha_scheme(r: int, alpha: float) -> str:
    """Classify an ``(r, alpha)`` pair under the three compared policies."""
    tags: list[str] = []
    if alpha == 8.0:
        tags.append("fixed_alpha_8")
    if alpha == float(r):
        tags.append("alpha_eq_r")
    if alpha == 2.0 * r:
        tags.append("alpha_eq_2r")
    return "+".join(tags) if tags else "other"
