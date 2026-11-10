"""exp04 - numerical merge check.

Hypothesis: the rank-r forward path and the merged dense weight ``W0 + BA``
produce the same output in float32. Setup: train each method exactly as in
exp02, then compare ``layer(x)`` with ``layer.merge()(x)`` on a test batch.
Expected outcome: maximum absolute error around 1e-6, well below the configured
threshold. The experiment fails loudly if the threshold is exceeded.
"""

from __future__ import annotations

import torch

from lora_lab.config import ExperimentConfig
from lora_lab.experiments.base import Experiment
from lora_lab.experiments.common import (
    make_dataset,
    make_run_record,
    prepare_base,
    train_and_select,
    with_base,
)
from lora_lab.records import RunRecord
from lora_lab.registry import EXPERIMENTS, register


@register(EXPERIMENTS, "exp04_merge_check")
class MergeCheckExperiment(Experiment):
    """Train each method and verify that merging preserves its outputs."""

    name = "exp04_merge_check"

    def run(self, cfg: ExperimentConfig) -> list[RunRecord]:
        threshold = float(cfg.require_extra("max_abs_err_threshold", (int, float)))
        eval_batch = int(cfg.get_extra("eval_batch_size", cfg.data.n_test))
        records: list[RunRecord] = []
        for target in cfg.targets:
            for seed in cfg.seeds:
                dataset = make_dataset(cfg, target, seed)
                quantized_base = prepare_base(dataset.w0, cfg.quantizer)
                train_dataset = with_base(dataset, quantized_base.w0)
                for method_spec in cfg.methods:
                    for hparams in method_spec.expand():
                        trained = train_and_select(
                            method_spec.name, hparams, train_dataset, cfg, seed
                        )
                        x_eval = dataset.x_test[:eval_batch]
                        trained.layer.eval()
                        with torch.no_grad():
                            direct = trained.layer(x_eval)
                            merged = trained.layer.merge()(x_eval)
                        max_abs_err = float((direct - merged).abs().max())
                        if max_abs_err > threshold:
                            raise RuntimeError(
                                f"merge check failed for {method_spec.name} {hparams} "
                                f"seed={seed}: {max_abs_err:.3e} > {threshold:.3e}"
                            )
                        records.append(
                            make_run_record(
                                experiment=self.name,
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
                                extra={
                                    "merge_max_abs_err": max_abs_err,
                                    "merge_ok": True,
                                    "merge_tolerance": threshold,
                                    "merge_eval_samples": int(x_eval.shape[0]),
                                },
                            )
                        )
        return records
