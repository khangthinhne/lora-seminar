"""exp01 - analytical parameter and memory table.

Hypothesis: for a 1024x1024 layer, LoRA stores ``r * (d + k)`` trainable
parameters while full fine-tuning stores ``d * k``. Setup: construct each
method around a dummy base weight; no training and no data are used. Expected
outcome: 8,192 / 16,384 / 32,768 LoRA parameters for r = 4 / 8 / 16 versus
1,048,576 for full fine-tuning.

Note: trainable parameter count is not GPU memory. The memory numbers are
analytical and exclude activations.
"""

from __future__ import annotations

import torch

import lora_lab.methods  # noqa: F401  (populates METHODS)
from lora_lab.config import ExperimentConfig
from lora_lab.experiments.base import Experiment
from lora_lab.memory import dense_weight_bytes, estimate_training_memory
from lora_lab.metrics import trainable_ratio
from lora_lab.records import RunRecord
from lora_lab.registry import EXPERIMENTS, METHODS, register


@register(EXPERIMENTS, "exp01_param_table")
class ParamTableExperiment(Experiment):
    """Compute parameter counts and analytical memory from the registry."""

    name = "exp01_param_table"

    def run(self, cfg: ExperimentConfig) -> list[RunRecord]:
        d, k = cfg.data.d, cfg.data.k
        dtype_bytes = int(cfg.get_extra("dtype_bytes", 4))
        optimizer_states = int(cfg.get_extra("optimizer_states", 2))
        full_params = d * k
        weight_bytes = dense_weight_bytes(d, k, dtype_bytes)

        base = torch.zeros(d, k)
        generator = torch.Generator().manual_seed(cfg.seeds[0])
        records: list[RunRecord] = []
        for method_spec in cfg.methods:
            for hparams in method_spec.expand():
                layer = METHODS[method_spec.name].from_base(
                    base, init_generator=generator, **hparams
                )
                trainable = layer.trainable_param_count()
                memory = estimate_training_memory(
                    weight_bytes,
                    trainable,
                    optimizer_states=optimizer_states,
                )
                records.append(
                    RunRecord(
                        experiment=self.name,
                        method=method_spec.name,
                        method_hparams=dict(hparams),
                        target="none",
                        target_cfg={},
                        quantizer=cfg.quantizer.name,
                        d=d,
                        k=k,
                        seed=cfg.seeds[0],
                        lr=None,
                        epochs=0,
                        trainable_params=trainable,
                        total_params=full_params,
                        trainable_ratio=trainable_ratio(full_params, trainable),
                        train_mse=None,
                        val_mse=None,
                        test_mse=None,
                        rel_delta_err=None,
                        train_time_s=0.0,
                        extra={
                            "analytical": True,
                            "dtype_bytes": dtype_bytes,
                            "optimizer_states": optimizer_states,
                            "memory_weight_bytes": memory.weight_bytes,
                            "memory_gradient_bytes": memory.gradient_bytes,
                            "memory_optimizer_bytes": memory.optimizer_bytes,
                            "memory_total_bytes": memory.total_bytes,
                            "activations_included": False,
                        },
                    )
                )
        return records
