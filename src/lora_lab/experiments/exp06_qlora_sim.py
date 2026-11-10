"""exp06 - optional simulated QLoRA experiment.

Hypothesis: keeping the frozen base in simple 4-bit storage adds quantization
error relative to a full-precision base, while the LoRA adapter remains in high
precision. Setup: identical data and training for ``quantizer=none`` and
``quantizer=absmax_block4`` (block size 64). Expected outcome: a small but
nonzero distance to ``W*`` caused by quantization. This is simple 4-bit absmax
quantization, **not NF4**, and it is a toy simulation only.
"""

from __future__ import annotations

from lora_lab.config import ExperimentConfig, parse_component_config
from lora_lab.experiments.base import Experiment
from lora_lab.experiments.common import run_sweep
from lora_lab.records import RunRecord
from lora_lab.registry import EXPERIMENTS, register


@register(EXPERIMENTS, "exp06_qlora_sim")
class SimulatedQLoRAExperiment(Experiment):
    """Compare LoRA on a full-precision versus simple 4-bit base weight."""

    name = "exp06_qlora_sim"

    def run(self, cfg: ExperimentConfig) -> list[RunRecord]:
        raw_quantizers = cfg.require_extra("quantizers", (list, tuple))
        if not raw_quantizers:
            raise ValueError("exp06 requires a non-empty extra.quantizers list")
        quantizers = [
            parse_component_config(value, f"extra.quantizers[{index}]")
            for index, value in enumerate(raw_quantizers)
        ]
        return run_sweep(cfg, self.name, quantizers)
