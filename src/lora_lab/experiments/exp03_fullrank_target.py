"""exp03 - limitation experiment with a full-rank target.

Hypothesis: when ``delta_star`` has a long spectral tail (small decay exponent
gamma), small LoRA ranks cannot represent the tail and will be worse than full
fine-tuning; larger gamma concentrates energy in a few directions, which should
narrow the gap. Setup: the same protocol as exp02 with ``decaying_spectrum``
targets and gamma in {0.5, 1.0, 2.0}. Expected outcome (to verify): the LoRA
gap grows as gamma decreases. Results are discussed only in this toy setup.
"""

from __future__ import annotations

from lora_lab.config import ExperimentConfig
from lora_lab.experiments.base import Experiment
from lora_lab.experiments.common import run_rank_sweep
from lora_lab.records import RunRecord
from lora_lab.registry import EXPERIMENTS, register


@register(EXPERIMENTS, "exp03_fullrank_target")
class FullRankTargetExperiment(Experiment):
    """Limitation experiment: full-rank decaying-spectrum target."""

    name = "exp03_fullrank_target"

    def run(self, cfg: ExperimentConfig) -> list[RunRecord]:
        return run_rank_sweep(cfg, self.name)
