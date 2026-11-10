"""exp02 - main rank sweep against full fine-tuning.

Hypothesis: with a true rank-8 target, LoRA ranks below 8 underfit while ranks
>= 8 approach full fine-tuning at a much smaller trainable parameter count.
Setup: synthetic ``W0``, low-rank ``delta_star`` (rank 8), matching
train/validation/test splits, five seeds, and a shared learning-rate grid
selected on validation. Expected outcome: test MSE falls as rank grows, with a
visible gap for ranks 1 and 2. Results are reported as measured, not forced.
"""

from __future__ import annotations

from lora_lab.config import ExperimentConfig
from lora_lab.experiments.base import Experiment
from lora_lab.experiments.common import run_rank_sweep
from lora_lab.records import RunRecord
from lora_lab.registry import EXPERIMENTS, register


@register(EXPERIMENTS, "exp02_rank_sweep")
class RankSweepExperiment(Experiment):
    """Main experiment: rank sweep versus full fine-tuning."""

    name = "exp02_rank_sweep"

    def run(self, cfg: ExperimentConfig) -> list[RunRecord]:
        return run_rank_sweep(cfg, self.name)
