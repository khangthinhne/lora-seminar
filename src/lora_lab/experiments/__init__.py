"""Experiment plugins. Importing this package registers every built-in experiment."""

from lora_lab.experiments.base import Experiment
from lora_lab.experiments.exp01_param_table import ParamTableExperiment
from lora_lab.experiments.exp02_rank_sweep import RankSweepExperiment
from lora_lab.experiments.exp03_fullrank_target import FullRankTargetExperiment
from lora_lab.experiments.exp04_merge_check import MergeCheckExperiment
from lora_lab.experiments.exp05_alpha_scaling import AlphaScalingExperiment
from lora_lab.experiments.exp06_qlora_sim import SimulatedQLoRAExperiment
from lora_lab.experiments.exp07_peft_distilbert import PeftDistilBertExperiment

__all__ = [
    "AlphaScalingExperiment",
    "Experiment",
    "FullRankTargetExperiment",
    "MergeCheckExperiment",
    "ParamTableExperiment",
    "PeftDistilBertExperiment",
    "RankSweepExperiment",
    "SimulatedQLoRAExperiment",
]
