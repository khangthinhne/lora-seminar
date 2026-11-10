"""Experiment interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from lora_lab.config import ExperimentConfig
from lora_lab.records import RunRecord


class Experiment(ABC):
    """A unit of work that returns one :class:`RunRecord` per run."""

    name: ClassVar[str]

    @abstractmethod
    def run(self, cfg: ExperimentConfig) -> list[RunRecord]:
        """Run the experiment and return records. The runner writes the CSV."""
