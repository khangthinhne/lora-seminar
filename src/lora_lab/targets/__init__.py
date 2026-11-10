"""Target plugins. Importing this package registers every built-in target."""

from lora_lab.targets.base import Target
from lora_lab.targets.decaying_spectrum import DecayingSpectrumTarget
from lora_lab.targets.low_rank import LowRankTarget

__all__ = ["DecayingSpectrumTarget", "LowRankTarget", "Target"]
