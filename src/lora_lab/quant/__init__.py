"""Quantizer plugins. Importing this package registers every built-in quantizer."""

from lora_lab.quant.absmax_block import AbsmaxBlockQuantizer
from lora_lab.quant.base import QuantizedTensor, Quantizer
from lora_lab.quant.identity import IdentityQuantizer

__all__ = ["AbsmaxBlockQuantizer", "IdentityQuantizer", "QuantizedTensor", "Quantizer"]
