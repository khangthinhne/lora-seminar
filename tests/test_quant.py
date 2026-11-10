from __future__ import annotations

import pytest
import torch

import lora_lab.quant  # noqa: F401  (populates QUANTIZERS)
from lora_lab.quant.absmax_block import AbsmaxBlockQuantizer
from lora_lab.quant.identity import IdentityQuantizer
from lora_lab.registry import QUANTIZERS


def test_registry_contains_quantizers() -> None:
    assert {"none", "absmax_block4"} <= set(QUANTIZERS)


def test_identity_is_exact() -> None:
    quantizer = IdentityQuantizer()
    w = torch.randn(5, 11, generator=torch.Generator().manual_seed(0))
    q = quantizer.quantize(w)
    torch.testing.assert_close(quantizer.dequantize(q), w)
    assert quantizer.storage_bytes(q) == w.numel() * w.element_size()


def test_absmax_round_trip_error_bound_and_storage() -> None:
    quantizer = AbsmaxBlockQuantizer(block_size=8, bits=4)
    w = torch.randn(4, 10, generator=torch.Generator().manual_seed(1))
    q = quantizer.quantize(w)
    restored = quantizer.dequantize(q)
    assert restored.shape == w.shape

    flat = w.reshape(-1)
    for start in range(0, flat.numel(), 8):
        block = flat[start : start + 8]
        bound = float(block.abs().max()) / 8.0 + 1e-6
        assert float((restored.reshape(-1)[start : start + 8] - block).abs().max()) <= bound

    code_bytes = (w.numel() + 1) // 2
    scale_bytes = q.scale.numel() * 4
    assert quantizer.storage_bytes(q) == code_bytes + scale_bytes
    assert quantizer.storage_bytes(q) > code_bytes


def test_absmax_handles_padding_and_zero_blocks() -> None:
    quantizer = AbsmaxBlockQuantizer(block_size=4, bits=4)
    w = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    q = quantizer.quantize(w)
    torch.testing.assert_close(quantizer.dequantize(q), w)


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_invalid_block_size_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="block_size"):
        AbsmaxBlockQuantizer(block_size=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, 9, 3.0])
def test_invalid_bits_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="bits"):
        AbsmaxBlockQuantizer(bits=bad)  # type: ignore[arg-type]
