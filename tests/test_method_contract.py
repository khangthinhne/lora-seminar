"""Generic contract test: every registered method must satisfy the interface."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import lora_lab.methods  # noqa: F401  (populates METHODS)
from lora_lab.registry import METHODS
from lora_lab.registry import METHODS as _METHODS


@pytest.mark.parametrize("method_name", sorted(_METHODS))
def test_method_contract(method_name: str) -> None:
    cls = METHODS[method_name]
    generator = torch.Generator().manual_seed(7)
    w0 = torch.randn(7, 5, generator=generator)
    x = torch.randn(4, 5, generator=generator)
    layer = cls.from_base(
        w0,
        init_generator=torch.Generator().manual_seed(11),
        **cls.default_hparams,
    )

    output = layer(x)
    assert output.shape == (4, 7)
    assert layer.effective_delta().shape == (7, 5)
    assert layer.trainable_param_count() > 0
    assert isinstance(layer.merge(), nn.Linear)

    if layer.zero_init:
        assert torch.allclose(layer.effective_delta(), torch.zeros(7, 5))

    target = torch.randn(4, 7, generator=generator)
    params = [parameter for parameter in layer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.05)
    loss = torch.mean((layer(x) - target) ** 2)
    optimizer.zero_grad()
    loss.backward()

    if hasattr(layer, "w0"):
        assert layer.w0.grad is None
    optimizer.step()

    merged = layer.merge()
    with torch.no_grad():
        torch.testing.assert_close(merged(x), layer(x), atol=1e-6, rtol=1e-5)


def test_all_expected_methods_are_registered() -> None:
    assert "full_ft" in METHODS
    assert "lora" in METHODS
