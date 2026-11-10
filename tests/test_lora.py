from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from lora_lab.methods.lora import LoRALinear


def _make_layer(r: int = 3, alpha: float = 6.0) -> LoRALinear:
    w0 = torch.randn(9, 7, generator=torch.Generator().manual_seed(0))
    return LoRALinear(w0, r=r, alpha=alpha, init_std=0.01)


def test_b_is_zero_and_a_is_nonzero() -> None:
    layer = _make_layer()
    assert torch.count_nonzero(layer.B) == 0
    assert torch.count_nonzero(layer.A) > 0
    assert torch.count_nonzero(layer.effective_delta()) == 0


def test_parameter_count_is_r_times_d_plus_k() -> None:
    layer = _make_layer(r=3)
    assert layer.trainable_param_count() == 3 * (9 + 7)


def test_alpha_over_r_scaling() -> None:
    layer = _make_layer(r=3, alpha=6.0)
    with torch.no_grad():
        layer.B.normal_(generator=torch.Generator().manual_seed(2))
    expected = (6.0 / 3.0) * (layer.B @ layer.A)
    torch.testing.assert_close(layer.effective_delta(), expected)


def test_forward_matches_rank_path_definition() -> None:
    layer = _make_layer(r=4, alpha=8.0)
    with torch.no_grad():
        layer.B.normal_(generator=torch.Generator().manual_seed(3))
    x = torch.randn(6, 7, generator=torch.Generator().manual_seed(4))
    expected = F.linear(x, layer.w0) + (8.0 / 4.0) * F.linear(F.linear(x, layer.A), layer.B)
    torch.testing.assert_close(layer(x), expected)


def test_merge_equals_forward_after_update() -> None:
    layer = _make_layer()
    with torch.no_grad():
        layer.B.normal_(generator=torch.Generator().manual_seed(5))
    x = torch.randn(6, 7, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        torch.testing.assert_close(layer.merge()(x), layer(x), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("bad_r", [0, -1, 8])
def test_invalid_rank_raises(bad_r: int) -> None:
    with pytest.raises(ValueError, match="r must satisfy"):
        _make_layer(r=bad_r)


def test_non_integer_rank_raises() -> None:
    with pytest.raises(ValueError, match="r must be an integer"):
        LoRALinear(torch.randn(4, 4), r=2.0, alpha=4.0, init_std=0.01)  # type: ignore[arg-type]


def test_missing_hyperparameter_raises() -> None:
    with pytest.raises(ValueError, match=r"requires hyperparameter 'r'"):
        LoRALinear.from_base(torch.randn(4, 4), alpha=4.0)
