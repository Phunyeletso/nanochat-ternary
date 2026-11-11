"""Phase 7 tests: coarser data rounding and the balanced-ternary "coin tables" multiply."""
import pytest
import torch

from ternary.layers import (act_quant_int8, qmax_for_tables, balanced_ternary_planes, ternary_planes_matmul,
                            TernaryLinear)


def test_qmax_for_tables():
    assert [qmax_for_tables(k) for k in range(5)] == [127, 1, 4, 13, 40]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_rounding_stays_in_range(k):
    q = qmax_for_tables(k)
    xi, inv_s = act_quant_int8(torch.randn(6, 50) * 3, qmax=q)
    assert xi.min() >= -q and xi.max() <= q and torch.equal(xi, xi.round())
    assert (xi.abs() == q).any()                              # the largest value per token reaches the edge


def test_default_is_unchanged_8bit():
    x = torch.randn(4, 30)
    a, sa = act_quant_int8(x)
    b, sb = act_quant_int8(x, qmax=127)
    assert torch.equal(a, b) and torch.equal(sa, sb)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_planes_are_ternary_and_exact(k):
    q = qmax_for_tables(k)
    x = torch.randint(-q, q + 1, (7, 33)).float()
    planes = balanced_ternary_planes(x, k)
    assert len(planes) == k
    for p in planes:
        assert set(p.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.equal(sum((3 ** i) * p for i, p in enumerate(planes)), x)
    ones, threes = balanced_ternary_planes(torch.tensor([2.0, 4.0, 3.0]), 2)   # 2 = 3-1, 4 = 3+1, 3 = 3+0
    assert torch.equal(threes, torch.tensor([1.0, 1.0, 1.0])) and torch.equal(ones, torch.tensor([-1.0, 1.0, 0.0]))


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_stacked_tables_multiply_equals_normal_multiply(k):
    q = qmax_for_tables(k)
    x = torch.randint(-q, q + 1, (5, 40)).float()
    w = torch.randint(-1, 2, (12, 40), dtype=torch.int8)
    assert torch.equal(ternary_planes_matmul(x, w, k), x @ w.float().t())


def test_layer_uses_coarser_rounding():
    m = TernaryLinear.from_float(torch.randn(8, 16), act_qmax=qmax_for_tables(1))
    x = torch.randn(3, 16, requires_grad=True)
    y = m(x)
    xi, inv_s = act_quant_int8(x.detach(), qmax=1)
    assert torch.allclose(y, (xi @ m.weight_t.float().t()) * inv_s * m.scale, atol=1e-6)
    y.sum().backward()
    assert x.grad is not None and m.weight_grad is not None
