"""Phase 2 tests: python -m pytest -q ternary/tests"""
import torch
import torch.nn as nn

from ternary.layers import TernaryLinear
from ternary.flip import TopKFlip, StochasticFlip, CounterFlip, make_flip_rule


class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m


def layer_with_grad(w, g):
    m = TernaryLinear(w.shape[1], w.shape[0])
    m.weight_t.copy_(w)
    m.weight_grad = g.clone()
    return Wrap(m), m


def test_topk_flips_k_largest_movable_toward_minus_sign():
    w = torch.tensor([[0, 1, -1, 0, 1, -1, 0, 0]], dtype=torch.int8)
    g = torch.tensor([[-5.0, -9.0, 9.0, 4.0, 3.0, 1.0, 0.1, 0.0]])
    # entry 1: w=+1 wants +1 -> saturated; entry 2: w=-1 wants -1 -> saturated (both excluded)
    model, m = layer_with_grad(w, g)
    rule = TopKFlip(total_steps=10, frac0=3 / 8)   # k = 3 at step 0
    frac = rule.step(model, 0)
    assert frac == 3 / 8
    # chosen: |g|=5 (0 -> +1), |g|=4 (0 -> -1), |g|=3 (+1 -> 0)
    assert m.weight_t.tolist() == [[1, 1, -1, -1, 0, -1, 0, 0]]
    assert m.weight_grad is None                       # gradient discarded


def test_topk_k_decays_to_zero():
    w = torch.zeros(10, 10, dtype=torch.int8)
    rule = TopKFlip(total_steps=100, frac0=0.1)
    model, m = layer_with_grad(w, torch.randn(10, 10))
    assert rule.step(model, 50) == 0.05                # k = 10 * 0.5 = 5 of 100
    model, m = layer_with_grad(w, torch.randn(10, 10))
    assert rule.step(model, 100) == 0.0


def test_values_always_stay_ternary_int8():
    torch.manual_seed(0)
    w = torch.randint(-1, 2, (64, 64), dtype=torch.int8)
    for rule in [TopKFlip(5, 0.2), StochasticFlip(5, 0.5), CounterFlip(5, 2)]:
        m = TernaryLinear(64, 64); m.weight_t.copy_(w); model = Wrap(m)
        for s in range(5):
            m.weight_grad = torch.randn(64, 64)
            rule.step(model, s)
            assert m.weight_t.dtype == torch.int8 and set(m.weight_t.unique().tolist()) <= {-1, 0, 1}


def test_stochastic_rate_and_direction():
    torch.manual_seed(0)
    n = 200_000
    w = torch.zeros(1, n, dtype=torch.int8)
    g = torch.randn(1, n)
    model, m = layer_with_grad(w, g)
    frac = StochasticFlip(total_steps=1000, lr=1e-2).step(model, 0, torch.Generator().manual_seed(1))
    assert 0.008 < frac < 0.012                        # ~lr of the weights flip at step 0
    moved = m.weight_t != 0
    assert torch.all(m.weight_t[moved] == -torch.sign(g[moved]).to(torch.int8))
    # bigger |g| -> more likely to flip
    assert g.abs()[moved].mean() > 1.5 * g.abs().mean()


def test_counter_flips_after_threshold_and_resets():
    w = torch.zeros(1, 2, dtype=torch.int8)
    model, m = layer_with_grad(w, torch.tensor([[-1.0, 1.0]]))
    rule = CounterFlip(total_steps=100, threshold=3)
    assert rule.breaks_no_hidden_copy_rule
    for s in range(2):
        rule.step(model, s); m.weight_grad = torch.tensor([[-1.0, 1.0]])
        assert m.weight_t.tolist() == [[0, 0]]
    rule.step(model, 2)
    assert m.weight_t.tolist() == [[1, -1]] and m.flip_counter.tolist() == [[0, 0]]


def test_make_flip_rule():
    assert make_flip_rule("none", 10) is None
    assert isinstance(make_flip_rule("topk", 10), TopKFlip)


def _recover_hidden_matrix(rule, steps=300):
    """Regression toy: targets come from a hidden ternary matrix; train ONLY by flipping."""
    torch.manual_seed(0)
    d_in, d_out = 32, 16
    w_true = torch.randint(-1, 2, (d_out, d_in)).float()
    m = TernaryLinear(d_in, d_out, quant_act=False)    # starts all-zero
    m.scale.data.fill_(1.0)                            # kept fixed: its .grad is computed but never applied
    model = Wrap(m)
    gen = torch.Generator().manual_seed(0)
    for s in range(steps):
        x = torch.randn(256, d_in)
        loss = ((m(x) - x @ w_true.t()) ** 2).mean()
        loss.backward()
        rule.step(model, s, gen)
        m.scale.grad = None
    return (m.weight_t.float() == w_true).float().mean().item()


def test_flip_training_learns_toy_problem():
    assert _recover_hidden_matrix(TopKFlip(300, frac0=0.02)) > 0.95
    assert _recover_hidden_matrix(StochasticFlip(300, lr=0.05)) > 0.95


def test_flip_schedules():
    lin = TopKFlip(100)                                   # default = Phase 2/3 behaviour
    assert lin.decay(0) == 1.0 and abs(lin.decay(50) - 0.5) < 1e-9 and lin.decay(100) == 0.0
    fl = StochasticFlip(100, schedule="linear", floor=0.2)
    assert abs(fl.decay(100) - 0.2) < 1e-9 and abs(fl.decay(50) - 0.6) < 1e-9
    cos = StochasticFlip(100, schedule="cosine", floor=0.1)
    assert abs(cos.decay(0) - 1.0) < 1e-9 and abs(cos.decay(50) - 0.55) < 1e-9 and abs(cos.decay(100) - 0.1) < 1e-9
    assert StochasticFlip(100, schedule="constant").decay(99) == 1.0
