"""Phase 1 tests: python -m pytest -q ternary/tests"""
import numpy as np
import pytest
import torch

from nanochat.gpt import GPT, GPTConfig, Linear
from ternary.layers import (TernaryLinear, TernaryEmbedding, act_quant_int8, absmean_ternarize,
                            ternary_matmul_addsub_reference)
from ternary.model import TernaryConfig, convert_to_ternary, setup_optimizer_ternary, ternary_modules


def rand_ternary(out_f, in_f, p_zero=0.4, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(-1, 2, (out_f, in_f), generator=g, dtype=torch.int8)
    w[torch.rand(out_f, in_f, generator=g) < p_zero] = 0
    return w


# ---------------------------------------------------------------- quantisers
def test_act_quant_range_and_error():
    x = torch.randn(5, 7, 64) * 3
    xi, inv_s = act_quant_int8(x)
    assert torch.equal(xi, xi.round()) and xi.min() >= -128 and xi.max() <= 127
    # max reconstruction error per token is half a quantisation step = absmax / 254
    err = (xi * inv_s - x).abs().amax(-1)
    assert torch.all(err <= x.abs().amax(-1) / 254 + 1e-6)


def test_absmean_ternarize_values():
    w = torch.randn(32, 48)
    wt, gamma = absmean_ternarize(w)
    assert wt.dtype == torch.int8 and set(wt.unique().tolist()) <= {-1, 0, 1}
    assert abs(gamma - w.abs().mean().item()) < 1e-6
    wt0, g0 = absmean_ternarize(torch.zeros(4, 4))
    assert g0 is None and torch.all(wt0 == 0)


# ---------------------------------------------------------------- TernaryLinear
def test_storage_is_int8_ternary_plus_one_scale():
    m = TernaryLinear.from_float(torch.randn(16, 32))
    sd = m.state_dict()
    assert set(sd.keys()) == {"weight_t", "scale"}
    assert sd["weight_t"].dtype == torch.int8 and set(sd["weight_t"].unique().tolist()) <= {-1, 0, 1}
    assert sd["scale"].numel() == 1
    assert [n for n, _ in m.named_parameters()] == ["scale"]  # no float weight matrix anywhere


@pytest.mark.parametrize("p_zero", [0.0, 0.5, 0.9])
def test_forward_equals_literal_add_subtract(p_zero):
    torch.manual_seed(0)
    m = TernaryLinear(24, 10)
    m.weight_t.copy_(rand_ternary(10, 24, p_zero))
    m.scale.data.fill_(0.37)
    x = torch.randn(2, 3, 24)
    y = m(x)
    xi, inv_s = act_quant_int8(x)
    acc_ref = ternary_matmul_addsub_reference(xi.reshape(-1, 24).numpy().astype(np.int64), m.weight_t.numpy())
    # integer accumulators from the fast path are exactly the add/subtract ones
    acc_fast = (xi.reshape(-1, 24) @ m.weight_t.float().t()).numpy()
    assert np.array_equal(acc_fast.astype(np.int64), acc_ref) and np.all(acc_fast == np.round(acc_fast))
    y_ref = torch.from_numpy(acc_ref).float().reshape(2, 3, 10) * inv_s * 0.37
    assert torch.allclose(y, y_ref, rtol=0, atol=1e-6)


def test_gradients_match_float_equivalent():
    torch.manual_seed(1)
    m = TernaryLinear(20, 12)
    m.weight_t.copy_(rand_ternary(12, 20))
    m.scale.data.fill_(0.5)
    x = torch.randn(4, 20, requires_grad=True)
    t = torch.randn(4, 12)
    ((m(x) - t) ** 2).sum().backward()

    # float reference: y = x_deq @ (scale*W)^T with straight-through activation rounding
    xi, inv_s = act_quant_int8(x.detach())
    x2 = x.detach().clone().requires_grad_(True)
    x_ste = x2 + (xi * inv_s - x2).detach()
    s = torch.tensor(0.5, requires_grad=True)
    W = m.weight_t.float().clone().requires_grad_(True)
    ((x_ste @ (s * W).t() - t) ** 2).sum().backward()

    assert torch.allclose(m.weight_grad, W.grad, atol=1e-5)    # dL/dW for each ternary weight
    assert torch.allclose(x.grad, x2.grad, atol=1e-5)
    assert torch.allclose(m.scale.grad, s.grad, atol=1e-4)


def test_weight_grad_accumulates_and_clears():
    m = TernaryLinear.from_float(torch.randn(6, 8))
    x = torch.randn(3, 8)
    m(x).sum().backward(); g1 = m.weight_grad.clone()
    m(x).sum().backward()
    assert torch.allclose(m.weight_grad, 2 * g1)
    m.clear_weight_grad(); assert m.weight_grad is None
    m.capture_grad = False
    m(x).sum().backward(); assert m.weight_grad is None


def test_zero_init_matrix_still_gets_gradient():
    m = TernaryLinear.from_float(torch.zeros(6, 8))
    assert torch.all(m.weight_t == 0) and m.scale.item() > 0
    m(torch.randn(3, 8)).sum().backward()
    assert m.weight_grad.abs().sum() > 0


# ---------------------------------------------------------------- TernaryEmbedding
def test_embedding_forward_and_grad():
    torch.manual_seed(2)
    e = TernaryEmbedding.from_float(torch.randn(50, 8))
    idx = torch.tensor([[1, 2, 2, 49]])
    out = e(idx)
    assert torch.allclose(out, e.weight_t[idx].float() * e.scale)
    g = torch.randn_like(out)
    (out * g).sum().backward()
    exp = torch.zeros(50, 8); exp.index_add_(0, idx.reshape(-1), g.reshape(-1, 8))
    assert torch.allclose(e.weight_grad, exp * e.scale.detach(), atol=1e-6)


# ---------------------------------------------------------------- model conversion
def tiny_model():
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=32, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device="cpu"); m.init_weights()
    return m


@pytest.mark.parametrize("embed,vemb", [(False, False), (True, False), (True, True)])
def test_convert_model_and_train_step(embed, vemb):
    m = tiny_model()
    convert_to_ternary(m, TernaryConfig(linear=True, embed=embed, value_embeds=vemb))
    mods = dict(ternary_modules(m))
    assert len(mods) == 6 * 2 + int(embed) + (len(m.value_embeds) if vemb else 0)
    assert isinstance(m.lm_head, Linear)                       # head stays full precision
    assert isinstance(m.transformer.h[0].attn.ve_gate or m.transformer.h[1].attn.ve_gate, Linear)
    # zero-init projections stay all-zero ternary; random-init ones are mixed
    assert torch.all(m.transformer.h[0].mlp.c_proj.weight_t == 0)
    assert len(m.transformer.h[0].attn.c_q.weight_t.unique()) == 3

    opt = setup_optimizer_ternary(m)
    before = {n: mm.weight_t.clone() for n, mm in mods.items()}
    idx = torch.randint(0, 128, (2, 16))
    loss = m(idx, idx)
    assert torch.isfinite(loss)
    loss.backward(); opt.step()
    for n, mm in mods.items():                                  # no optimizer touches ternary weights
        assert torch.equal(mm.weight_t, before[n]) and mm.weight_t.dtype == torch.int8
        assert mm.weight_grad is not None                       # gradient available for Phase 2


def test_eval_helpers_work_with_ternary_embedding():
    # regression: nanochat's evaluate_bpb calls model.get_device() -> wte.weight.device
    m = tiny_model()
    convert_to_ternary(m, TernaryConfig(linear=True, embed=True, value_embeds=True))
    assert m.get_device() == torch.device("cpu")
    assert m.transformer.wte.weight.dtype == torch.int8
    idx = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        per_tok = m(idx, idx, loss_reduction="none")
    assert per_tok.shape == (32,) and torch.isfinite(per_tok).all()


def test_original_model_unchanged_by_default():
    m = tiny_model()
    assert not ternary_modules(m)
    assert isinstance(m.transformer.h[0].attn.c_q, Linear)


def test_linear_saves_int8_activations_only():
    """Memory fix: between forward and backward only the int8 activations, one float per token and the
    scale are kept (no float32 copy of the input, no copy of the output)."""
    m = TernaryLinear.from_float(torch.randn(64, 32))
    x = torch.randn(10, 32, requires_grad=True)
    saved = []

    def pack(t):
        saved.append((t.dtype, t.numel()))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = m(x)
    nbytes = sum(n * torch.empty((), dtype=d).element_size() for d, n in saved)
    assert (torch.int8, 10 * 32) in saved
    assert nbytes == 10 * 32 * 1 + 10 * 4 + 4          # int8 x, per-token scale, matrix scale
    y.sum().backward()
    assert m.weight_grad is not None and x.grad is not None
