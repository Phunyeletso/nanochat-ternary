"""Phase 3 tests for baseline B (standard BitNet b1.58 with latent weights)."""
import torch

from nanochat.gpt import GPT, GPTConfig
from ternary.bitnet import BitNetLinear, convert_to_bitnet, weight_quant_ternary_ste
from ternary.layers import act_quant_int8


def test_weight_quant_is_ternary_times_gamma_with_identity_gradient():
    w = torch.randn(8, 16, requires_grad=True)
    wq = weight_quant_ternary_ste(w)
    gamma = w.detach().abs().mean()
    assert set((wq.detach() / gamma).round().unique().tolist()) <= {-1.0, 0.0, 1.0}
    wq.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))      # straight-through


def test_bitnet_linear_forward_matches_definition():
    torch.manual_seed(0)
    m = BitNetLinear(16, 8, bias=False)
    x = torch.randn(3, 16)
    w = m.weight.detach()
    gamma = w.abs().mean()
    wt = (w / gamma).round().clamp(-1, 1)
    xi, inv_s = act_quant_int8(x)
    assert torch.allclose(m(x), (xi @ wt.t()) * inv_s * gamma, atol=1e-5)


def test_bitnet_model_trains_latent_weights_with_original_optimizer():
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=32, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
    with torch.device("meta"):
        m = GPT(cfg)
    m.to_empty(device="cpu"); m.init_weights()
    convert_to_bitnet(m)
    assert isinstance(m.transformer.h[0].attn.c_q, BitNetLinear)
    opt = m.setup_optimizer()                               # nanochat's own optimizer, unchanged
    w0 = m.transformer.h[0].mlp.c_proj.weight.detach().clone()   # c_fc gets no grad at step 0 (zero-init c_proj blocks it, same as model A)
    idx = torch.randint(0, 128, (2, 16))
    m(idx, idx).backward(); opt.step()
    assert m.transformer.h[0].mlp.c_fc.weight.dtype == torch.float32   # latent float weights exist (that's B)
    assert not torch.equal(w0, m.transformer.h[0].mlp.c_proj.weight)
