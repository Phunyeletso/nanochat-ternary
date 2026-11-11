"""
Baseline B (Phase 3): standard BitNet b1.58 linear layer, WITH latent full-precision weights.

    gamma = mean|W_latent|                         (one scale per matrix)
    W_q   = clip(round(W_latent / gamma), -1, 1)   (ternary, recomputed every forward pass)
    y     = BitLinear(x) = (Q8(x) @ W_q^T) * gamma (8-bit absmax per-token activations)
Gradients reach W_latent through the straight-through estimator (rounding treated as identity),
and nanochat's normal optimizer (Muon for these matrices) updates W_latent, with the same
hyperparameters as the full-precision model A. Only the six block matrices are converted,
exactly the same set as model C.
"""
import torch
import torch.nn.functional as F

from nanochat.gpt import Linear
from ternary.layers import act_quant_int8
from ternary.model import TARGETS


def weight_quant_ternary_ste(w, eps=1e-5):
    gamma = w.abs().mean().clamp(min=eps)
    wq = (w / gamma).round().clamp(-1, 1) * gamma
    return w + (wq - w).detach()


def act_quant_ste(x):
    xi, inv_s = act_quant_int8(x)
    return x + (xi * inv_s - x).detach()


class BitNetLinear(Linear):
    """Subclass of nanochat's Linear, so the latent weight is an ordinary 2-D parameter that
    GPT.setup_optimizer puts in its Muon groups, just like model A."""

    def forward(self, x):
        w = weight_quant_ternary_ste(self.weight.float())
        return F.linear(act_quant_ste(x.float()), w).to(x.dtype)


def convert_to_bitnet(model):
    for block in model.transformer.h:
        for sub, name in TARGETS:
            parent = getattr(block, sub)
            lin = getattr(parent, name)
            b = BitNetLinear(lin.in_features, lin.out_features, bias=False, device=lin.weight.device)
            with torch.no_grad():
                b.weight.copy_(lin.weight)
            setattr(parent, name, b)
    return model
