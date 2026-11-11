"""
Ternary layers for nanochat (Phase 1).

Storage rule (no latent weights):
  * The ONLY stored weight is `weight_t`, an int8 buffer whose values are in {-1, 0, +1}.
  * Plus ONE float scale per matrix (`scale`, an nn.Parameter trained by AdamW like other scalars).
  * There is no hidden full-precision copy. During a forward pass the int8 matrix is cast to float
    just to feed the matmul kernel; that temporary holds exactly the same {-1,0,+1} values and is
    thrown away. It carries no state from step to step.

Forward semantics (TernaryLinear), per output row r:
    acc_r = sum_{j: W[r,j]=+1} x_q[j]  -  sum_{j: W[r,j]=-1} x_q[j]      (weights that are 0 are skipped)
    y_r   = acc_r * scale * (absmax(x) / 127)
where x_q are the 8-bit (absmax, per token) quantised activations, as in BitNet b1.58.
`ternary_matmul_addsub_reference` below does this literally (adds and subtracts, no multiplies).
The fast path multiplies by a float copy of the {-1,0,+1} matrix, which is mathematically the same
operation; because x_q are integers in [-128,127], the accumulator is an exact integer in float32,
so the fast path and the add/subtract reference agree EXACTLY (tested in tests).

Gradients for Phase 2 (no latent weights):
  Backprop computes dL/dW for each ternary weight (dL/dW = scale * g^T x_deq). It is accumulated
  into `module.weight_grad` (a temporary float32 buffer) so a flip rule can read it and then
  discard it (`module.weight_grad = None`). It is never applied to any stored float weight.
  Activations use the straight-through estimator for the 8-bit rounding (standard BitNet practice).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Quantisers

def act_quant_int8(x, eps=1e-5, qmax=127):
    """Per-token absmax quantisation. Returns (x_int, inv_s) with x ~= x_int * inv_s.
    qmax=127 (default): 8-bit, integers in [-128, 127].
    qmax=(3^k-1)/2: integers in [-qmax, qmax], i.e. exactly what k balanced-ternary "coin tables" can hold
    (k=4 -> +-40, k=3 -> +-13, k=2 -> +-4, k=1 -> +-1 = pure ternary data)."""
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    s = qmax / absmax
    lo = -128 if qmax == 127 else -qmax
    x_int = (x * s).round().clamp(lo, qmax)
    return x_int, 1.0 / s


def qmax_for_tables(k):
    """Largest number k balanced-ternary tables (coins 1, 3, 9, ...) can write. k=0 means 8-bit (127)."""
    return 127 if k == 0 else (3 ** k - 1) // 2


def balanced_ternary_planes(x_int, k):
    """Split integers |x| <= (3^k-1)/2 into k tables of -1/0/+1 (ones, threes, nines, ...): x = sum 3^i * plane_i."""
    planes, r = [], x_int.clone()
    for _ in range(k):
        d = torch.remainder(r, 3)            # 0, 1, 2
        d = torch.where(d == 2, torch.full_like(d, -1), d)
        planes.append(d)
        r = (r - d) / 3
    assert torch.all(r == 0), "number too large for this many tables"
    return planes


def ternary_planes_matmul(x_int, w_t, k):
    """x_int @ W^T computed as k ternary x ternary multiplies (one per coin table), then combined with 1, 3, 9, ..."""
    wf = w_t.to(torch.float32)
    return sum((3 ** i) * (p @ wf.t()) for i, p in enumerate(balanced_ternary_planes(x_int, k)))


def absmean_ternarize(w, eps=1e-8):
    """BitNet b1.58 rule: gamma = mean|W|, W_t = clip(round(W / gamma), -1, 1). Used ONLY to turn
    nanochat's float initialisation into an initial ternary matrix (once, at init)."""
    gamma = w.abs().mean()
    if gamma < eps:
        return torch.zeros_like(w, dtype=torch.int8), None
    wt = (w / gamma).round().clamp(-1, 1).to(torch.int8)
    return wt, float(gamma)


# -----------------------------------------------------------------------------
# Autograd functions

class _TernaryLinearFn(torch.autograd.Function):
    """Memory-lean version (Phase 5): saves the 8-bit activations as int8 (1 byte/value) plus one float per
    token, instead of a float32 copy of the input AND a float32 copy of the output. The scale's gradient is
    computed from the weight gradient: dL/dscale = sum(G_w * W), where G_w = g^T x_deq."""

    @staticmethod
    def forward(ctx, x, scale, module):
        wf = module.weight_t.to(torch.float32)  # transient cast of the {-1,0,+1} values (not stored)
        x32 = x.to(torch.float32)
        if module.quant_act:
            x_int, inv_s = act_quant_int8(x32, qmax=getattr(module, "act_qmax", 127))
            acc = x_int @ wf.t()                # exact integer add/subtract result
            y = acc * inv_s * scale
            ctx.save_for_backward(x_int.to(torch.int8), inv_s, scale)
        else:
            y = (x32 @ wf.t()) * scale
            ctx.save_for_backward(x32, None, scale)
        ctx.module = module
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x_saved, inv_s, scale = ctx.saved_tensors
        module = ctx.module
        x_deq = x_saved.to(torch.float32) * inv_s if inv_s is not None else x_saved
        g32 = g.to(torch.float32)
        wf = module.weight_t.to(torch.float32)
        grad_x = (g32 @ wf) * scale                       # straight-through for activation rounding
        gw = g32.reshape(-1, g32.shape[-1]).t() @ x_deq.reshape(-1, x_deq.shape[-1])   # dL/d(scale*W) / ... unscaled
        grad_scale = (gw * wf).sum().reshape(scale.shape)
        if module.capture_grad:
            module._accumulate_weight_grad(gw * scale)
        return grad_x.to(g.dtype), grad_scale, None


class _TernaryEmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, idx, scale, module):
        rows = module.weight_t[idx].to(torch.float32)
        ctx.module = module
        ctx.save_for_backward(idx, scale, rows)
        return rows * scale

    @staticmethod
    def backward(ctx, g):
        idx, scale, rows = ctx.saved_tensors
        module = ctx.module
        g32 = g.to(torch.float32)
        grad_scale = (g32 * rows).sum().reshape(scale.shape)
        if module.capture_grad:
            gw = torch.zeros(module.weight_t.shape, dtype=torch.float32, device=g.device)
            gw.index_add_(0, idx.reshape(-1), g32.reshape(-1, g32.shape[-1]))
            module._accumulate_weight_grad(gw * scale)
        return None, grad_scale, None


# -----------------------------------------------------------------------------
# Modules

class _TernaryBase(nn.Module):
    def _accumulate_weight_grad(self, gw):
        if self.weight_grad is None:
            self.weight_grad = gw.detach().clone()
        else:
            self.weight_grad.add_(gw)

    def zero_fraction(self):
        return (self.weight_t == 0).float().mean().item()

    def clear_weight_grad(self):
        self.weight_grad = None

    @property
    def weight(self):
        """Read-only alias of the int8 ternary matrix, so nanochat code that only asks for
        `.weight.device` (e.g. GPT.get_device) keeps working. Not a float copy."""
        return self.weight_t


class TernaryLinear(_TernaryBase):
    """Bias-free linear layer with int8 {-1,0,+1} weights and one float scale per matrix."""

    def __init__(self, in_features, out_features, quant_act=True, device=None, act_qmax=127):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.quant_act = quant_act
        self.act_qmax = act_qmax
        self.capture_grad = True
        self.weight_grad = None
        self.register_buffer("weight_t", torch.zeros(out_features, in_features, dtype=torch.int8, device=device))
        self.scale = nn.Parameter(torch.full((), 0.5 * math.sqrt(3.0 / in_features), device=device))

    @classmethod
    def from_float(cls, weight, quant_act=True, act_qmax=127):
        """Build from an initialised float weight (out, in) using absmean ternarisation.
        All-zero weights (nanochat's zero-init projections) stay all-zero; their scale is set to the
        absmean of a U(-sqrt(3/in), sqrt(3/in)) matrix so gradients can flow (scale 0 would block them)."""
        out_f, in_f = weight.shape
        m = cls(in_f, out_f, quant_act=quant_act, device=weight.device, act_qmax=act_qmax)
        wt, gamma = absmean_ternarize(weight.detach().float())
        m.weight_t.copy_(wt)
        if gamma is not None:
            m.scale.data.fill_(gamma)
        return m

    def forward(self, x):
        return _TernaryLinearFn.apply(x, self.scale, self)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, int8 ternary, quant_act={self.quant_act}, act_qmax={self.act_qmax}"


class TernaryEmbedding(_TernaryBase):
    """Embedding table with int8 {-1,0,+1} entries and one float scale for the whole table."""

    def __init__(self, num_embeddings, embedding_dim, device=None):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.capture_grad = True
        self.weight_grad = None
        self.register_buffer("weight_t", torch.zeros(num_embeddings, embedding_dim, dtype=torch.int8, device=device))
        self.scale = nn.Parameter(torch.ones((), device=device))

    @classmethod
    def from_float(cls, weight):
        n, d = weight.shape
        m = cls(n, d, device=weight.device)
        wt, gamma = absmean_ternarize(weight.detach().float())
        m.weight_t.copy_(wt)
        if gamma is not None:
            m.scale.data.fill_(gamma)
        return m

    def forward(self, idx):
        return _TernaryEmbeddingFn.apply(idx, self.scale, self)

    def extra_repr(self):
        return f"{self.num_embeddings}x{self.embedding_dim}, int8 ternary"


# -----------------------------------------------------------------------------
# Literal add/subtract reference (slow, for testing only)

def ternary_matmul_addsub_reference(x_int, w_t):
    """x_int: (N, in) integer array; w_t: (out, in) array in {-1,0,1}.
    Returns (N, out) integer accumulators computed ONLY with additions/subtractions:
    +1 -> add the input, -1 -> subtract it, 0 -> skip it."""
    x_int = np.asarray(x_int, dtype=np.int64)
    w_t = np.asarray(w_t)
    n, d_in = x_int.shape
    d_out = w_t.shape[0]
    plus = [np.nonzero(w_t[r] == 1)[0] for r in range(d_out)]
    minus = [np.nonzero(w_t[r] == -1)[0] for r in range(d_out)]
    out = np.zeros((n, d_out), dtype=np.int64)
    for i in range(n):
        xi = x_int[i]
        for r in range(d_out):
            acc = 0
            for j in plus[r]:
                acc += xi[j]
            for j in minus[r]:
                acc -= xi[j]
            out[i, r] = acc
    return out
