"""
Phase 2: training ternary weights WITHOUT latent weights.

Each step, backprop leaves dL/dW for every ternary weight in `module.weight_grad` (a temporary buffer).
A flip rule reads it, changes some int8 weights by one step toward -sign(gradient)
(-1 -> 0 -> +1 or +1 -> 0 -> -1), and then the gradient is discarded. No float copy of the
weights is kept between steps.

Rules (choose with --flip-rule):
  topk       For each matrix, among weights that CAN still move toward -sign(g) (i.e. not already
             at +1 wanting +, or -1 wanting -), pick the k with the largest |g| and move each one step.
             k = round(frac0 * numel * decay(step)), decay = 1 - step/total (linear to zero).
             (Saturated weights are excluded so that k means "k actual changes"; picking them would
             only waste part of the budget on no-ops after clamping.)
  stochastic Each movable weight moves one step toward -sign(g) with probability
             p = min(1, lr * decay(step) * |g| / mean|g|). Normalising by the matrix's mean |g| makes lr
             mean the same thing in every matrix: at the start ~lr of each matrix's weights flip.
  counter    FALLBACK, breaks the "no hidden copy" rule: an int8 counter per weight accumulates
             -sign(g); when |counter| >= threshold the weight moves one step that way and the counter
             resets. Only to be used if topk and stochastic both fail.
"""
import math

import torch

from ternary.model import ternary_modules


def _movable(w, direction):
    """direction in {-1,0,+1} (int8). True where w + direction stays inside [-1, 1] and direction != 0."""
    return ((direction > 0) & (w < 1)) | ((direction < 0) & (w > -1))


class FlipRule:
    name = "base"
    breaks_no_hidden_copy_rule = False

    def __init__(self, total_steps, schedule="linear", floor=0.0):
        """schedule: 'linear' (1 -> floor, straight line), 'cosine' (1 -> floor, cosine curve),
        'constant' (always 1). floor = fraction of the starting rate kept at the last step.
        Default linear with floor 0 = the Phase 2/3 behaviour."""
        assert schedule in ("linear", "cosine", "constant")
        self.total_steps = total_steps
        self.schedule = schedule
        self.floor = floor

    def decay(self, step):
        f = min(max(step / self.total_steps, 0.0), 1.0)
        if self.schedule == "constant":
            return 1.0
        if self.schedule == "linear":
            base = 1.0 - f
        else:
            base = 0.5 * (1.0 + math.cos(math.pi * f))
        return self.floor + (1.0 - self.floor) * base

    def flip_module(self, m, step, gen):
        raise NotImplementedError

    @torch.no_grad()
    def step(self, model, step, gen=None):
        """Apply the rule to every ternary module, discard gradients, return fraction of weights changed."""
        changed, total = 0, 0
        for _, m in ternary_modules(model):
            total += m.weight_t.numel()
            if m.weight_grad is not None:
                changed += self.flip_module(m, step, gen)
            m.weight_grad = None  # discard
        return changed / max(total, 1)


class TopKFlip(FlipRule):
    name = "topk"

    def __init__(self, total_steps, frac0=1e-3, schedule="linear", floor=0.0):
        super().__init__(total_steps, schedule, floor)
        self.frac0 = frac0

    def flip_module(self, m, step, gen):
        g = m.weight_grad
        w = m.weight_t
        k = int(round(self.frac0 * w.numel() * self.decay(step)))
        if k <= 0:
            return 0
        direction = (-torch.sign(g)).to(torch.int8)
        mov = _movable(w, direction)
        score = torch.where(mov, g.abs(), torch.full_like(g, -1.0)).reshape(-1)
        k = min(k, int(mov.sum().item()))
        if k <= 0:
            return 0
        idx = torch.topk(score, k, sorted=False).indices
        flat_w = w.view(-1)
        flat_w[idx] = (flat_w[idx] + direction.view(-1)[idx]).clamp_(-1, 1)
        return k


class StochasticFlip(FlipRule):
    name = "stochastic"

    def __init__(self, total_steps, lr=1e-3, schedule="linear", floor=0.0):
        super().__init__(total_steps, schedule, floor)
        self.lr = lr

    def flip_module(self, m, step, gen):
        g = m.weight_grad
        w = m.weight_t
        lr = self.lr * self.decay(step)
        if lr <= 0:
            return 0
        a = g.abs()
        mean = a.mean()
        if mean <= 0:
            return 0
        p = (lr * a / mean).clamp_(max=1.0)
        direction = (-torch.sign(g)).to(torch.int8)
        r = torch.rand(g.shape, generator=gen, device=g.device)
        do = (r < p) & _movable(w, direction)
        w.add_(torch.where(do, direction, torch.zeros_like(direction))).clamp_(-1, 1)
        return int(do.sum().item())


class CounterFlip(FlipRule):
    """FALLBACK: keeps an int8 counter per weight (a hidden per-weight state)."""
    name = "counter"
    breaks_no_hidden_copy_rule = True

    def __init__(self, total_steps, threshold=8, schedule="linear", floor=0.0):
        super().__init__(total_steps, schedule, floor)
        self.threshold = threshold

    def flip_module(self, m, step, gen):
        if not hasattr(m, "flip_counter"):
            m.register_buffer("flip_counter", torch.zeros_like(m.weight_t))
        c = m.flip_counter
        direction = (-torch.sign(m.weight_grad)).to(torch.int8)
        c.add_(direction).clamp_(-127, 127)
        fire = c.abs() >= self.threshold
        fdir = torch.sign(c).to(torch.int8)
        do = fire & _movable(m.weight_t, fdir)
        m.weight_t.add_(torch.where(do, fdir, torch.zeros_like(fdir))).clamp_(-1, 1)
        c[fire] = 0
        return int(do.sum().item())


def make_flip_rule(name, total_steps, frac0=1e-3, lr=1e-3, threshold=8, schedule="linear", floor=0.0):
    if name == "none":
        return None
    if name == "topk":
        return TopKFlip(total_steps, frac0, schedule, floor)
    if name == "stochastic":
        return StochasticFlip(total_steps, lr, schedule, floor)
    if name == "counter":
        return CounterFlip(total_steps, threshold, schedule, floor)
    raise ValueError(name)
