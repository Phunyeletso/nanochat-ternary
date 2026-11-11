"""
Phase 4 experiment: incremental patching vs full rebuild when ternary weights flip during training.
Each "training step" changes a random fraction of weights by one step (like the flip rules), then:
  patch   : re-express only the rows that changed, re-using existing temporaries (no new ones)
  rebuild : build a fresh plan from scratch for the new matrix
We record time and resulting additions for both, and check the patched plan is exact.
Flips here are random positions (a stand-in for gradient-chosen ones).
    python -m ternary.phase4_patch
"""
import csv
import time

import numpy as np

from ternary.rowformula import build_plan, naive_adds, random_ternary, PatchablePlan

CASES = [  # (n_out, n_in, zero_frac, flip_frac_per_step, steps)
    (256, 256, 0.5, 0.0007, 20),
    (256, 256, 0.5, 0.01, 10),
    (256, 256, 0.5, 0.05, 5),
    (1024, 256, 0.5, 0.0007, 20),
    (1024, 256, 0.5, 0.01, 10),
    (1024, 256, 0.5, 0.05, 5),
]


def flip(W, frac, rng):
    W2 = W.copy()
    flat = W2.reshape(-1)
    n = max(1, int(round(frac * W.size)))
    idx = rng.choice(W.size, n, replace=False)
    step = rng.choice(np.array([-1, 1], dtype=np.int8), n)
    flat[idx] = np.clip(flat[idx] + step, -1, 1)
    return W2


out = open("runs/phase4_patch.csv", "w", newline="")
wr = csv.writer(out)
wr.writerow(["n_out", "n_in", "flip_frac", "step", "rows_changed", "patch_s", "patched_adds",
             "rebuild_s", "rebuilt_adds", "naive_adds", "exact"])
for (n_out, n_in, zf, ff, steps) in CASES:
    rng = np.random.default_rng(7)
    W = random_ternary(n_out, n_in, zf, rng)
    pp = PatchablePlan(build_plan(W))
    for s in range(1, steps + 1):
        W2 = flip(W, ff, rng)
        changed = np.nonzero((W2 != W).any(axis=1))[0].tolist()
        t0 = time.perf_counter(); pp.patch(W2, changed); tp = time.perf_counter() - t0
        W = W2
        X = rng.integers(-128, 128, size=(4, n_in))
        exact = bool(np.array_equal(pp.p.evaluate(X), X @ W.astype(np.int64).T))
        if s in (1, steps) or s % 5 == 0:
            t0 = time.perf_counter(); fresh = build_plan(W); tb = time.perf_counter() - t0
            r = [n_out, n_in, ff, s, len(changed), round(tp, 4), pp.p.adds(), round(tb, 2), fresh.adds(),
                 naive_adds(W), exact]
            wr.writerow(r); out.flush()
            print(r, flush=True)
        assert exact
