"""
Phase 4 experiment: additions needed by the row-formula plan vs naive add/subtract, for the model's
matrix shapes and several zero fractions. Every plan is checked for exactness before it is counted.
    python -m ternary.phase4_count --out runs/phase4_counts.csv
"""
import argparse
import csv
import time

import numpy as np

from ternary.rowformula import build_plan, naive_adds, random_ternary

SHAPES = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 256), (256, 1024)]
ZFS = [0.3, 0.5, 0.7]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/phase4_counts.csv")
args = ap.parse_args()

rows = []
with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_out", "n_in", "zero_frac", "nonzeros", "naive_adds", "plan_adds", "temps",
                "saving_pct", "adds_per_output_naive", "adds_per_output_plan", "build_s", "exact"])
    for (n_out, n_in) in SHAPES:
        for zf in ZFS:
            rng = np.random.default_rng(1000 * n_out + n_in + int(zf * 100))
            W = random_ternary(n_out, n_in, zf, rng)
            t0 = time.perf_counter()
            p = build_plan(W)
            bt = time.perf_counter() - t0
            X = rng.integers(-128, 128, size=(8, n_in))
            exact = bool(np.array_equal(p.evaluate(X), X @ W.astype(np.int64).T))
            n = naive_adds(W)
            r = [n_out, n_in, zf, int((W != 0).sum()), n, p.adds(), p.n_temps(),
                 round(100 * (1 - p.adds() / n), 2), round(n / n_out, 1), round(p.adds() / n_out, 1),
                 round(bt, 2), exact]
            w.writerow(r); f.flush()
            print(r, flush=True)
