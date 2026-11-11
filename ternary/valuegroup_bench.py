"""
Counts and single-thread timing for the value-grouped integer matmul.
Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m ternary.valuegroup_bench
"""
import csv

import numpy as np
import torch

from ternary.valuegroup import quantized_weights, naive_counts, grouped_counts, GroupedMatmul, NaiveIntMatmul
from ternary.phase4_bench import bench

torch.set_num_threads(1)
SHAPES = [(256, 256), (1024, 256), (256, 1024)]
BITS = [4, 8]
NS = [1, 16, 256]
R = "/home/claude/runs/"

cw = csv.writer(open(R + "valuegroup_counts.csv", "w", newline=""))
cw.writerow(["n_out", "n_in", "bits", "naive_mults", "grouped_mults", "mults_saved_pct", "adds_both",
             "distinct_values_per_row_avg"])
tw = csv.writer(open(R + "valuegroup_bench.csv", "w", newline=""))
tw.writerow(["n_out", "n_in", "bits", "N", "method", "time_us"])
for (n_out, n_in) in SHAPES:
    for bits in BITS:
        rng = np.random.default_rng(n_out + n_in + bits)
        W = quantized_weights(n_out, n_in, bits, rng)
        nm, na = naive_counts(W)
        gm, ga = grouped_counts(W)
        assert na == ga
        dv = np.mean([len(np.unique(W[r][W[r] != 0])) for r in range(n_out)])
        row = [n_out, n_in, bits, nm, gm, round(100 * (1 - gm / nm), 1), na, round(dv, 1)]
        cw.writerow(row); print("counts", row, flush=True)
        Wf = W.astype(np.float32); Wt = torch.from_numpy(Wf)
        naive = NaiveIntMatmul(W)
        for N in NS:
            X = rng.integers(-128, 128, size=(N, n_in)).astype(np.int32)
            X_T = np.ascontiguousarray(X.T); Xf = X.astype(np.float32); Xt = torch.from_numpy(Xf)
            grouped = GroupedMatmul(W, N)
            o1 = np.zeros((n_out, N), dtype=np.int32); o2 = np.zeros((n_out, N), dtype=np.int32)
            ref = X.astype(np.int64) @ W.astype(np.int64).T
            assert np.array_equal(naive(X_T, o1).T, ref) and np.array_equal(grouped(X_T, o2).T, ref)
            for name, fn in [("numpy float32 matmul (BLAS)", lambda: Xf @ Wf.T),
                             ("torch float32 matmul", lambda: Xt @ Wt.T),
                             ("naive int multiply-add (Numba)", lambda: naive(X_T, o1)),
                             ("grouped by value (Numba)", lambda: grouped(X_T, o2))]:
                t = bench(fn)
                r = [n_out, n_in, bits, N, name, round(t * 1e6, 2)]
                tw.writerow(r); print(r, flush=True)
