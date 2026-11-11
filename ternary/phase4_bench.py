"""
Phase 4 timing: row-formula plan vs naive add/subtract vs plain float matmul, single thread.
Run with:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m ternary.phase4_bench
All methods compute y = W x for N token vectors; results are checked against each other first.
"""
import csv
import os
import time

import re
import subprocess

import numpy as np
import torch

from ternary.rowformula import build_plan, naive_adds, random_ternary
from ternary.rowformula_fast import FastPlanMatmul, FastNaiveMatmul

torch.set_num_threads(1)
BITNET_BIN = "/home/claude/bench_bitnet/test_gemm_kernel"   # bitnet.cpp's own I2_S GEMM benchmark (ggml_gemm_i2_i8_s)


def bitnet_cpp_us(n_in, N, n_out):
    """bitnet.cpp kernel: 2-bit packed ternary weights x int8 activations -> float, single thread.
    Its time does not depend on the zero fraction (zeros are stored and multiplied like any value)."""
    if not os.path.exists(BITNET_BIN):
        return None
    iters = max(50, int(2e8 / (n_in * N * n_out)))
    res = subprocess.run([BITNET_BIN, "-n", str(n_in), "-r", str(N), "-c", str(n_out), "-i", str(iters), "-w", "20"],
                         capture_output=True, text=True, timeout=600)
    m = re.search(r"Average us\s*:\s*([0-9.]+)", res.stdout)
    return float(m.group(1)) if m else None
SHAPES = [(256, 256), (1024, 256), (256, 1024)]
NS = [1, 16, 256]
ZF = 0.5


def bench(fn, min_time=0.3, repeats=7):
    fn()                                           # warm-up (also triggers JIT compile)
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        dt = time.perf_counter() - t0
        if dt > min_time / repeats:
            break
        n *= 2
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        times.append((time.perf_counter() - t0) / n)
    return float(np.median(times))


out = open("/home/claude/runs/phase4_bench.csv", "w", newline="")
wr = csv.writer(out)
wr.writerow(["n_out", "n_in", "N", "method", "time_us", "adds_or_madds"])
print(f"threads: torch={torch.get_num_threads()} OMP={os.environ.get('OMP_NUM_THREADS')}")
for (n_out, n_in) in SHAPES:
    rng = np.random.default_rng(11)
    W = random_ternary(n_out, n_in, ZF, rng)
    t0 = time.perf_counter(); plan = build_plan(W); bt = time.perf_counter() - t0
    print(f"{n_out}x{n_in}: plan built in {bt:.1f}s, naive adds {naive_adds(W)}, plan adds {plan.adds()}", flush=True)
    Wf = W.astype(np.float32)
    Wt = torch.from_numpy(Wf)
    naive = FastNaiveMatmul(W)
    for N in NS:
        X = rng.integers(-128, 128, size=(N, n_in)).astype(np.int32)
        X_T = np.ascontiguousarray(X.T)
        Xf = X.astype(np.float32)
        Xt = torch.from_numpy(Xf)
        fast = FastPlanMatmul(plan, N)
        o1 = np.zeros((n_out, N), dtype=np.int32); o2 = np.zeros((n_out, N), dtype=np.int32)
        ref = Xf @ Wf.T
        assert np.array_equal(naive(X_T, o1).T, ref.astype(np.int32))
        assert np.array_equal(fast(X_T, o2).T, ref.astype(np.int32))
        methods = {
            "numpy float32 matmul (BLAS)": (lambda: Xf @ Wf.T, n_out * n_in * N),
            "torch float32 matmul": (lambda: Xt @ Wt.T, n_out * n_in * N),
            "naive add/sub (Numba)": (lambda: naive(X_T, o1), naive_adds(W) * N),
            "row-formula plan (Numba)": (lambda: fast(X_T, o2), plan.adds() * N),
        }
        for name, (fn, ops) in methods.items():
            t = bench(fn)
            r = [n_out, n_in, N, name, round(t * 1e6, 2), ops]
            wr.writerow(r); out.flush()
            print(r, flush=True)
        tb = bitnet_cpp_us(n_in, N, n_out)
        if tb is not None:
            r = [n_out, n_in, N, "bitnet.cpp I2_S kernel (C++, AVX)", round(tb, 2), n_out * n_in * N]
            wr.writerow(r); out.flush()
            print(r, flush=True)
