"""Phase 4 charts from phase4_counts.csv, phase4_patch.csv and phase4_bench.csv."""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "runs/"

# 1) additions saved
rows = list(csv.DictReader(open(R + "phase4_counts.csv")))
fig, ax = plt.subplots(figsize=(7, 4))
by = defaultdict(list)
for r in rows:
    by[(int(r["n_out"]), int(r["n_in"]))].append((float(r["zero_frac"]), float(r["saving_pct"])))
for (o, i), v in by.items():
    v.sort()
    ax.plot([100 * z for z, _ in v], [s for _, s in v], marker="o", label=f"{o} x {i}" + ("  (model)" if (o, i) in [(256, 256), (1024, 256), (256, 1024)] else ""))
ax.set_xlabel("weights that are 0 (%)"); ax.set_ylabel("additions saved vs naive (%)")
ax.set_title("Row-formula plan: additions saved (random ternary matrices, all outputs exact)")
ax.grid(alpha=.3); ax.legend(fontsize=8, title="rows x inputs")
fig.tight_layout(); fig.savefig(R + "phase4_savings.png", dpi=130)

# 2) patch vs rebuild
rows = list(csv.DictReader(open(R + "phase4_patch.csv")))
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
grp = defaultdict(list)
for r in rows:
    grp[(r["n_out"], r["n_in"], r["flip_frac"])].append(r)
for k, v in grp.items():
    lab = f"{k[0]}x{k[1]}, {100*float(k[2]):g}% flips/step"
    st = [int(r["step"]) for r in v]
    a.plot(st, [100 * (int(r["patched_adds"]) / int(r["rebuilt_adds"]) - 1) for r in v], marker="o", label=lab)
    b.plot(st, [float(r["rebuild_s"]) / max(float(r["patch_s"]), 1e-6) for r in v], marker="o", label=lab)
a.set_title("Extra additions in patched plan vs fresh rebuild (%)"); a.set_xlabel("training steps since last rebuild")
b.set_title("How many times cheaper patching is than rebuilding"); b.set_xlabel("training steps since last rebuild"); b.set_yscale("log")
for ax in (a, b):
    ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(R + "phase4_patch.png", dpi=130)

# 3) timing
rows = list(csv.DictReader(open(R + "phase4_bench.csv")))
methods = ["numpy float32 matmul (BLAS)", "torch float32 matmul", "bitnet.cpp I2_S kernel (C++, AVX)",
           "naive add/sub (Numba)", "row-formula plan (Numba)"]
shapes = [("256", "256"), ("1024", "256"), ("256", "1024")]
Ns = ["1", "16", "256"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
for ax, N in zip(axes, Ns):
    x = np.arange(len(shapes)); w = 0.16
    for k, m in enumerate(methods):
        vals = [next(float(r["time_us"]) for r in rows if (r["n_out"], r["n_in"]) == s and r["N"] == N and r["method"] == m) for s in shapes]
        ax.bar(x + (k - 2) * w, vals, w, label=m)
    ax.set_xticks(x); ax.set_xticklabels([f"{o}x{i}" for o, i in shapes])
    ax.set_yscale("log"); ax.set_title(f"{N} token{'s' if N != '1' else ''} per call"); ax.set_ylabel("microseconds (log scale, lower is better)")
    ax.grid(alpha=.3, axis="y")
axes[0].legend(fontsize=7)
fig.suptitle("Time for y = W x, single CPU thread, 50% zeros (measured)")
fig.tight_layout(); fig.savefig(R + "phase4_timing.png", dpi=130)
print("ok")
