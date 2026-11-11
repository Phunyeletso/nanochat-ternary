# Phase 4: Row-formula matmul

All numbers below are measured on this machine (cloud container, Intel Xeon @ 2.1 GHz with AVX-512, single
thread for every timing). Matrices are random ternary with the stated zero fraction (stand-ins for trained
ones; C's trained matrices were ~48% zeros).

## What was built
- `ternary/rowformula.py` (reference, plain Python + NumPy)
  - `build_plan(W)`: greedy sharing. Repeatedly find the signed pair (x_a + x_b or x_a - x_b) that appears
    in the most rows, compute it once as a temporary, substitute it in every row that contains it.
    Temporaries can be paired again, so formulas nest. Stops when no pair appears in 2+ rows.
  - `Plan.evaluate` (NumPy, adds/subs only) and `Plan.evaluate_counting` (pure Python, counts every op).
  - `PatchablePlan.patch(W_new, changed_rows)`: after flips, rebuild only the changed rows' formulas,
    re-using existing temporaries (none created); temporaries nothing uses any more are dropped.
- `ternary/rowformula_fast.py`: compiled (Numba) kernels for the plan and for naive add/subtract, same
  layout and loop style for both.
- Experiments: `phase4_count.py`, `phase4_patch.py`, `phase4_bench.py`; charts: `plot_phase4.py`.
- Tests: `ternary/tests/test_phase4_rowformula.py` (24 tests). Plans give exactly the same integers as
  `X @ W.T` on random int8-range inputs (sizes 1x5 to 96x64, 0-90% zeros), after patching, and in the fast
  kernels; the counted operations equal the cost model. All 49 tests in `ternary/tests` pass.

Cost model (same for naive and plan): a temporary = 1 add/sub; a row with k terms = k-1 add/subs, +1
negation if all its terms are negative.

## 1. Additions saved (all 18 plans verified exact)
| Shape (rows x inputs) | 30% zeros | 50% zeros | 70% zeros | Build time at 50% (pure Python) |
|---|---|---|---|---|
| 64 x 64 | 58.3% | 49.3% | 39.6% | 0.04 s |
| 128 x 128 | 62.5% | 54.8% | 45.0% | 0.5 s |
| **256 x 256** (attention Q/K/V/out) | 66.3% | 59.3% | 50.2% | 5 s |
| 512 x 512 | 69.4% | 63.0% | 54.8% | 60 s |
| **1024 x 256** (MLP up) | 69.5% | 63.2% | 55.0% | 21 s |
| **256 x 1024** (MLP down) | 69.7% | 63.4% | 55.4% | 173 s |
Example: 256 x 256 at 50% zeros needs 127 add/subs per output naively, 52 with the plan.
Savings grow with matrix size and shrink as zeros increase (fewer terms leave less to share).

## 2. Patching vs rebuilding when weights flip (random flip positions; all patched plans exact)
| Shape | Flips per step | Patch time | Rebuild time | Extra additions vs fresh plan after N steps |
|---|---|---|---|---|
| 256 x 256 | 0.07% (C's training rate) | 0.02-0.04 s | 4.8-5.5 s | +0.4% (1), +2.5% (10), +5.1% (20) |
| 1024 x 256 | 0.07% | 0.26-0.34 s | 20.6-21.4 s | +0.5% (1), +3.2% (10), +6.2% (20) |
| 256 x 256 | 1% | 0.14-0.20 s | 5.2-6.1 s | +4.1% (1), +26.0% (10) |
| 1024 x 256 | 5% | 2.3-2.4 s | 22.7-24.4 s | +17.5% (1), +37.1% (5) |
Patching is 60-250x cheaper at the training flip rate, but plans slowly get worse because patching never
creates new temporaries. A sensible scheme: patch every step, rebuild every ~10-20 steps (or when the
addition count drifts more than a few %). Note the pure-Python patch (~0.3 s for one 1024x256 matrix) is
still comparable to a whole training step (~2 s), so it would need a compiled version to be practical.

## 3. Speed (single thread, 50% zeros, median of 7 timed repeats)
Microseconds per call, y = W x for N token vectors:

| Shape | N | numpy fp32 (BLAS) | torch fp32 | bitnet.cpp I2_S | naive add/sub (Numba) | row-formula plan (Numba) |
|---|---|---|---|---|---|---|
| 256x256 | 1 | 4.3 | 7.0 | **2.0** | 88.9 | 47.7 |
| 1024x256 | 1 | 13.8 | 17.1 | **10.6** | 332.4 | 201.6 |
| 256x1024 | 1 | 12.5 | 23.6 | **4.8** | 346.1 | 213.0 |
| 256x256 | 16 | 53.3 | 28.8 | **21.8** | 132.5 | 63.7 |
| 1024x256 | 16 | 188.9 | 104.8 | **86.0** | 497.5 | 271.9 |
| 256x1024 | 16 | 194.7 | 124.8 | **108.8** | 489.2 | 292.1 |
| 256x256 | 256 | 308.4 | **263.4** | 344.2 | 695.8 | 876.2 |
| 1024x256 | 256 | 1185.5 | **1134.0** | 1607.0 | 2714.0 | 4000.1 |
| 256x1024 | 256 | 1311.3 | 1361.8 | **1123.7** | 4205.2 | 4310.4 |

bitnet.cpp: Microsoft's own standalone benchmark for its default x86 kernel (`ggml_gemm_i2_i8_s`, 2-bit packed
weights x int8 activations), built from github.com/microsoft/BitNet at the same sizes, single thread. Only the
print statement was changed (to show microseconds). Its time does not depend on the zero fraction. Its TL2
lookup-table kernels need per-shape code generation and were not measured.

## Plain verdict
- **The idea works on paper**: 50-63% fewer additions on the model's matrix shapes at realistic zero fractions,
  with exactly identical outputs.
- **Fewer additions became real speed only against the naive add/subtract method**: the compiled plan is
  1.6-2.1x faster than naive add/subtract for 1-16 tokens.
- **It is NOT faster than ordinary matrix multiplication or bitnet.cpp.** For one token it is 11-16x slower
  than numpy's float matmul and 23-44x slower than bitnet.cpp; for 16 tokens 2.2-3.2x slower than the best.
  For 256 tokens the plan is even slower than naive add/subtract.
- Why: on modern CPUs the count of additions is not the bottleneck. BLAS and bitnet.cpp do 16-64
  multiply-adds per instruction (SIMD) over data laid out contiguously. The plan's formulas jump around
  memory (each temporary reads two arbitrary earlier values), which SIMD cannot batch for a single token; and
  for 256 tokens the temporaries (~12,000 x 256 values for the MLP matrices) no longer fit in cache.
  My Numba kernels are a first attempt, not hand-tuned C, but the gap to bitnet.cpp is too large to blame on that alone.
- Planning is slow in pure Python (5 s to 3 min per matrix), so planning must happen offline or be compiled.

## What might close the gap (untested ideas)
- Restrict sharing to fixed small groups of neighbouring inputs (e.g. every pair/triple of inputs gets all
  its +/- combinations precomputed once per token). That keeps memory access regular and SIMD-friendly; it
  is close to what bitnet.cpp's TL1/TL2 "lookup table" kernels and T-MAC already do, so it would be worth
  measuring those kernels first.
- Block the token dimension so temporaries stay in cache for large N.
- Where the plan could matter more: hardware without fast multipliers or SIMD (microcontrollers, FPGAs,
  custom chips), where an addition really is the unit of cost.
