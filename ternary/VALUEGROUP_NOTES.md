# Group-by-value integer matmul (your "T" / row-minimum idea, generalised)

## The idea
Your method: y = min(row) * T(column) + (row - min) . column. Correct (checked on your 3x3 example, all 9
entries), but it removes only one term per row while adding one, so the work stays the same.
Generalised: keep peeling every level of the row, which ends at grouping the row by value:

    y_r = sum over distinct values v in row r of  v * (sum of x_j where W[r,j] == v)

One multiply per DISTINCT value instead of one per weight (+1/-1 need none). Additions do not change.

Files: `ternary/valuegroup.py` (reference + Numba kernels), `ternary/valuegroup_bench.py`,
`ternary/tests/test_valuegroup.py` (11 tests incl. your 3x3 example; all outputs exact).
Weights: Gaussian, per-row absmax quantised to int4 / int8 (how real LLM weights look).

## Counts (exact, per matrix)
| Shape | Bits | Naive multiplies | Grouped multiplies | Saved | Additions (both) | Distinct values per row |
|---|---|---|---|---|---|---|
| 256x256 | int4 | 33,878 | 2,743 | **91.9%** | 53,978 | 12.7 |
| 1024x256 | int4 | 135,040 | 10,972 | **91.9%** | 215,975 | 12.7 |
| 256x1024 | int4 | 120,917 | 2,854 | **97.6%** | 210,983 | 13.1 |
| 256x256 | int8 | 63,612 | 32,482 | 48.9% | 64,639 | 128.7 |
| 1024x256 | int8 | 254,538 | 129,542 | 49.1% | 258,527 | 128.3 |
| 256x1024 | int8 | 253,592 | 44,656 | 82.4% | 258,956 | 176.4 |

## Time (microseconds, single thread, measured; see results/valuegroup_bench.csv)
| Shape | Bits | Tokens | float matmul (BLAS) | naive int multiply-add | grouped by value |
|---|---|---|---|---|---|
| 256x256 | int4 | 1 | 4.1 | 156.2 | 156.1 |
| 256x1024 | int4 | 16 | 205.1 | 963.8 | 784.3 |
| 256x1024 | int4 | 256 | 1427.1 | 6127.3 | 4463.2 |
| 1024x256 | int4 | 1 | 16.0 | 526.9 | 716.2 |
| 256x256 | int8 | 1 | 5.7 | 156.2 | 516.1 |
| 1024x256 | int8 | 256 | 1439.9 | 6661.4 | 9279.0 |

## Verdict
- Multiplies drop a lot for int4 (92-98%) and moderately for int8 (49-82%).
- Speed: for int4 the grouped kernel ranges from 1.4x slower (1024x256, 1 token) to 1.4x faster (256x1024,
  256 tokens) than the same-style naive kernel; for int8 it is up to 3.7x slower. It is 3-130x slower than ordinary float matrix multiply everywhere (3x at best: int4, 256 tokens; 130x at worst: int8, 1 token).
- Why: on a CPU a multiply costs about the same as an add (both are one SIMD instruction for 8-16 numbers),
  so removing multiplies saves little; the additions (unchanged) and the scattered memory reads dominate,
  and int8 rows have so many distinct values that the per-group bookkeeping costs more than it saves.
- Where it would pay off: hardware where multiplies are genuinely expensive (no hardware multiplier, FPGAs,
  custom chips, some microcontrollers), and very low-bit weights (int2-int4).
