# Phase 7: How coarse can the data entering ternary layers be? (balanced-ternary "coin tables")

Idea: store the data (activations) entering each ternary layer as k stacked balanced-ternary tables
(coins 1, 3, 9, 27, ...), so that every multiply becomes ternary x ternary. k tables hold whole numbers in
+-(3^k-1)/2. For model quality this is exactly the same as rounding the data to that range, so we test the rounding.

New: `--act-tables k` (0 = 8-bit as before) in `ternary/train.py`; `act_quant_int8(x, qmax=...)`,
`balanced_ternary_planes`, `ternary_planes_matmul` in `ternary/layers.py`. Tests (`tests/test_phase7_tables.py`)
prove the stacked-table multiply gives exactly the same integers as a normal multiply for k = 1..4.

Setup: Phase 5 model C (six layer matrices ternary, trained by stochastic flips, cosine flip schedule to 10%),
width 256, seed 0, 600 steps. Only the rounding of the data entering the ternary layers changes.

## Results (final validation bits per byte, lower is better)
| Data entering ternary layers | Levels | Val bpb | vs 8-bit |
|---|---|---|---|
| 8-bit (-127..127), reference | 255 | 1.403 | - |
| 4 tables (-40..40) | 81 | 1.402 | same |
| 3 tables (-13..13) | 27 | 1.403 | same |
| 2 tables (-4..4) | 9 | 1.416 | +0.013 |
| 1 table (-1, 0, +1), pure ternary data | 3 | 1.519 | +0.117 |
(BitNet with 8-bit data: 1.370; Phase 1 frozen-weights floor: 1.59.) One seed each; seed-to-seed noise was ~0.002-0.007.

## Reading it
- The data needs far less precision than 8 bits: 27 levels (3 tables) is as good as 255 levels, and 9 levels
  costs only a little. This is a real, clean finding.
- Pure ternary data (3 levels) costs a lot: +0.117, more than the whole gap between our model and BitNet.
  The model still learns (well below the 1.59 frozen floor), but clearly worse. Text samples
  (`results/phase7_samples.txt`) show more invented words.
- The reference run reproduces Phase 5 (1.403 vs 1.401), so the code changes since then did not affect results.

## What it means for speed (estimates from spec sheets, not measured)
On a GPU with a 1-bit tensor mode (A100: binary 4,992 TOPS vs INT8 624 TOPS dense, 8x), one ternary x ternary
multiply takes 4 one-bit passes (plus/plus, minus/minus, plus/minus, minus/plus), so ~2x INT8.
Data with k tables times ternary weights takes 4k passes:
| Data | Quality | 1-bit passes | vs INT8 on A100 (paper) |
|---|---|---|---|
| 1 table | -0.117 bpb | 4 | ~2x faster |
| 2 tables | -0.013 bpb | 8 | about equal |
| 3 tables | no loss | 12 | ~1.5x slower |
| 8-bit data (plain INT8 tensor cores) | no loss | - | baseline |
So on today's GPUs there is no free lunch: the setting that is faster than 8-bit (1 table) costs quality, and the
settings that keep quality are no faster than plain 8-bit. 2 tables is the interesting middle (small quality loss,
about INT8 speed, and 4x less activation memory than 8-bit if stored packed).
On custom hardware where ternary x ternary is nearly free, 3 tables (27 levels) is a no-loss option.

## Ideas worth testing next
- Pure ternary data with per-group scales (e.g. one scale per 32 numbers instead of one per token) to recover quality.
- 2 tables everywhere plus a slightly wider model, to see whether the small loss can be bought back cheaply.
- Ternary or low-bit gradients for the weight-gradient step (the ~31% of training no current trick speeds up).
