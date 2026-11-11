# Phase 6: Everything ternary

Question: what happens if the token table, value tables, output layer and the two tiny gates are also
stored as -1/0/+1 and trained only by flips, like the six layer matrices?

Setup: same data, 600 steps x 8,192 tokens, seed 0, stochastic flip rule, cosine flip schedule to 10%.
New flags: `--ternary-lm-head`, `--ternary-gates`, `--ternary-all` (in `ternary/model.py` and `ternary/train.py`).
With `--ternary-all` only **41 decimal numbers** remain in the whole model (one scale per matrix/table and the
per-layer mixing scalars).

## Results (final validation bits per byte, lower is better)
| Model | Ternary weights | Val bpb | Train time |
|---|---|---|---|
| A: full precision, 256 | 0 | 1.295 | 22 min |
| B: BitNet, 256 (layers ternary, hidden weights) | 3.15M | 1.370 | 27 min |
| C: layers ternary, 256 (Phase 5) | 3.15M | 1.401 | ~20 min alone |
| C: layers ternary, 512 (Phase 5) | 12.58M | 1.368 | 60 min |
| + output layer ternary, 256 | 4.19M | 1.628 | 27 min |
| + token and value tables ternary, 256 | 6.29M | 1.695 | 28 min |
| **everything ternary, 256** | 7.34M | **2.038** | 30 min |
| **everything ternary, 512** | 20.97M | **1.938** | 62 min |

## Reading it
- Making the output layer ternary costs +0.23; making the tables ternary costs +0.29; doing both costs +0.64.
  The costs add up (and a bit more). Everything-ternary at 256 is even worse than Phase 1's model whose layer
  weights could not learn at all (1.59).
- Doubling the width helps only a little here (2.038 -> 1.938, vs 1.401 -> 1.368 when only the layers are ternary).
- Text samples (`results/phase6_samples.txt`) show the drop: fully ternary models repeat fragments
  ("the LORD", "the earth") and produce more invented words.
- Likely reason: the token table must give 4,096 tokens distinguishable meanings, and the output layer must
  tell 4,096 words apart, with only three values per number plus one scale for the whole table. That is very
  little room. This matches why BitNet keeps these parts in normal precision. Our flip rule also uses one
  normalisation per table, so rarely-seen tokens' rows get flips decided from very little evidence.

## Size after training (the one place this helps)
Packed at 1.58 bits per ternary weight, decimals at 2 bytes (bf16):
| Model | Size | Quality |
|---|---|---|
| B: BitNet 256 | ~0.6 MB ternary + ~8.4 MB decimal tables/head = **~9 MB** | 1.370 |
| everything ternary, 256 | **~1.5 MB** | 2.038 |
| everything ternary, 512 | **~4.1 MB** | 1.938 |
At this tiny size the decimal tables are most of BitNet's file, so going fully ternary shrinks the model 2-6x,
but the quality loss is far too large to be worth it. In big models the tables are a small share anyway.

## Verdict
Keep the token table, value tables and output layer in normal precision (as BitNet does). Ternary layer matrices
trained by flips (Phase 5) remain the useful result. If fully ternary is revisited, ideas: per-row scales for the
tables (one scale per token instead of one per table), or 2-bit/4-bit tables instead of ternary.

## Infrastructure notes
- The cloud workspace is recycled a few minutes after activity stops unless a tracked background task is running;
  the width-512 run was cut off three times (steps 228, 42, 30). `ternary/train.py` now has `--ckpt-every N
  --resume 1` (atomic checkpoint of model, optimizer, RNG and data-loader position). Resuming continues training
  correctly but not bit-for-bit: nanochat's data loader resumes at the nearest row group, so a few documents come
  in a slightly different order (checked: losses after resume differ by ~0.001).
- The final width-512 run (started 21:27, with a tracked watcher keeping the workspace alive) ran from step 0 to 600
  without interruption, so no resume was needed. Its log file also contains the lines of the two earlier attempts
  that day (steps 0-31), above the last "(re)start" marker; the .jsonl holds only the final run.
