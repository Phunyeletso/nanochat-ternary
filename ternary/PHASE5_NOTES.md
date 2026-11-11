# Phase 5 (final): Improving direct-ternary training (C): flip schedule and width

Same data, steps (600 x 8,192 tokens), schedules and evaluation as Phase 3. Only C was changed or tuned;
A and B are exactly the Phase 3 models (B was NOT tuned in this phase, so this is not an equal-tuning comparison).

## 1. Flip-rate schedule (width 256, seed 0)
`--flip-schedule {linear,cosine,constant} --flip-floor F` (F = share of the starting flip rate kept at the end).
| Schedule | Final val bpb |
|---|---|
| linear to 0 (Phase 3) | 1.404 |
| constant | 1.410 |
| linear to 25% | 1.408 |
| cosine to 10% | **1.401** |
| (B, BitNet, for reference) | 1.370 |
All within ~0.01 of each other (seed-to-seed noise is ~0.002-0.007). **The schedule is not what holds C back**;
the Phase 3 hypothesis that "the flip rate decays to zero" caused the plateau was wrong.

## 2. Width (cosine to 10%)
| Model | Block-matrix weights | Val bpb seed 0 | Val bpb seed 1 | Train time (2 CPU threads) |
|---|---|---|---|---|
| A: full precision, 256 | 3.15M fp32 | 1.295 | 1.300 | 22 min |
| B: BitNet, 256 | 3.15M latent fp32 | 1.370 | 1.363 | 27 min |
| C: yours, 256 | 3.15M int8 ternary | 1.401 | - | 43 min* |
| C: yours, 384 | 7.08M int8 ternary | 1.390 | - | 42 min |
| C: yours, 512 | 12.58M int8 ternary | **1.368** | 1.375 | 60-62 min |
\* the 256 runs in this phase shared the CPU two at a time; the others ran alone.

**Width works: at 512 wide, C roughly matches BitNet.** Mean over 2 seeds: C 512 = 1.371, B 256 = 1.366
(C is 0.005 behind, about the size of seed-to-seed noise). C 512 is ahead of BitNet for most of training
(e.g. 1.47 vs 1.56 at step 250 on seed 0). "Matches" is fair; "beats" is not.

## 3. What it costs (be careful with "equal memory")
Memory tied to the block-matrix weights (computed from tensor sizes):
| Model | Stored between steps | Peak during a step (+ gradient) |
|---|---|---|
| B 256 (latent fp32 + Muon momentum) | 25.2 MB (8 B/weight) | 37.7 MB (12 B/weight) |
| C 384 | 7.1 MB (1 B/weight) | 35.4 MB (5 B/weight) |
| C 512 | 12.6 MB (1 B/weight) | 62.9 MB (5 B/weight) |

Whole-process peak RAM, measured (4 training steps, batch 32 x 256 tokens):
| | A 256 | B 256 | C 256 | C 384 | C 512 |
|---|---|---|---|---|---|
| Before memory fix | 2.6 GB | 2.8 GB | 3.1 GB | 4.1 GB | 5.0 GB |
| After memory fix | (unchanged) | 2.8 GB | **2.4 GB** | **3.0 GB** | **3.5 GB** |

Memory fix (commit "Memory fix"): TernaryLinear now keeps only the int8 activations (1 byte per value) plus one
float per token between forward and backward, instead of float32 copies of its input and output; the scale's
gradient is computed from the weight gradient. Training is unchanged (losses identical to 4 decimals for the
first 20 steps, max difference 0.001 over 30 steps from float rounding order). All width results above were
trained before the fix.

- At this tiny model size, activations (values kept for backprop) dominate RAM, not weights. After the fix,
  C 256 uses less total RAM than B 256 and C 384 about the same, but C 512 still uses 0.7 GB more than B 256.
  The weight-memory advantage (1 vs 8 bytes per weight) would dominate in much larger models.
- C 512 does ~4x the block-matrix compute per token of B 256, so it trained ~2.3x slower here.
- At inference, both B and C use ternary weights, so a C 512 model is 4x bigger and slower than a B 256 model
  of the same quality. The advantage of C is during training only (no fp32 latent copy, no optimizer state).

## 4. Text generation
`ternary/train.py` now saves models (`--save 1`, default) and `ternary/generate.py` writes text from them.
Reloaded models give the same val bpb as at the end of training (checked for A, B and C).
Samples (same prompts, temperature 0.8, top-k 50, seed 42) are in `results/phase5_samples.txt`. All three
produce English-looking text with the style of the source books but little meaning beyond a few words,
as expected for ~7-15M-parameter models trained for under an hour on 3M words.

## Bugs / mistakes during this phase (fixed)
- The first job launcher dropped an argument, so 3 runs failed instantly (relaunched).
- A 512-wide run was killed for running out of memory because test jobs ran at the same time (rerun alone).
- Earlier phases did not save models, so no text could be generated from them (saving now built in).
