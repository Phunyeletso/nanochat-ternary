# Phase 3: Baselines (A vs B vs C)

## Setup (identical for all three)
Same script (`ternary/train.py`, no torch.compile for any run), same model size (4 layers, width 256, vocab 4096),
same stand-in data and data order, 600 steps x 8,192 tokens, same LR/momentum/weight-decay schedules, 2 seeds
(seed changes the weight initialisation; and C's flip randomness). Seed 0 of A reproduces the earlier
full-precision runs exactly.
- **A**: original nanochat, full precision, nanochat's optimizer (Muon + AdamW), unmodified hyperparameters.
- **B**: standard BitNet b1.58 (`ternary/bitnet.py`): latent fp32 weights, absmean ternarisation + 8-bit absmax
  activations in every forward pass, straight-through estimator. Latent weights trained by nanochat's own
  optimizer with A's hyperparameters. Same six matrices as C.
- **C**: direct ternary, stochastic flip, exactly as in Phase 2 (no further tuning).
Nothing was tuned for any of the three beyond the defaults already in place.

## Results (final validation bits per byte, lower is better)
| Model | Seed 0 | Seed 1 | Mean |
|---|---|---|---|
| A: full precision | 1.2951 | 1.3004 | **1.2978** |
| B: BitNet b1.58 (latent + STE) | 1.3697 | 1.3629 | **1.3663** |
| C: direct ternary, stochastic flip | 1.4040 | 1.4059 | **1.4049** |

Seed-to-seed spread is 0.002-0.007 bpb, so the gaps below are much larger than the noise.

## Plain verdict
- **C is worse than standard BitNet (B)**: +0.039 bpb at the end (1.405 vs 1.366).
  B gives up 0.069 bpb versus full precision; C gives up 0.107.
- C is ahead of B early: better than B until about step 300 (e.g. 1.55 vs 1.63 at step 200),
  then B overtakes and C flattens out.
- C is genuinely training (it is far better than the frozen floor of 1.59), and it does so with no
  hidden float copy: during training C keeps 3.1 MB of int8 weights for these matrices, versus B's
  12.6 MB of fp32 latent weights plus Muon's optimizer state (computed from tensor sizes, not measured RAM).
- No speed difference in training time (18-22 min for all runs on 2 CPU cores); no speed claim is made.

## Likely reasons C falls behind (hypotheses, not yet tested)
1. **The flip budget decays to zero.** C's flip rate falls linearly to 0, so in the last third of training it
   can barely change anything; the crossover with B (~step 300) is where C's flip rate has halved.
   B keeps making small latent updates until the end.
2. **No memory across steps.** B's latent weights (and Muon's momentum) add up many small, consistent
   gradient signals before a weight crosses a rounding threshold. C decides each flip from one batch's
   gradient, so noisy gradients cause flips that later have to be undone, and weak-but-consistent signals
   are never acted on. That is exactly what latent weights buy you; the counter fallback is a cheap form of it.
3. **Coarse moves.** A flip changes a weight by a full step (0 -> +/-1) times the matrix scale, a much
   bigger change than a latent update; this suits early training (fast progress) but not fine-tuning late.
4. **Untuned settings.** The 0.1% start rate and linear decay were first guesses; by the "don't tune C more
   than A and B" rule they were left as is.

## Ideas worth testing later (would need your OK, since they change C's design)
- A slower or non-zero-floor flip-rate schedule (e.g. cosine to 10% of the start rate).
- Averaging the gradient over more tokens before deciding flips (bigger batch for the flip decision only).
- A small amount of memory: e.g. the int8 counter fallback, or a 2-bit "pending flip" state per weight;
  this trades part of the no-hidden-copy goal for quality, so it is your call.
