# Phase 2: Training ternary weights without latent weights

## What was built
- `ternary/flip.py`, selected with `--flip-rule` in `ternary/train.py`:
  - `topk`: per matrix, the k movable weights with the largest |dL/dW| move one step toward
    -sign(gradient). k = 0.1% of the matrix at step 0, decaying linearly to 0 at the last step.
    ("Movable" = not already at +1 wanting to go up / -1 wanting to go down, so k counts real changes.)
  - `stochastic`: each movable weight moves one step toward -sign(gradient) with probability
    min(1, lr * decay * |g| / mean|g| of its matrix), lr = 0.001, same linear decay.
  - `counter` (fallback; keeps an int8 counter per weight, which breaks the no-hidden-copy rule):
    implemented and tested but **not used**, because both other rules worked.
- After each step the gradient buffers are discarded; only the int8 weights and one scale per matrix persist.
- Every step logs loss, fraction of ternary weights changed, and fraction that are zero
  (`results/p2_*.log` and `.jsonl`).
- Tests: `ternary/tests/test_phase2_flip.py` (7 tests; 22 in total, all passing). They include a toy problem
  where outputs come from a hidden ternary matrix: starting from all zeros and training only by flips,
  both rules recover 100% of the hidden matrix's entries.

## Results (600 steps, same data, model, schedule and evaluation as Phases 0-1; one seed each)
| Model | Val bpb @600 | Share of the frozen-to-full-precision gap closed | Total flips (sum of per-step fractions) | Zeros at end | Train time |
|---|---|---|---|---|---|
| A: original full precision | **1.294** | (100%) | n/a | n/a | 16.0 min* |
| Ternary, frozen (Phase 1 floor) | 1.593 | 0% | 0 | 56.3% | 18.6 min |
| Ternary, top-k flip | 1.500 | 31% | 30.0% | 54.0% | 19.6 min |
| Ternary, stochastic flip | **1.404** | 63% | 22.6% | 48.2% | 20.9 min |
\* A was run with torch.compile; the ternary runs were not.

## Reading the results honestly
- Both rules genuinely train the ternary weights: both beat the frozen floor and keep improving to the end.
- Stochastic is clearly better here (1.404 vs 1.500) even though it changed fewer weights in total. A likely
  reason: top-k forces every matrix to change the same fraction of weights each step, while stochastic
  spends its changes where gradients are relatively large and lets each matrix move at its own pace.
- The best ternary model is still worse than full precision: 1.404 vs 1.294 bpb (+0.11, about 8.5%).
  That gap is expected for ternary weights at this size; the important comparison is against standard
  BitNet (latent weights + straight-through estimator) in Phase 3, which has not been run yet.
- Not tuned: flip rate 0.1%, linear decay to zero and the stochastic normalisation are the first settings
  tried. One seed per run, so differences of a few hundredths of a bpb could be noise.
- The flip budget decays to zero by design, so the curves flatten at the end partly because flipping stops.
- Speed: no claim. The ternary forward still uses a float matmul on {-1,0,+1}; Phase 4 deals with speed.
