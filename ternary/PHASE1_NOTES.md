# Phase 1: Ternary layers

## What was built (all new files; nanochat code untouched)
- `ternary/layers.py`
  - `TernaryLinear`: stores ONLY an int8 matrix `weight_t` with values in {-1, 0, +1} plus ONE float
    `scale` per matrix. No float weight matrix exists anywhere (checked by a test on `state_dict`).
  - Activations entering the layer are quantised to 8 bits, absmax, per token (BitNet b1.58 style).
  - Forward = for each output row, add inputs where weight is +1, subtract where -1, skip where 0, then
    multiply by `scale` (and by the activation de-quantisation factor). `ternary_matmul_addsub_reference`
    does this literally with only additions/subtractions; the fast path (a float matmul on the
    {-1,0,+1} values) gives **bit-identical integer accumulators** (tested at 0%, 50%, 90% zeros).
  - Backward produces dL/dW for every ternary weight into a temporary `weight_grad` buffer (for Phase 2
    to read, then discard), plus normal gradients for the input and the scale. Activation rounding uses
    the straight-through estimator.
  - `TernaryEmbedding`: same idea for the token table (int8 table + one scale).
  - Initial ternary values come from nanochat's own float init via BitNet's absmean rounding (used once,
    at init). nanochat's two zero-initialised projections stay all-zero, as agreed.
- `ternary/model.py`: `convert_to_ternary(model, TernaryConfig(...))` with flags
  `linear` (the six block matrices, default on), `embed` (token table, default off),
  `value_embeds` (default off), `quant_act` (default on). lm_head, gates, scalars stay full precision.
  `setup_optimizer_ternary` keeps all of nanochat's optimizer groups; the per-matrix scales get their own
  AdamW group; nothing ever updates the int8 weights (Phase 2's job).
- `ternary/train.py`: lean single-process copy of base_train's logic (same data, LR/momentum/weight-decay
  schedules, batch-size scaling, val bpb) with the ternary flags. Logs loss, flip fraction and zero
  fraction every step to the log and a `.jsonl` file.
- `ternary/tests/test_phase1_layers.py`: 15 tests, all passing.

## Checks
- With all flags off, `ternary/train.py` reproduces the original `base_train.py` loss to ~6 decimals for the
  first 16 steps; afterwards tiny float differences (original uses torch.compile) grow. For Phase 3 the
  full-precision baseline A will be re-run through this same script so A/B/C share one code path.
- Bug found and fixed during the runs: nanochat's eval calls `wte.weight.device`; `TernaryEmbedding` now
  exposes `.weight` as a read-only alias of the int8 table. Regression test added.

## Storage (depth 4, width 256, vocab 4096)
| Run | Ternary weights | int8 as stored | packed at 1.58 bit | Remaining float params (fp32) |
|---|---|---|---|---|
| block matrices ternary | 3,145,728 | 3.15 MB | 0.62 MB | 4.19M (16.8 MB) |
| + token embedding ternary | 4,194,304 | 4.19 MB | 0.83 MB | 3.15M (12.6 MB) |
The original stores the 3.15M block weights as 12.6 MB of fp32.

## Training runs (600 steps, same data/schedule as Phase 0; ternary weights FROZEN)
| Run | Val bpb @600 | Zero fraction | Training time |
|---|---|---|---|
| A: original full precision (Phase 0) | **1.294** | n/a | 16.0 min (compiled) |
| 6 block matrices ternary, frozen | 1.593 | 56.3% (constant) | 18.6 min |
| + token embedding ternary, frozen | 2.108 | 50.0% (constant) | 18.2 min |

How to read this: these runs are a plumbing test, not a test of your idea. With the ternary weights frozen
and the two output projections starting at all-zero, every attention and MLP block outputs exactly zero,
so the model is effectively "embedding -> previous-token smear -> output head" (a bigram-like model).
1.59 bpb is what that skeleton can learn; it is the floor Phase 2 must beat. Freezing the token table too
removes most of the remaining learnable capacity, hence 2.11.

No speed claim: the fast path still multiplies by a float copy of {-1,0,+1}, so ternary is not faster yet
(it was slightly slower here, but the original was compiled and mine was not). Speed is Phase 4.
