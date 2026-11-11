# Phase 0: Understanding the existing code (no changes made)

Code base: nanochat at commit 92d63d4. All local edits in your copy are comments only
(verified: `git diff -U0` contains no non-comment, non-blank changed lines).

## Architecture (nanochat/gpt.py)
- **Token embedding** `transformer.wte`: nn.Embedding(V, C). Output is RMS-normed, then "smeared"
  (each position mixes in the previous token's embedding through a tiny gate).
- **Value embeddings** `value_embeds[i]`: a second vocab-sized table (V x C) on alternating layers
  (layers 1 and 3 for depth 4), added into the attention V vectors through a learned gate.
- **Blocks** (pre-norm): `x = x + attn(norm(x)); x = x + mlp(norm(x))`, plus per-layer scalars
  `resid_lambdas`, `x0_lambdas` (re-mixing the input embedding) and a mid-depth "backout" subtraction.
- **Attention**: Q/K/V projections, rotary position embeddings, QK-norm, causal SDPA (FA3 on GPU),
  output projection. No biases.
- **MLP**: C -> 4C, relu(x)^2, 4C -> C.
- **Norms**: parameter-free RMSNorm (`F.rms_norm`), no learnable gain.
- **Output head** `lm_head`: Linear(C, V), logits soft-capped at +/-15.

## Every Linear (nanochat's `Linear`, a weight-casting nn.Linear)
| Module | Shape (out x in) | Phase 1 plan |
|---|---|---|
| `h[i].attn.c_q` / `c_k` / `c_v` | C x C | -> TernaryLinear |
| `h[i].attn.c_proj` | C x C | -> TernaryLinear |
| `h[i].mlp.c_fc` | 4C x C | -> TernaryLinear |
| `h[i].mlp.c_proj` | C x 4C | -> TernaryLinear |
| `h[i].attn.ve_gate` | n_kv_head x 12 (value-embedding layers only) | keep (tiny gate) |
| `smear_gate` | 1 x 24 | keep (tiny gate) |
| `lm_head` | V x C | keep full precision (per your spec) |

## Optimizer (nanochat/optim.py, `MuonAdamW`)
- All 2-D block matrices go to **Muon** (momentum + Newton-Schulz orthogonalisation, 5 steps,
  variance normalisation, cautious weight decay with a cosine schedule).
- Embeddings, value embeddings, lm_head and scalars go to **AdamW**, each group with its own LR.
- LRs scale with 1/sqrt(C/768) (AdamW groups) and sqrt(batch/524288) (all groups).

## Training loop (scripts/base_train.py)
- Model built on the meta device, then `init_weights()`; `torch.compile`d.
- Data: parquet shards -> BOS-aligned best-fit packing into (B, T) batches; the last shard is validation.
- Each step: gradient accumulation to reach the total batch, linear warmup / constant / linear warmdown
  LR schedule, Muon momentum and weight-decay schedules, `optimizer.step()`.
- Logs a debiased EMA of the training loss (smoothed, beta 0.9), and validation bits per byte every N steps.

## Baseline run
- Data: **stand-in** text (NLTK's public-domain Gutenberg corpus, 11.6M chars, 9,224 chunks,
  5% held out), because Hugging Face / TinyStories is unreachable from this environment.
  Prepared with `ternary/prep_standin_data.py`. Tokenizer: the original `scripts/tok_train.py`, vocab 4096.
- Command (original script, unmodified):
  `python -m scripts.base_train --depth=4 --aspect-ratio=64 --head-dim=64 --window-pattern=L --max-seq-len=256 --device-batch-size=32 --total-batch-size=8192 --num-iterations=600 --eval-every=50 --eval-tokens=131072 --core-metric-every=-1 --sample-every=-1 --device-type=cpu`
- Model: 4 layers, C=256, 4 heads, 7.34M params total
  (block matrices 3.15M, wte 1.05M, value embeds 2.10M, lm_head 1.05M, scalars 34).
- Hardware: 2 CPU cores, cloud container. 16.0 min of training for 600 steps (~1.6 s/step, ~5k tok/s).
- Result: validation bpb 3.155 -> 1.294 at step 600; train loss (smoothed) 8.32 -> 3.42.
  Full numbers in `results/phase0_baseline.csv`.
