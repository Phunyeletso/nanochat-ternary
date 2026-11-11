# Ternary-weight experiments

Research by Phunyeletso, built on [Andrej Karpathy's nanochat](https://github.com/karpathy/nanochat). The upstream MIT licence and history are retained. The root README distinguishes this work from the upstream project.

## Question and method

Can the six large attention and MLP matrices in a small transformer be trained directly as ternary weights (`-1`, `0`, `+1`), without storing latent full-precision weights between steps?

`ternary/train.py` converts those matrices to ternary layers. During backpropagation, each layer captures a temporary weight gradient; a flip rule then moves selected weights one step in the direction suggested by the gradient, and discards the gradient. The tested `topk` and `stochastic` rules do not keep a hidden full-precision weight copy. Each ternary matrix has a learned scale.

The accumulator method under investigation adds signed gradient votes to a small per-weight counter (targeting 4- or 8-bit state); the ternary weight flips when the counter crosses a threshold. The current `counter` implementation in `ternary/flip.py` uses an int8 counter, so 4-bit counters are not implemented. This per-weight state is a deliberate memory/quality tradeoff, unlike the no-hidden-copy `topk` and `stochastic` rules.

The comparisons include:

- **A:** original nanochat full-precision training.
- **B:** BitNet b1.58, with latent weights and a straight-through estimator (STE).
- **C:** direct ternary weights trained by discrete flips.

The planned comparison is a controlled 25-million-token run against BitNet. It is **in progress** and has not produced results yet. The numbers below are from earlier, smaller stand-in-data experiments; they are not results from that planned run and do not test the 25-million-token accumulator comparison.

## Results

All three Phase 3 models used the same 4-layer, width-256 architecture, stand-in dataset, 600 steps, 8,192 tokens per step, and two seeds. Lower validation bits per byte (bpb) is better.

| Model | Seed 0 | Seed 1 | Mean |
|---|---:|---:|---:|
| A: full precision | 1.2951 | 1.3004 | **1.2978** |
| B: BitNet b1.58 | 1.3697 | 1.3629 | **1.3663** |
| C: direct ternary, stochastic flips | 1.4040 | 1.4059 | **1.4049** |

The initial direct-ternary result trails BitNet by 0.039 bpb at this width. Later experiments found that a width-512 direct-ternary model reached 1.368 and 1.375 bpb across two seeds (mean 1.371), roughly matching the width-256 BitNet result. That is not a same-size comparison: width 512 has about four times as many block-matrix weights and took about 2.3 times as long to train on the measured CPU setup. No training-speed advantage is claimed.

An activation-precision experiment changed only the quantization of data entering ternary layers. At width 256, the 8-bit reference scored 1.403 bpb, 3 balanced-ternary tables (27 levels) scored 1.403, 2 tables (9 levels) scored 1.416, and 1 table (3 levels) scored 1.519. These are single-seed results; observed seed-to-seed variation elsewhere was about 0.002–0.007 bpb.

## Reproducing a run

Install the project's CPU and development dependencies, and prepare the nanochat tokenizer and data in the usual nanochat locations first:

```bash
uv sync --extra cpu --group dev
```

The experiment runs used a public-domain Gutenberg stand-in (11.6 million characters, 9,224 chunks, 5% held out) because the intended training dataset was unavailable. To prepare equivalent stand-in shards from a directory of `.txt` files:

```bash
python -m ternary.prep_standin_data --src /path/to/gutenberg --out "$NANOCHAT_BASE_DIR/base_data_climbmix"
```

Run the width-256 direct-ternary experiment with the defaults used in the notes:

```bash
python -m ternary.train --tag ternary-c256 --out-dir runs/ternary \
  --ternary-linear 1 --flip-rule stochastic --seed 0
```

For the alternatives, use the same training settings and data with `--tag full-precision` (A, no ternary flags), or `--tag bitnet --bitnet 1` (B). Training defaults are 600 steps, width 256, and 8,192 tokens per step. Results and logs are written to the selected output directory.

Run the ternary unit tests with:

```bash
uv run --group dev pytest ternary/tests
```

## Files

- `train.py`, `model.py`, `layers.py`, and `flip.py`: training entry point, model conversion, ternary operators, and weight-update rules.
- `bitnet.py`: BitNet b1.58 comparison implementation.
- `prep_standin_data.py`: creates nanochat-format shards from local text files.
- `rowformula.py`, `rowformula_fast.py`, `valuegroup.py`, and the `phase4_*.py` scripts: integer-matmul experiments and benchmarks.
- `tests/`: correctness and training tests for the experiment components.
- `PHASE*_NOTES.md` and `VALUEGROUP_NOTES.md`: experiment design, results, and limitations.
- `results/`: recorded logs, metrics, tables, samples, and plots from the earlier experiments.

## Scope and limitations

- The stand-in corpus and small CPU model are not representative of large-scale pretraining.
- The width-256 comparison is the controlled same-size comparison; the width-512 result trades a wider model for quality.
- The results do not demonstrate faster training or inference. The phase-4 CPU row-formula experiments reduced counted additions but were slower than ordinary matrix multiplication and BitNet's kernel in the measured cases.
- Some later phases test ternary embeddings, output layers, or activation precision; those are exploratory and do not change the main comparison.

The detailed phase notes, plots, logs, scripts, and tests are kept in this directory. The license for the upstream project is in the repository root.
