"""
Phase 3 chart: A (full precision), B (BitNet b1.58, latent weights), C (direct ternary, stochastic flip),
each run with seeds 0 and 1. Thin lines = individual seeds, thick = mean.
    python -m ternary.plot_phase3 --runs-dir runs --out runs/phase3_abc.png
"""
import argparse
import json
import os

MODELS = [("A: full precision (original)", "p3_A_fp"),
          ("B: BitNet b1.58 (latent weights + STE)", "p3_B_bitnet"),
          ("C: direct ternary, stochastic flip (yours)", "p3_C_stochastic")]
COLORS = ["#1f77b4", "#2ca02c", "#d62728"]


def load(path):
    train, val = {}, {}
    for line in open(path):
        r = json.loads(line)
        if "val_bpb" in r:
            val[r["step"]] = r["val_bpb"]
        else:
            train[r["step"]] = r["loss_smooth"]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="runs/phase3_abc.png")
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    summary = []
    for (label, prefix), c in zip(MODELS, COLORS):
        runs = [load(os.path.join(args.runs_dir, f"{prefix}_s{s}.jsonl")) for s in seeds]
        steps_t = sorted(set.intersection(*[set(r[0]) for r in runs]))
        steps_v = sorted(set.intersection(*[set(r[1]) for r in runs]))
        for tr, va in runs:
            ax1.plot(steps_t, [tr[s] for s in steps_t], color=c, lw=0.6, alpha=0.35)
            ax2.plot(steps_v, [va[s] for s in steps_v], color=c, lw=0.8, alpha=0.35)
        mean_t = [sum(r[0][s] for r in runs) / len(runs) for s in steps_t]
        mean_v = [sum(r[1][s] for r in runs) / len(runs) for s in steps_v]
        ax1.plot(steps_t, mean_t, color=c, lw=1.6, label=label)
        ax2.plot(steps_v, mean_v, color=c, lw=2, marker="o", ms=3, label=label)
        finals = [r[1][max(steps_v)] for r in runs]
        summary.append((label, finals, sum(finals) / len(finals)))
    ax1.set_title("Training loss (smoothed)"); ax1.set_xlabel("step"); ax1.set_ylabel("nats / token")
    ax2.set_title("Validation bits per byte (lower is better)"); ax2.set_xlabel("step")
    ax2.set_ylim(top=2.2)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Phase 3: same model size, data, steps; 2 seeds each (thin = single seed, thick = mean)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    for label, finals, mean in summary:
        print(f"{label:45s} final val bpb per seed: {', '.join(f'{f:.4f}' for f in finals)}  mean: {mean:.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
