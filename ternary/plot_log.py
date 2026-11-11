"""
Parse nanochat base_train logs into CSV and plot loss curves.

Usage:
    python -m ternary.plot_log --out curve.png name1=run1.log [name2=run2.log ...]
Writes <log>.csv next to each log (step, train_loss, val_bpb).
"""
import argparse
import csv
import re

STEP_RE = re.compile(r"^step (\d+)/\d+ .*?\| loss: ([0-9.]+)")
VAL_RE = re.compile(r"^Step (\d+) \| Validation bpb: ([0-9.]+)")


def parse(path):
    train, val = {}, {}
    with open(path) as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                train[int(m.group(1))] = float(m.group(2))
            m = VAL_RE.match(line)
            if m:
                val[int(m.group(1))] = float(m.group(2))
    return train, val


def write_csv(path, train, val):
    steps = sorted(set(train) | set(val))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_bpb"])
        for s in steps:
            w.writerow([s, train.get(s, ""), val.get(s, "")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("runs", nargs="+", help="name=path.log")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for spec in args.runs:
        name, path = spec.split("=", 1)
        train, val = parse(path)
        write_csv(path.rsplit(".", 1)[0] + ".csv", train, val)
        ts = sorted(train)
        ax1.plot(ts, [train[s] for s in ts], label=name, lw=1)
        vs = sorted(val)
        ax2.plot(vs, [val[s] for s in vs], marker="o", ms=3, label=name)
    ax1.set_xlabel("step"); ax1.set_ylabel("train loss (nats/token)"); ax1.set_title("Training loss")
    ax2.set_xlabel("step"); ax2.set_ylabel("validation bits per byte"); ax2.set_title("Validation bpb")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.legend()
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
