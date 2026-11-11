"""
Stand-in dataset for experiments while TinyStories is unreachable.

Takes a folder of plain .txt files (we use NLTK's public-domain Gutenberg corpus),
splits them into paragraph-aligned "documents" of ~1-2k characters, shuffles them
with a fixed seed, and writes nanochat-format parquet shards (one 'text' column).
nanochat convention: the LAST shard is the validation split.

Usage:
    python -m ternary.prep_standin_data --src /path/to/gutenberg --out $NANOCHAT_BASE_DIR/base_data_climbmix

To switch to TinyStories later: write the stories (one per row) into shards the same way.
"""
import argparse
import glob
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq


def chunk_text(text, min_chars=1000, max_chars=2000):
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    paras = [" ".join(p.split()) for p in paras if p.strip()]
    docs, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 1 > max_chars:
            docs.append(cur)
            cur = ""
        cur = (cur + "\n" + p) if cur else p
        if len(cur) >= min_chars:
            docs.append(cur)
            cur = ""
    if cur:
        docs.append(cur)
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--n-train-shards", type=int, default=4)
    ap.add_argument("--row-group-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    docs = []
    for path in sorted(glob.glob(os.path.join(args.src, "*.txt"))):
        with open(path, encoding="latin-1") as f:
            docs.extend(chunk_text(f.read()))
    random.Random(args.seed).shuffle(docs)
    n_val = int(len(docs) * args.val_frac)
    val, train = docs[:n_val], docs[n_val:]

    os.makedirs(args.out, exist_ok=True)
    per = (len(train) + args.n_train_shards - 1) // args.n_train_shards
    shards = [train[i * per:(i + 1) * per] for i in range(args.n_train_shards)] + [val]
    for i, rows in enumerate(shards):
        table = pa.table({"text": rows})
        pq.write_table(table, os.path.join(args.out, f"shard_{i:05d}.parquet"), row_group_size=args.row_group_size)
    nchars = sum(map(len, docs))
    print(f"{len(docs)} docs, {nchars:,} chars -> {len(train)} train docs in {args.n_train_shards} shards, {len(val)} val docs")


if __name__ == "__main__":
    main()
