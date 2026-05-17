"""Summarize real iiptm vs predicted iiptm of our submitted nb sequences.

Joins:
- cache/submission_queue/nb.jsonl     (what we submitted)
- cache/submission_labels/<target>.parquet  (real iiptm from BoltzGen)

Prints headline stats per fast_batch_size cohort, plus the per-epoch ledger.
This is the only place that answers "did n=5000 actually help?"

Usage:
    python -m elite_miner.scripts.analyze_submission_labels [--target Q9NZQ7]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def load_queue(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return pd.DataFrame(rows)


def load_labels(output_dir: str, target: str) -> pd.DataFrame:
    p = os.path.join(output_dir, f"{target}.parquet")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_parquet(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="cache/submission_queue/nb.jsonl")
    parser.add_argument("--labels-dir", default="cache/submission_labels")
    parser.add_argument("--target", default="Q9NZQ7")
    args = parser.parse_args()

    q = load_queue(args.queue)
    l = load_labels(args.labels_dir, args.target)

    if q.empty:
        print(f"no queue entries at {args.queue}")
        return
    print(f"queue: {len(q)} submissions (target={args.target})")
    q = q[q["target"] == args.target]
    print(f"  for {args.target}: {len(q)}")

    if l.empty:
        print(f"no labeled submissions at {args.labels_dir}/{args.target}.parquet yet")
        print(f"  ({len(q)} queued, waiting for worker to label them)")
        return

    print(f"labels: {len(l)} entries")

    # Per-cohort summary by fast_batch_size
    if "fast_batch_size" in l.columns:
        print()
        print("=== by fast_batch_size cohort ===")
        for fbs, grp in l.groupby("fast_batch_size"):
            if grp["real_iiptm"].isna().all():
                continue
            real = grp["real_iiptm"].dropna()
            pred = grp["predicted_iiptm"].dropna()
            print(
                f"  fbs={fbs}: n={len(grp)} "
                f"real_iiptm mean={real.mean():.4f} median={real.median():.4f} max={real.max():.4f}"
            )
            if not pred.empty:
                # systematic bias of surrogate vs reality
                joined = grp.dropna(subset=["real_iiptm", "predicted_iiptm"])
                if not joined.empty:
                    bias = (joined["real_iiptm"] - joined["predicted_iiptm"]).mean()
                    print(f"        pred mean={pred.mean():.4f}; (real - pred) mean bias={bias:+.4f}")

    # Latest 10 labeled, sorted by labeled_at if present, else by index
    print()
    print("=== latest labeled submissions ===")
    sort_key = "labeled_at" if "labeled_at" in l.columns else None
    if sort_key:
        l_sorted = l.sort_values(sort_key, ascending=False)
    else:
        l_sorted = l.iloc[::-1]
    cols = [c for c in ["epoch", "fast_batch_size", "predicted_iiptm", "real_iiptm", "elapsed_s"] if c in l_sorted.columns]
    print(l_sorted[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
