"""Bootstrap nanobody surrogate training data by running real BoltzGen.

Strategy: generate ~N candidates via the template + CDR mutator (already wired
in elite_miner.nanobody.generator), pass them through BoltzGen, write labels.

Usage:
    python -m elite_miner.scripts.collect_labels_nb \
        --target Q9NZQ7 \
        --target-clip-interval 17,132 \
        --n-candidates 500 \
        --batch-size 4 \
        --output cache/nb_labels/Q9NZQ7.parquet

Cost: BoltzGen is slower per call than Boltz2; expect ~30-60s/sequence on A100.
500 candidates → ~5-8 hours = ~$5-8 at $1.05/hr A100 80GB spot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from elite_miner.nanobody.filters import NanobodyFilter, NanobodyValidityConfig
from elite_miner.nanobody.generator import NanobodyGenerator, GenerationConfig
from elite_miner.nanobody.scorer import BoltzGenScorer
from elite_miner.nanobody.label_writer import (
    append_labels,
    default_path,
    make_label_row,
    BOLTZGEN_METRICS,
)


def already_scored(parquet_path: str, target: str) -> set[str]:
    if not os.path.exists(parquet_path):
        return set()
    try:
        df = pd.read_parquet(parquet_path, columns=["sequence", "target"])
        return set(df[df["target"] == target]["sequence"].tolist())
    except Exception:
        return set()


def stratified_generate(n_candidates: int, validity_cfg: NanobodyValidityConfig) -> list[str]:
    """Generate stratified candidates by template + length bucket.

    Round-robin across templates so we get balanced coverage. Within each
    template, mix mutation counts (low / mid / high) to span the length-equivalent
    space (since mutation count doesn't change length, this is mostly about diversity).
    """
    from elite_miner.nanobody.templates import TEMPLATES

    # Mutation budgets: low (1-3), mid (4-6), high (7-10) — different perturbation levels
    budgets = [(1, 3), (4, 6), (7, 10)]
    per_template_per_budget = max(5, n_candidates // (len(TEMPLATES) * len(budgets)))
    candidates: set[str] = set()
    seqs: list[str] = []

    for template in TEMPLATES:
        for (lo, hi) in budgets:
            gen = NanobodyGenerator(
                validity_cfg,
                gen_cfg=GenerationConfig(min_mutations=lo, max_mutations=hi),
                templates=[template],
            )
            batch = gen.generate_batch(per_template_per_budget)
            for s in batch:
                if s not in candidates:
                    candidates.add(s)
                    seqs.append(s)
                if len(seqs) >= n_candidates:
                    return seqs
    return seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="UniProt target id")
    p.add_argument("--target-clip-interval", default="None",
                   help="clip_interval as 'None' or 'start,end'")
    p.add_argument("--n-candidates", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4,
                   help="BoltzGen is heavier per call than Boltz2; small batches are fine")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    output = args.output or default_path(args.target)

    validity_cfg = NanobodyValidityConfig.from_config({})
    print(f"[collect-nb] generating {args.n_candidates} valid candidates...")
    candidates = stratified_generate(args.n_candidates, validity_cfg)
    print(f"[collect-nb] generated {len(candidates)} unique valid sequences")

    seen = already_scored(output, args.target)
    fresh = [s for s in candidates if s not in seen]
    print(f"[collect-nb] {len(candidates) - len(fresh)} already scored, {len(fresh)} fresh")
    if not fresh:
        print("[collect-nb] nothing to score")
        return

    clip = None if args.target_clip_interval == "None" else tuple(
        int(x) for x in args.target_clip_interval.split(",")
    )
    boltzgen_config = {
        "nanobody_target": [args.target],
        "nanobody_target_clip_interval": [clip],
        "boltzgen_rank_mode": "min",
        "boltzgen_rank_by": "rank_sum",
        "num_sequences": 1,
    }

    # Disable label streaming inside scorer — this script writes explicitly
    scorer = BoltzGenScorer(target=args.target, stream_labels=False)
    run_id = "boot-" + hashlib.sha256(f"{args.target}-{dt.datetime.now()}".encode()).hexdigest()[:12]

    rows = []
    for batch_start in range(0, len(fresh), args.batch_size):
        batch = fresh[batch_start: batch_start + args.batch_size]
        print(f"[collect-nb] scoring {batch_start+1}..{batch_start+len(batch)} of {len(fresh)}")
        try:
            scored = scorer.score_batch(batch, boltzgen_config)
        except Exception as e:
            print(f"[collect-nb] batch failed: {e}")
            continue
        for s in scored:
            if s.raw is None:
                continue
            row = make_label_row(s.sequence, args.target, s.raw)
            row["run_id"] = run_id
            rows.append(row)

        # Flush every batch — protect against long-running failures losing data
        if rows:
            append_labels(output, rows)
            print(f"[collect-nb] wrote {len(rows)} total rows so far to {output}")
            rows = []  # don't double-write

    if rows:
        append_labels(output, rows)


if __name__ == "__main__":
    main()
