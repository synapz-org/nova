"""Extract ESM2-650M embeddings for all archive sequences.

This is the critical-path GPU step from kb/strategy/02-the-plan.md Component 1.
The hypothesis (per literature: pLDDT-Predictor 2410.21283, Twin Peaks 2509.22950)
is that ESM2 embeddings + small head will lift our high-band Spearman from 0.46
(baseline) toward 0.7+. Tonight's PoC with 33d/734d features couldn't break 0.46.
Validating that hypothesis on actual ESM2 features is the first thing to do on
a GPU box.

Output: cache/archive_Q9NZQ7_esm2.parquet — same rows as archive_Q9NZQ7.parquet
but with an 'esm2' column = pickled np.array of shape (1280,) per sequence
(mean-pooled across residues).

Usage (on the Basilica box):
    pip install fair-esm
    python -m elite_miner.scripts.extract_esm2_features \
        --archive cache/archive_Q9NZQ7.parquet \
        --output cache/archive_Q9NZQ7_esm2.parquet \
        --batch-size 8 --device cuda

Cost: ~6 hours on A100 for 8129 sequences at batch_size=8 with ESM2-650M.
Memory: ~12GB GPU.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="cache/archive_Q9NZQ7.parquet")
    parser.add_argument("--output", default="cache/archive_Q9NZQ7_esm2.parquet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model",
        default="esm2_t33_650M_UR50D",
        help="esm2 model name. Defaults to 650M; use _t12_35M_UR50D for fast smoke test (smaller embeddings).",
    )
    parser.add_argument("--resume", action="store_true",
                        help="If output exists, only extract missing rows.")
    args = parser.parse_args()

    print(f"[esm2] loading archive: {args.archive}")
    df = pd.read_parquet(args.archive)
    print(f"[esm2] {len(df)} sequences")

    # Resume support
    seen_seqs = set()
    if args.resume and os.path.exists(args.output):
        prev = pd.read_parquet(args.output)
        seen_seqs = set(prev["sequence"])
        print(f"[esm2] resume: {len(seen_seqs)} already extracted")

    todo = df[~df["sequence"].isin(seen_seqs)].reset_index(drop=True)
    print(f"[esm2] {len(todo)} sequences to extract")
    if len(todo) == 0:
        print("[esm2] nothing to do — output is complete")
        return 0

    # Load ESM2
    print(f"[esm2] loading model: {args.model}")
    try:
        import esm
        import torch
    except ImportError:
        print("[esm2] esm not installed. Run: pip install fair-esm")
        return 1

    model_loader = getattr(esm.pretrained, args.model)
    model, alphabet = model_loader()
    model = model.eval().to(args.device)
    batch_converter = alphabet.get_batch_converter()
    repr_layer = model.num_layers
    print(f"[esm2] using layer {repr_layer} for embeddings")

    sequences = todo["sequence"].tolist()
    results = []
    n_batches = (len(sequences) + args.batch_size - 1) // args.batch_size
    t0 = time.time()

    for bi in range(n_batches):
        batch = sequences[bi * args.batch_size:(bi + 1) * args.batch_size]
        data = [(f"s{i}", s) for i, s in enumerate(batch)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(args.device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        emb = out["representations"][repr_layer]  # (B, L+2, D)
        # Mean-pool over non-padding residues (exclude BOS/EOS)
        for i, seq in enumerate(batch):
            L = len(seq)
            v = emb[i, 1:L + 1].mean(dim=0).cpu().numpy().astype(np.float32)
            results.append({"sequence": seq, "esm2": pickle.dumps(v)})
        if bi % 50 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            rate = (bi + 1) * args.batch_size / elapsed
            eta = (len(sequences) - (bi + 1) * args.batch_size) / max(rate, 1)
            print(f"[esm2] batch {bi+1}/{n_batches} done — {rate:.1f} seqs/s, eta {eta/60:.1f} min")

    out_df = pd.DataFrame(results)
    if args.resume and os.path.exists(args.output):
        prev = pd.read_parquet(args.output)
        out_df = pd.concat([prev, out_df], ignore_index=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    print(f"[esm2] wrote {len(out_df)} embeddings to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
