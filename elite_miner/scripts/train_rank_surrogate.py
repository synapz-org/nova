"""Train a rank-sum-aware multi-head nb surrogate with pairwise ranking loss.

This is the corrective for yesterday's surrogate which:
  - regressed on design_iiptm only (5th most important metric)
  - used MSE loss (wrong for ranking-validator)
  - hit high-band Spearman 0.46 (too noisy for top-1 picks)

New approach (informed by kb/raw/papers-2026-05-sn68-nanobody-survey.md):
  - Train ten heads, one per BoltzGen metric, sharing a backbone.
  - Pairwise margin-ranking loss instead of MSE (AbRank, MosPro).
  - Combine head outputs into predicted rank-sum at inference.

Features: starts with existing 33d seqstat features (the same nb_features() the
production surrogate uses). The architectural improvement (loss + multi-head)
should beat baseline even without richer features. The ESM2 upgrade comes after.

Usage:
    python -m elite_miner.scripts.train_rank_surrogate \
        --archive cache/archive_Q9NZQ7.parquet \
        --target Q9NZQ7 \
        --output-dir models/nb_rank_v1 \
        --epochs 50 --batch-size 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from elite_miner.nanobody.features import nb_features


METRICS = [
    ("design_iiptm",            "max"),
    ("design_ptm",              "max"),
    ("design_to_target_iptm",   "max"),
    ("min_design_to_target_pae","min"),
    ("interaction_pae",         "min"),
    ("plip_hbonds_refolded",    "max"),
    ("plip_saltbridge_refolded","max"),
    ("delta_sasa_refolded",     "max"),
    ("liability_score",         "min"),
    ("liability_num_violations","min"),
]


def featurize(seqs: list[str]) -> tuple[np.ndarray, list[int]]:
    """Build feature matrix using existing nb_features. Returns (X, kept_indices)."""
    rows = []
    kept = []
    for i, s in enumerate(seqs):
        v = nb_features(s)
        if v is not None:
            rows.append(v)
            kept.append(i)
    return np.stack(rows).astype(np.float32), kept


def compute_rank_sum(df: pd.DataFrame) -> np.ndarray:
    """Validator-style: dense-rank each metric, sum across the 10. Lower = better."""
    rank_cols = []
    for m, mode in METRICS:
        if m not in df.columns:
            raise KeyError(f"missing metric {m}")
        ranks = df[m].rank(method='dense', ascending=(mode == 'min')).values
        rank_cols.append(ranks)
    return np.stack(rank_cols, axis=1).sum(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="cache/archive_Q9NZQ7.parquet")
    parser.add_argument("--target", default="Q9NZQ7")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"[train_rank] loading archive from {args.archive}")
    df = pd.read_parquet(args.archive)
    print(f"[train_rank] {len(df)} rows total")

    # Validate columns
    for m, _ in METRICS:
        if m not in df.columns:
            print(f"[train_rank] missing metric column: {m}")
            return 1

    # Drop rows with any missing metric values
    df = df.dropna(subset=[m for m, _ in METRICS]).reset_index(drop=True)
    print(f"[train_rank] {len(df)} rows after dropna")

    # Featurize
    X, kept = featurize(df["sequence"].tolist())
    df = df.iloc[kept].reset_index(drop=True)
    print(f"[train_rank] {len(df)} after featurize; feat dim={X.shape[1]}")

    # Compute targets: dense rank per metric (ascending so lower = better)
    rank_per_metric = np.zeros((len(df), len(METRICS)), dtype=np.float32)
    for i, (m, mode) in enumerate(METRICS):
        rank_per_metric[:, i] = df[m].rank(method='dense', ascending=(mode == 'min')).values
    # Normalize ranks to [0, 1] per metric for stable training
    max_rank = rank_per_metric.max(axis=0, keepdims=True)
    rank_norm = rank_per_metric / max_rank  # 0 = best, 1 = worst

    rank_sum = rank_per_metric.sum(axis=1)
    print(f"[train_rank] rank_sum: min={rank_sum.min():.0f} median={np.median(rank_sum):.0f} max={rank_sum.max():.0f}")

    # Train/holdout split — stratify by rank_sum quartile so high band is represented in holdout
    rng = np.random.default_rng(args.seed)
    quartiles = pd.qcut(rank_sum, q=4, labels=False, duplicates='drop')
    holdout_idx = []
    for q in np.unique(quartiles):
        idx = np.where(quartiles == q)[0]
        n_hold = max(1, int(len(idx) * args.holdout_frac))
        sel = rng.choice(idx, size=n_hold, replace=False)
        holdout_idx.extend(sel.tolist())
    holdout_mask = np.zeros(len(df), dtype=bool)
    holdout_mask[holdout_idx] = True
    X_train, X_hold = X[~holdout_mask], X[holdout_mask]
    y_train_norm = rank_norm[~holdout_mask]
    y_hold_norm = rank_norm[holdout_mask]
    y_train_rs = rank_sum[~holdout_mask]
    y_hold_rs = rank_sum[holdout_mask]
    print(f"[train_rank] train={len(X_train)} holdout={len(X_hold)}")

    # Build model — small MLP with shared backbone + 10 heads
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except ImportError:
        print("[train_rank] torch not installed — install: pip install torch")
        return 1

    device = torch.device(args.device)
    H = args.hidden

    class MultiHeadRanker(nn.Module):
        def __init__(self, in_dim, hidden, n_heads):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
            )
            self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_heads)])

        def forward(self, x):
            z = self.backbone(x)
            return torch.cat([h(z) for h in self.heads], dim=1)  # (N, 10)

    model = MultiHeadRanker(X.shape[1], H, len(METRICS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    X_tr_t = torch.from_numpy(X_train).to(device)
    y_tr_t = torch.from_numpy(y_train_norm).to(device)
    X_ho_t = torch.from_numpy(X_hold).to(device)
    y_ho_t = torch.from_numpy(y_hold_norm).to(device)

    from scipy.stats import spearmanr

    best_high_rho = -1
    best_state = None
    n_train = len(X_train)

    print("[train_rank] training with margin-ranking loss")
    for epoch in range(args.epochs):
        model.train()
        # Random pairs for ranking loss — sample 2*batch indices, compare
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = max(1, n_train // args.batch_size)
        for b in range(n_batches):
            i = perm[b * args.batch_size:(b + 1) * args.batch_size]
            j = perm[(b + 1) * args.batch_size:(b + 2) * args.batch_size]
            if len(j) < len(i):
                # wrap
                j = np.concatenate([j, perm[:len(i) - len(j)]])
            ii = torch.from_numpy(i).long().to(device)
            jj = torch.from_numpy(j).long().to(device)
            pred_i = model(X_tr_t[ii])  # (B, 10)
            pred_j = model(X_tr_t[jj])
            true_i = y_tr_t[ii]
            true_j = y_tr_t[jj]
            # Margin-ranking loss per metric: if true_i < true_j (i better), want pred_i < pred_j
            sign = torch.sign(true_j - true_i)  # +1 if i is better, -1 if j is better
            margin = 0.1
            loss = F.relu(margin - sign * (pred_j - pred_i)).mean()
            # Also a regression auxiliary so heads don't go on a constant
            mse = F.mse_loss(model(X_tr_t[ii]), true_i) + F.mse_loss(model(X_tr_t[jj]), true_j)
            total = loss + 0.1 * mse
            opt.zero_grad()
            total.backward()
            opt.step()
            epoch_loss += total.item()
        avg_loss = epoch_loss / max(1, n_batches)

        # Eval
        model.eval()
        with torch.no_grad():
            pred_ho = model(X_ho_t).cpu().numpy()  # (N, 10)
        # Predicted rank-sum = sum of head outputs (heads predict normalized rank, lower = better)
        pred_rs = pred_ho.sum(axis=1)
        rho, _ = spearmanr(pred_rs, y_hold_rs)

        # High-band Spearman: restricted to top 10% by true rank_sum (= lowest rank_sum)
        threshold = np.quantile(y_hold_rs, 0.10)
        hi_mask = y_hold_rs <= threshold
        if hi_mask.sum() >= 10:
            hi_rho, _ = spearmanr(pred_rs[hi_mask], y_hold_rs[hi_mask])
        else:
            hi_rho = float('nan')

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[train_rank] epoch={epoch} loss={avg_loss:.4f} holdout_rho={rho:.3f} high_band_rho={hi_rho:.3f}")

        if not np.isnan(hi_rho) and hi_rho > best_high_rho:
            best_high_rho = hi_rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"[train_rank] best high-band Spearman: {best_high_rho:.3f} (baseline was 0.465)")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    metadata = {
        "feature_dim": X.shape[1],
        "hidden": args.hidden,
        "n_metrics": len(METRICS),
        "metrics": [m for m, _ in METRICS],
        "best_high_band_rho": best_high_rho,
        "n_train": int((~holdout_mask).sum()),
        "n_holdout": int(holdout_mask.sum()),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train_rank] saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
