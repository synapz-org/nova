"""Train the rank-sum multi-head ranker using ESM2 embeddings.

This is the critical experiment from kb/strategy/05-probability-of-success.md:
if high-band Spearman exceeds 0.55 after this trains, the strategy is viable;
if not, we abort the deploy and revisit assumptions.

Run after extract_esm2_features.py has populated cache/archive_Q9NZQ7_esm2.parquet.

Usage:
    python -m elite_miner.scripts.train_rank_surrogate_esm2 \
        --archive cache/archive_Q9NZQ7.parquet \
        --esm2 cache/archive_Q9NZQ7_esm2.parquet \
        --output-dir models/nb_rank_esm2_v1 \
        --epochs 100 --batch-size 256 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

METRICS = [
    ("design_iiptm", "max"),
    ("design_ptm", "max"),
    ("design_to_target_iptm", "max"),
    ("min_design_to_target_pae", "min"),
    ("interaction_pae", "min"),
    ("plip_hbonds_refolded", "max"),
    ("plip_saltbridge_refolded", "max"),
    ("delta_sasa_refolded", "max"),
    ("liability_score", "min"),
    ("liability_num_violations", "min"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="cache/archive_Q9NZQ7.parquet")
    parser.add_argument("--esm2", default="cache/archive_Q9NZQ7_esm2.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--include-seqstat",
        action="store_true",
        help="Concat 33d seqstat features alongside ESM2. Costs nothing, may help.",
    )
    args = parser.parse_args()

    # Load + merge
    print("[train_esm2] loading data")
    arch = pd.read_parquet(args.archive)
    esm = pd.read_parquet(args.esm2)
    df = arch.merge(esm, on="sequence", how="inner")
    df = df.dropna(subset=[m for m, _ in METRICS]).reset_index(drop=True)
    print(f"[train_esm2] {len(df)} rows after merge + dropna")

    # Unpickle embeddings
    print("[train_esm2] unpickling esm2")
    X_esm = np.stack([pickle.loads(b) for b in df["esm2"].values]).astype(np.float32)
    print(f"[train_esm2] ESM2 features: {X_esm.shape}")

    # Optional concat seqstat
    if args.include_seqstat:
        from elite_miner.nanobody.features import nb_features
        seqstat = []
        for s in df["sequence"]:
            v = nb_features(s)
            if v is None:
                v = np.zeros(33, dtype=np.float32)
            seqstat.append(v)
        X_seq = np.stack(seqstat).astype(np.float32)
        # z-score normalize each block
        X_esm = (X_esm - X_esm.mean(axis=0, keepdims=True)) / (X_esm.std(axis=0, keepdims=True) + 1e-6)
        X_seq = (X_seq - X_seq.mean(axis=0, keepdims=True)) / (X_seq.std(axis=0, keepdims=True) + 1e-6)
        X = np.concatenate([X_esm, X_seq], axis=1)
    else:
        X = (X_esm - X_esm.mean(axis=0, keepdims=True)) / (X_esm.std(axis=0, keepdims=True) + 1e-6)
    print(f"[train_esm2] final feature dim: {X.shape[1]}")

    # Targets
    ranks = np.zeros((len(df), len(METRICS)), dtype=np.float32)
    for i, (m, mode) in enumerate(METRICS):
        ranks[:, i] = df[m].rank(method="dense", ascending=(mode == "min")).values
    maxr = ranks.max(axis=0, keepdims=True)
    rn = ranks / maxr
    rs = ranks.sum(axis=1)
    print(f"[train_esm2] rank_sum: min={rs.min():.0f} median={np.median(rs):.0f} max={rs.max():.0f}")

    # Stratified holdout
    rng = np.random.default_rng(args.seed)
    q = pd.qcut(rs, q=4, labels=False, duplicates="drop")
    holdout = []
    for qi in np.unique(q):
        idx = np.where(q == qi)[0]
        n = max(1, int(len(idx) * args.holdout_frac))
        holdout.extend(rng.choice(idx, size=n, replace=False).tolist())
    hm = np.zeros(len(df), dtype=bool)
    hm[holdout] = True
    print(f"[train_esm2] train={(~hm).sum()} hold={hm.sum()}")

    # Train
    import torch
    from torch import nn
    import torch.nn.functional as F
    from scipy.stats import spearmanr

    device = torch.device(args.device)
    H = args.hidden
    D = X.shape[1]

    class MultiHeadRanker(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = nn.Sequential(
                nn.Linear(D, H), nn.GELU(), nn.Dropout(args.dropout),
                nn.Linear(H, H), nn.GELU(), nn.Dropout(args.dropout),
                nn.Linear(H, H), nn.GELU(),
            )
            self.heads = nn.ModuleList([nn.Linear(H, 1) for _ in METRICS])

        def forward(self, x):
            z = self.bb(x)
            return torch.cat([h(z) for h in self.heads], dim=1)

    model = MultiHeadRanker().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Xt = torch.from_numpy(X[~hm]).to(device)
    yt = torch.from_numpy(rn[~hm]).to(device)
    Xh = torch.from_numpy(X[hm]).to(device)
    y_ho = rn[hm]
    rs_ho = rs[hm]
    n_tr = len(Xt)

    best_high = -1
    best_state = None
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(n_tr)
        L = 0.0
        nb = 0
        bs = args.batch_size
        for b in range(n_tr // bs):
            i = perm[b * bs:(b + 1) * bs]
            j = perm[(b + 1) * bs:(b + 2) * bs]
            if len(j) < len(i):
                j = np.concatenate([j, perm[:len(i) - len(j)]])
            ii = torch.from_numpy(i).long().to(device)
            jj = torch.from_numpy(j).long().to(device)
            pi = model(Xt[ii])
            pj = model(Xt[jj])
            ti = yt[ii]
            tj = yt[jj]
            sign = torch.sign(tj - ti)
            margin = 0.1
            rank_loss = F.relu(margin - sign * (pj - pi)).mean()
            mse = F.mse_loss(model(Xt[ii]), ti) + F.mse_loss(model(Xt[jj]), tj)
            total = rank_loss + 0.1 * mse
            opt.zero_grad()
            total.backward()
            opt.step()
            L += total.item()
            nb += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            ph = model(Xh).cpu().numpy()
        pred_rs = ph.sum(axis=1)
        rho, _ = spearmanr(pred_rs, rs_ho)
        thr = np.quantile(rs_ho, 0.10)
        hi_mask = rs_ho <= thr
        hi_rho, _ = spearmanr(pred_rs[hi_mask], rs_ho[hi_mask])

        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[train_esm2] ep={ep} loss={L/max(1,nb):.4f} rho={rho:.3f} high={hi_rho:.3f}")
        if hi_rho > best_high:
            best_high = hi_rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"[train_esm2] BEST high-band Spearman: {best_high:.3f}")
    print(f"            (baseline iiptm-only was 0.465; PoC 33d/734d hit ~0.32 ceiling)")
    if best_high < 0.55:
        print("[train_esm2] ⚠ DID NOT BEAT 0.55 THRESHOLD — abort live deploy per strategy 05-probability-of-success.md")
    else:
        print("[train_esm2] ✓ above 0.55 threshold — proceed to live deploy")

    os.makedirs(args.output_dir, exist_ok=True)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    import torch
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    metadata = {
        "feature_dim": int(D),
        "hidden": args.hidden,
        "metrics": [m for m, _ in METRICS],
        "best_high_band_rho": float(best_high),
        "n_train": int((~hm).sum()),
        "n_holdout": int(hm.sum()),
        "include_seqstat": args.include_seqstat,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train_esm2] saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
