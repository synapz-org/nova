"""Train a LightGBM surrogate for Boltz2 from collected labels.

Usage:
    python -m elite_miner.scripts.train_surrogate \
        --labels-glob 'cache/labels/*.parquet' \
        --output-dir models/surrogate_2026-05-15 \
        [--target Q6P6W3]  # restrict to one target

Labels schema (parquet):
    smiles : str
    name : str (rxn:r:m1:m2[:m3])
    heavy_atoms : int
    target : str  (UniProt ID)
    pb : float   (affinity_probability_binary)
    pv : float   (affinity_pred_value)
    raw_score : float  (= (pb - pv) / heavy_atoms — the validator's combined output)
    run_id : str   (groups labels by collection run)
    timestamp : str

The surrogate predicts the *numerator* (pb - pv), not raw_score. This
removes HA as a confound and lets the predictor transfer across HA buckets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from elite_miner.molecule.features import mol_features_batch, feature_dim
from elite_miner.molecule.surrogate import SurrogateMetrics
from elite_miner.protein.features import target_features as protein_target_features


def load_labels(labels_glob: str, target: Optional[str] = None) -> pd.DataFrame:
    paths = sorted(glob.glob(labels_glob))
    if not paths:
        raise FileNotFoundError(f"No label files match {labels_glob}")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    required_cols = {"smiles", "heavy_atoms", "target", "pb", "pv"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Labels missing columns: {missing}")
    if target is not None:
        df = df[df["target"] == target].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No labels for target={target}")
    # Compute numerator label
    df["numerator"] = df["pb"] - df["pv"]
    # Drop NaN and infinite
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["numerator", "heavy_atoms"])
    df = df[df["heavy_atoms"] > 0].reset_index(drop=True)
    return df


def stratified_split(df: pd.DataFrame, holdout_frac: float = 0.10, seed: int = 42):
    """Stratified by HA quartile × numerator quartile."""
    df = df.copy()
    df["ha_bucket"] = pd.qcut(df["heavy_atoms"], q=4, duplicates="drop").astype(str)
    df["score_bucket"] = pd.qcut(df["numerator"], q=4, duplicates="drop").astype(str)
    df["strat_key"] = df["ha_bucket"] + "|" + df["score_bucket"]

    rng = np.random.default_rng(seed)
    holdout_idx = []
    for _, group in df.groupby("strat_key"):
        n_holdout = max(1, int(len(group) * holdout_frac))
        holdout_idx.extend(rng.choice(group.index.values, size=n_holdout, replace=False))
    holdout_idx = sorted(set(holdout_idx))
    holdout_mask = df.index.isin(holdout_idx)
    return df[~holdout_mask].reset_index(drop=True), df[holdout_mask].reset_index(drop=True)


def build_features(df: pd.DataFrame, use_protein: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Featurize the SMILES + (optionally) target. Drops unparseable rows."""
    X_mol, kept = mol_features_batch(df["smiles"].tolist())
    df = df.iloc[kept].reset_index(drop=True)
    if use_protein:
        prot_feats = {}
        rows = []
        for _, r in df.iterrows():
            tgt = r["target"]
            if tgt not in prot_feats:
                # We expect the cache to be warmed by collect_labels; fail loudly if not
                prot_feats[tgt] = protein_target_features(tgt, sequence=None)
            rows.append(prot_feats[tgt])
        X_prot = np.stack(rows)
        X = np.concatenate([X_mol, X_prot], axis=1)
    else:
        X = X_mol
    y = df["numerator"].to_numpy(dtype=np.float32)
    return X, y, df


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x, y)
    return float(rho) if not np.isnan(rho) else 0.0


def top_k_recall(true_scores: np.ndarray, pred_scores: np.ndarray, frac: float = 0.05) -> float:
    """What fraction of the true top-frac is in the predicted top-frac?"""
    n = len(true_scores)
    if n == 0:
        return 0.0
    k = max(1, int(n * frac))
    true_top = set(np.argsort(true_scores)[-k:])
    pred_top = set(np.argsort(pred_scores)[-k:])
    return len(true_top & pred_top) / k


def per_bucket_spearman(true_scores: np.ndarray, pred_scores: np.ndarray, ha: np.ndarray) -> dict:
    out = {}
    edges = [0, 20, 30, 40, 1000]
    labels = ["<20", "20-30", "30-40", ">40"]
    for i in range(len(edges) - 1):
        mask = (ha >= edges[i]) & (ha < edges[i + 1])
        if mask.sum() >= 5:
            out[labels[i]] = spearman(true_scores[mask], pred_scores[mask])
        else:
            out[labels[i]] = None
    return out


def train(args):
    df = load_labels(args.labels_glob, target=args.target)
    print(f"[train] loaded {len(df)} labeled rows; targets={df['target'].nunique()}")

    train_df, holdout_df = stratified_split(df, holdout_frac=args.holdout_frac, seed=args.seed)
    print(f"[train] train={len(train_df)} holdout={len(holdout_df)}")

    use_protein = not args.no_protein
    X_train, y_train, train_df = build_features(train_df, use_protein=use_protein)
    X_holdout, y_holdout, holdout_df = build_features(holdout_df, use_protein=use_protein)
    print(f"[train] feature dim={X_train.shape[1]}")

    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        random_state=args.seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_holdout, y_holdout)],
        callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)],
    )

    # Evaluate
    pred_holdout = model.predict(X_holdout)
    rho = spearman(y_holdout, pred_holdout)
    r5 = top_k_recall(y_holdout, pred_holdout, frac=0.05)
    bucket_rho = per_bucket_spearman(y_holdout, pred_holdout, holdout_df["heavy_atoms"].to_numpy())
    print(f"[train] holdout spearman={rho:.3f} top5%-recall={r5:.3f}")
    print(f"[train] per-HA-bucket spearman: {bucket_rho}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model.booster_.save_model(os.path.join(args.output_dir, "model.txt"))
    metrics = SurrogateMetrics(
        spearman_rho=rho,
        top5_recall=r5,
        per_ha_bucket_rho=bucket_rho,
        n_train=len(train_df),
        n_holdout=len(holdout_df),
        trained_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        target=args.target,
    )
    metrics.save(os.path.join(args.output_dir, "holdout_metrics.json"))

    with open(os.path.join(args.output_dir, "feature_version.json"), "w") as f:
        json.dump({"mol_feature_dim": feature_dim(), "use_protein": use_protein}, f)

    print(f"[train] wrote model to {args.output_dir}")
    return rho


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-glob", default="cache/labels/*.parquet")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target", default=None, help="restrict to single UniProt target")
    p.add_argument("--no-protein", action="store_true", help="train without protein features")
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=64)
    p.add_argument("--min-child-samples", type=int, default=20)
    p.add_argument("--early-stopping-rounds", type=int, default=30)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
