"""Train a LightGBM surrogate for BoltzGen from collected nanobody labels.

Usage:
    python -m elite_miner.scripts.train_nb_surrogate \
        --labels-glob 'cache/nb_labels/*.parquet' \
        --output-dir models/nb_surrogate_2026-05-16 \
        --target Q9NZQ7

Labels schema (parquet):
    sequence, target, seq_length,
    {metric_name}: float (one per BoltzGen metric),
    run_id, timestamp

The surrogate predicts the *combined* min-is-better scalar (computed in
nanobody.scorer.combined_score_from_metrics), not the individual metrics
or pool-relative rank_sum.
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

from elite_miner.nanobody.features import nb_features_batch, feature_dim
from elite_miner.nanobody.scorer import combined_score_from_metrics, METRIC_DIRECTIONS
from elite_miner.nanobody.surrogate import NanobodySurrogateMetrics
from elite_miner.protein.features import target_features as protein_target_features


def load_labels(labels_glob: str, target: Optional[str] = None) -> pd.DataFrame:
    paths = sorted(glob.glob(labels_glob))
    if not paths:
        raise FileNotFoundError(f"No label files match {labels_glob}")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    required_cols = {"sequence", "target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Labels missing columns: {missing}")
    if target is not None:
        df = df[df["target"] == target].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No labels for target={target}")

    # Compute the combined label from raw metrics
    def row_to_combined(row):
        metrics = {m: row.get(m) for m in METRIC_DIRECTIONS if m in row.index}
        # pd NaN passes None-check in combined_score_from_metrics? Only None is filtered;
        # treat NaN as None
        clean = {k: (None if pd.isna(v) else v) for k, v in metrics.items()}
        return combined_score_from_metrics(clean)

    df["combined_score"] = df.apply(row_to_combined, axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["combined_score", "seq_length"])
    df = df[df["seq_length"] > 0].reset_index(drop=True)
    return df


def stratified_split(df: pd.DataFrame, holdout_frac: float = 0.15, seed: int = 42):
    """Stratify by length quartile × combined_score quartile."""
    df = df.copy()
    df["len_bucket"] = pd.qcut(df["seq_length"], q=4, duplicates="drop").astype(str)
    df["score_bucket"] = pd.qcut(df["combined_score"], q=4, duplicates="drop").astype(str)
    df["strat_key"] = df["len_bucket"] + "|" + df["score_bucket"]

    rng = np.random.default_rng(seed)
    holdout_idx = []
    for _, group in df.groupby("strat_key"):
        n_holdout = max(1, int(len(group) * holdout_frac))
        holdout_idx.extend(rng.choice(group.index.values, size=n_holdout, replace=False))
    holdout_idx = sorted(set(holdout_idx))
    holdout_mask = df.index.isin(holdout_idx)
    return df[~holdout_mask].reset_index(drop=True), df[holdout_mask].reset_index(drop=True)


def build_features(df: pd.DataFrame, use_protein: bool = True) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_nb, kept = nb_features_batch(df["sequence"].tolist())
    df = df.iloc[kept].reset_index(drop=True)
    if use_protein:
        prot_feats = {}
        rows = []
        for _, r in df.iterrows():
            tgt = r["target"]
            if tgt not in prot_feats:
                prot_feats[tgt] = protein_target_features(tgt, sequence=None)
            rows.append(prot_feats[tgt])
        X_prot = np.stack(rows)
        X = np.concatenate([X_nb, X_prot], axis=1)
    else:
        X = X_nb
    y = df["combined_score"].to_numpy(dtype=np.float32)
    return X, y, df


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x, y)
    return float(rho) if not np.isnan(rho) else 0.0


def top_k_recall(true_scores: np.ndarray, pred_scores: np.ndarray, frac: float = 0.10) -> float:
    """Both arrays are 'lower is better'. What fraction of true top-N is in pred top-N?"""
    n = len(true_scores)
    if n == 0:
        return 0.0
    k = max(1, int(n * frac))
    true_top = set(np.argsort(true_scores)[:k])    # lowest values
    pred_top = set(np.argsort(pred_scores)[:k])
    return len(true_top & pred_top) / k


def train(args):
    df = load_labels(args.labels_glob, target=args.target)
    print(f"[nb-train] loaded {len(df)} labeled rows; targets={df['target'].nunique()}")
    train_df, holdout_df = stratified_split(df, holdout_frac=args.holdout_frac, seed=args.seed)
    print(f"[nb-train] train={len(train_df)} holdout={len(holdout_df)}")

    use_protein = not args.no_protein
    X_train, y_train, train_df = build_features(train_df, use_protein=use_protein)
    X_hold, y_hold, holdout_df = build_features(holdout_df, use_protein=use_protein)
    print(f"[nb-train] feature dim={X_train.shape[1]}")

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
        eval_set=[(X_hold, y_hold)],
        callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)],
    )

    pred_hold = model.predict(X_hold)
    rho = spearman(y_hold, pred_hold)
    r10 = top_k_recall(y_hold, pred_hold, frac=0.10)
    print(f"[nb-train] holdout spearman={rho:.3f} top10%-recall={r10:.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.booster_.save_model(os.path.join(args.output_dir, "model.txt"))
    metrics = NanobodySurrogateMetrics(
        spearman_rho=rho,
        top_decile_recall=r10,
        n_train=len(train_df),
        n_holdout=len(holdout_df),
        trained_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        target=args.target,
    )
    metrics.save(os.path.join(args.output_dir, "holdout_metrics.json"))

    with open(os.path.join(args.output_dir, "feature_version.json"), "w") as f:
        json.dump({"nb_feature_dim": feature_dim(), "use_protein": use_protein}, f)

    print(f"[nb-train] wrote model to {args.output_dir}")
    return rho


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-glob", default="cache/nb_labels/*.parquet")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target", default=None)
    p.add_argument("--no-protein", action="store_true")
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=32)
    p.add_argument("--min-child-samples", type=int, default=10)
    p.add_argument("--early-stopping-rounds", type=int, default=30)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
