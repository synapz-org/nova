"""Bootstrap surrogate training data by running real Boltz2 on stratified candidates.

Usage (on Basilica A100):
    python -m elite_miner.scripts.collect_labels \
        --target Q6P6W3 \
        --rxn-id 1 \
        --n-candidates 2000 \
        --batch-size 8 \
        --output cache/labels/Q6P6W3.parquet

Produces a parquet append-only label file matching the schema train_surrogate.py expects.
Idempotent: existing rows (by SMILES + target) are not re-scored on re-run.

Stratification: candidates are bucketed by HA (Q1-Q4) and an ECFP fingerprint cluster
(Butina at Tanimoto 0.6). Sampling is balanced across buckets to maximize coverage
of the reaction's product space — guards against the surrogate over-fitting to a
narrow mode.

Cost estimate: ~5s/candidate on A100 80GB → 2000 candidates ≈ 2.8 hours ≈ $1.40
at $0.50/hr spot pricing. Per target, per refresh.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

from elite_miner.molecule.searcher import CombinatorialSearcher
from elite_miner.molecule.filters import ValidityFilter
from elite_miner.molecule.scorer import _heavy_atom_count


def _ecfp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def butina_clusters(fps: list, cutoff: float = 0.6) -> list[int]:
    """Simple Butina clustering: assign each fp to first existing cluster within cutoff."""
    centroids: list = []
    assignments: list[int] = []
    for fp in fps:
        if fp is None:
            assignments.append(-1)
            continue
        placed = False
        for ci, cen in enumerate(centroids):
            if TanimotoSimilarity(fp, cen) >= cutoff:
                assignments.append(ci)
                placed = True
                break
        if not placed:
            centroids.append(fp)
            assignments.append(len(centroids) - 1)
    return assignments


def stratified_sample(
    rxn_id: int,
    n_candidates: int,
    config: dict,
    *,
    pool_size_multiplier: int = 5,
    rng_seed: int = 42,
) -> list[tuple[str, str]]:
    """Pull pool_size_multiplier × n_candidates raw products, filter validity,
    then bucket by HA quartile × ECFP-cluster and round-robin sample n_candidates."""
    pool_size = n_candidates * pool_size_multiplier
    searcher = CombinatorialSearcher(rxn_id)
    vfilter = ValidityFilter.from_config(config)

    print(f"[collect] sampling {pool_size} candidates from rxn:{rxn_id} ...")
    raw = searcher.generate_batch(pool_size)
    valid = vfilter.filter_batch(raw)
    print(f"[collect] {len(valid)}/{len(raw)} passed validity")

    if not valid:
        return []

    # HA quartiles
    ha_arr = np.array([_heavy_atom_count(s) for _, s in valid])
    ha_quartiles = np.percentile(ha_arr, [25, 50, 75])

    def ha_bucket(ha: int) -> int:
        if ha < ha_quartiles[0]:
            return 0
        if ha < ha_quartiles[1]:
            return 1
        if ha < ha_quartiles[2]:
            return 2
        return 3

    # ECFP clusters (capped — full Butina is O(n^2))
    fp_sample_n = min(len(valid), 1000)
    rng = np.random.default_rng(rng_seed)
    fp_indices = rng.choice(len(valid), size=fp_sample_n, replace=False)
    fps_sub = [_ecfp(valid[i][1]) for i in fp_indices]
    cluster_centroids = []
    for fp in fps_sub:
        if fp is None:
            continue
        if not any(TanimotoSimilarity(fp, cen) >= 0.6 for cen in cluster_centroids):
            cluster_centroids.append(fp)
        if len(cluster_centroids) >= 20:
            break
    print(f"[collect] derived {len(cluster_centroids)} ECFP cluster centroids")

    def cluster_of(smi: str) -> int:
        fp = _ecfp(smi)
        if fp is None:
            return 0
        for ci, cen in enumerate(cluster_centroids):
            if TanimotoSimilarity(fp, cen) >= 0.6:
                return ci
        return len(cluster_centroids)  # "other"

    # Bucket all valid into ha_bucket × cluster
    buckets: dict[tuple, list] = {}
    for name, smi in valid:
        ha = _heavy_atom_count(smi)
        key = (ha_bucket(ha), cluster_of(smi))
        buckets.setdefault(key, []).append((name, smi))

    print(f"[collect] {len(buckets)} stratification buckets")

    # Round-robin sample
    chosen: list = []
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    bucket_iters = {k: iter(rng.permutation(len(buckets[k])).tolist()) for k in bucket_keys}
    while len(chosen) < n_candidates:
        progress = False
        for k in bucket_keys:
            try:
                idx = next(bucket_iters[k])
                chosen.append(buckets[k][idx])
                progress = True
                if len(chosen) >= n_candidates:
                    break
            except StopIteration:
                continue
        if not progress:
            break

    print(f"[collect] selected {len(chosen)} stratified candidates")
    return chosen


def already_scored(parquet_path: str, target: str) -> set[str]:
    if not os.path.exists(parquet_path):
        return set()
    try:
        df = pd.read_parquet(parquet_path, columns=["smiles", "target"])
        return set(df[df["target"] == target]["smiles"].tolist())
    except Exception:
        return set()


def run_boltz2_scoring(
    candidates: list[tuple[str, str]],
    target: str,
    batch_size: int,
    config: dict,
) -> pd.DataFrame:
    """Calls Boltz2Scorer to label candidates. Returns a DataFrame matching the
    label schema."""
    from elite_miner.molecule.scorer import Boltz2Scorer

    # Disable label streaming inside the scorer — this script writes its own rows
    # explicitly via append_parquet. Without this flag, every row gets written twice.
    scorer = Boltz2Scorer(target=target, stream_labels=False)
    rows = []
    run_id = hashlib.sha256(f"{target}-{dt.datetime.now()}".encode()).hexdigest()[:12]

    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start: batch_start + batch_size]
        print(f"[collect] scoring {batch_start+1}..{batch_start+len(batch)} of {len(candidates)}")
        try:
            scored = scorer.score_batch(batch, config)
        except Exception as e:
            print(f"[collect] batch failed: {e}")
            continue
        for s in scored:
            raw = s.raw or {}
            pb = raw.get("affinity_probability_binary")
            pv = raw.get("affinity_pred_value")
            if pb is None or pv is None:
                continue
            rows.append({
                "smiles": s.smiles,
                "name": s.name,
                "heavy_atoms": s.heavy_atoms,
                "target": target,
                "pb": float(pb),
                "pv": float(pv),
                "raw_score": float(s.score),
                "run_id": run_id,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            })
    return pd.DataFrame(rows)


def append_parquet(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path)
    print(f"[collect] wrote {len(df)} total rows to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="UniProt target id")
    p.add_argument("--rxn-id", type=int, required=True)
    p.add_argument("--n-candidates", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", default=None,
                   help="parquet output path (default: cache/labels/{target}.parquet)")
    p.add_argument("--target-clip-interval", default="None",
                   help="clip_interval as 'None' or 'start,end'")
    p.add_argument("--min-heavy-atoms", type=int, default=10)
    p.add_argument("--min-rotatable-bonds", type=int, default=1)
    p.add_argument("--max-rotatable-bonds", type=int, default=10)
    p.add_argument("--banned-atoms", default="Se,Na,Fe,Zn")
    args = p.parse_args()

    output = args.output or f"cache/labels/{args.target}.parquet"

    config = {
        "min_heavy_atoms": args.min_heavy_atoms,
        "min_rotatable_bonds": args.min_rotatable_bonds,
        "max_rotatable_bonds": args.max_rotatable_bonds,
        "banned_atom_types": [a.strip() for a in args.banned_atoms.split(",") if a.strip()],
    }

    candidates = stratified_sample(args.rxn_id, args.n_candidates, config)
    if not candidates:
        print("[collect] no candidates produced — exiting")
        return

    # Skip already-scored
    seen = already_scored(output, args.target)
    fresh = [(n, s) for n, s in candidates if s not in seen]
    print(f"[collect] {len(candidates) - len(fresh)} already scored, {len(fresh)} fresh")
    if not fresh:
        print("[collect] nothing to score")
        return

    clip = None if args.target_clip_interval == "None" else tuple(
        int(x) for x in args.target_clip_interval.split(",")
    )
    boltz_config = {
        "small_molecule_target": [args.target],
        "small_molecule_target_clip_interval": [clip],
        "boltz_metric": ["affinity_probability_binary", "affinity_pred_value"],
        "combination_strategy": "heavy_atom_normalization",
        "boltz_mode": "max",
    }

    df = run_boltz2_scoring(fresh, args.target, args.batch_size, boltz_config)
    if df.empty:
        print("[collect] no successful scores produced")
        return

    append_parquet(output, df)


if __name__ == "__main__":
    main()
