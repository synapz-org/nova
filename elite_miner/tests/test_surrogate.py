"""Tests for SurrogateScorer + training pipeline.

Uses synthetic labels (computed from a known function of the molecule features)
to verify the LightGBM training learns the signal — no real Boltz2 needed.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from elite_miner.molecule.features import mol_features_batch
from elite_miner.molecule.scorer import ProxyScorer
from elite_miner.molecule.surrogate import SurrogateScorer


# A small but diverse SMILES set
SYNTH_SMILES = [
    "CCO",
    "c1ccccc1",
    "c1ccccc1C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CN1CCC[C@H]1c1cccnc1",  # nicotine
    "CN(C)CCCC1(c2cc(OC)c(OC)cc2)OCC2(C1)CCCO2",
    "CC(C)NCC(O)c1ccc(O)c(CO)c1",  # salbutamol
    "Oc1ccc(C[C@H](N)C(=O)O)cc1",  # tyrosine
    "OC(=O)c1ccccc1O",  # salicylic acid
    "CCOC(=O)c1ccccc1",
    "Cc1ccccc1",
    "c1ccc(N(c2ccccc2)c2ccccc2)cc1",
    "CCCCCCCC(=O)O",
    "OCC(O)C(O)C(O)C(O)CO",  # sorbitol
    "C[C@H](N)C(=O)O",  # alanine
    "NC(N)=N",  # guanidine
    "CC(C)(C)c1ccc(O)cc1",  # BHT-precursor
    "c1ccc2[nH]ccc2c1",  # indole
    "c1cncc(C2=NCCN2)c1",
    "CC(C)C(=O)O",
    "Nc1ccc(N=Nc2ccc(S(=O)(=O)O)cc2)cc1",
    "CCN(CC)CC",
    "c1ccc2c(c1)oc1ccccc12",  # dibenzofuran
    "OC1CCCCC1",
]


def _synthetic_label(mol_feats: np.ndarray, ha_index: int = 2050) -> float:
    """Deterministic function of mol features used as 'ground truth' for training.

    We use a few specific feature indices to ensure the model has to learn
    a real combination. ha_index is into the descriptor portion (the
    heavy-atom-count descriptor lives in there).
    """
    # ECFP bits 100, 250, 500 — favor certain substructures
    bits = mol_feats[100] + 0.5 * mol_feats[250] - 0.3 * mol_feats[500]
    # MolLogP (Crippen) is at position ECFP_BITS + RDKIT_DESCRIPTOR_NAMES.index('MolLogP')=12
    # Just use a known descriptor index for stability
    logp = mol_feats[2048 + 12]  # MolLogP
    return bits + 0.4 * logp


def _build_synthetic_labels_df(target: str = "Q_SYNTH") -> pd.DataFrame:
    feats, kept = mol_features_batch(SYNTH_SMILES)
    rng = np.random.default_rng(0)
    rows = []
    for i, smi in enumerate([SYNTH_SMILES[k] for k in kept]):
        v = feats[i]
        numerator = float(_synthetic_label(v))
        # Add 5% noise
        numerator += rng.normal(0, 0.05)
        ha = int(v[2048 + 2])  # HeavyAtomCount descriptor
        pb = 0.5 + numerator * 0.3
        pv = 0.5 - numerator * 0.3
        rows.append({
            "smiles": smi,
            "name": f"synth:{i}",
            "heavy_atoms": ha if ha > 0 else 5,
            "target": target,
            "pb": pb,
            "pv": pv,
            "raw_score": numerator / max(ha, 1),
            "run_id": "synth",
            "timestamp": "2026-05-15T00:00:00Z",
        })
    return pd.DataFrame(rows)


def test_surrogate_falls_back_to_proxy_when_no_model(tmp_path):
    s = SurrogateScorer(str(tmp_path / "nonexistent"))
    assert not s.is_ready
    assert "not found" in s.fallback_reason
    # Should still produce valid ScoredMolecule via proxy
    result = s.score("rxn:1:1:2", "CCO")
    assert result is not None
    assert result.smiles == "CCO"


def test_synthetic_training_learns_signal(tmp_path):
    """Train surrogate on synthetic labels; verify it predicts better than random."""
    # Replicate the labels several times with small noise variations to give
    # the model enough training data
    base_df = _build_synthetic_labels_df()
    parts = []
    rng = np.random.default_rng(0)
    for k in range(10):
        df_k = base_df.copy()
        df_k["pb"] = df_k["pb"] + rng.normal(0, 0.02, len(df_k))
        df_k["pv"] = df_k["pv"] + rng.normal(0, 0.02, len(df_k))
        parts.append(df_k)
    df = pd.concat(parts, ignore_index=True)

    # Write a parquet file the training script can load
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    df.to_parquet(labels_dir / "synth.parquet")

    # Pre-cache the protein features so the trainer can find them
    # (sequence path uses 'seqstat' fallback)
    from elite_miner.protein.features import target_features
    target_features("Q_SYNTH", sequence="MKVLLAVLLATAATALAQQNTCEPVCDPLQCDLAVPVCNTLD")

    output_dir = tmp_path / "model"

    # Run training as if from CLI
    from argparse import Namespace
    from elite_miner.scripts.train_surrogate import train

    args = Namespace(
        labels_glob=str(labels_dir / "*.parquet"),
        output_dir=str(output_dir),
        target="Q_SYNTH",
        no_protein=False,
        holdout_frac=0.2,
        seed=42,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=16,
        min_child_samples=2,
        early_stopping_rounds=20,
    )
    rho = train(args)
    # Synthetic signal-to-noise is high → expect strong rho
    assert rho > 0.3, f"surrogate should learn synthetic signal, but rho={rho:.3f}"

    # Verify files
    assert (output_dir / "model.txt").exists()
    assert (output_dir / "holdout_metrics.json").exists()
    with open(output_dir / "holdout_metrics.json") as f:
        metrics = json.load(f)
    assert metrics["spearman_rho"] == rho


def test_surrogate_loads_trained_model(tmp_path):
    """Round-trip: train, then load via SurrogateScorer and score."""
    base_df = _build_synthetic_labels_df()
    parts = [base_df.copy() for _ in range(8)]
    df = pd.concat(parts, ignore_index=True)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    df.to_parquet(labels_dir / "synth.parquet")

    from elite_miner.protein.features import target_features
    target_features("Q_SYNTH", sequence="MKVLLAVLLATAATALAQQNTCEPVCDPLQCDLAVPVCNTLD")

    output_dir = tmp_path / "model"
    from argparse import Namespace
    from elite_miner.scripts.train_surrogate import train

    args = Namespace(
        labels_glob=str(labels_dir / "*.parquet"),
        output_dir=str(output_dir),
        target="Q_SYNTH",
        no_protein=False,
        holdout_frac=0.2,
        seed=42,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=16,
        min_child_samples=2,
        early_stopping_rounds=20,
    )
    train(args)

    # Load via SurrogateScorer
    target_feats = target_features("Q_SYNTH")
    scorer = SurrogateScorer(str(output_dir), target_features=target_feats)
    assert scorer.is_ready
    assert scorer.metrics is not None

    # Score the original SMILES
    candidates = [(f"x:{i}", s) for i, s in enumerate(SYNTH_SMILES[:5])]
    results = scorer.score_batch(candidates)
    assert len(results) == 5
    for r in results:
        assert r.score == r.score  # not NaN
        # Score should look like a numerator divided by HA
        assert -1.0 < r.score < 1.0, f"score {r.score:.3f} outside reasonable range"


def test_surrogate_respects_min_spearman(tmp_path):
    """If model's spearman is below threshold, scorer falls back to proxy."""
    output_dir = tmp_path / "weak_model"
    output_dir.mkdir()

    # Write a fake model + metrics with low spearman
    import lightgbm as lgb
    from elite_miner.molecule.features import feature_dim
    # Train a junk model on noise
    rng = np.random.default_rng(7)
    X_junk = rng.random((50, feature_dim() + 25)).astype(np.float32)
    y_junk = rng.random(50).astype(np.float32)
    m = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    m.fit(X_junk, y_junk)
    m.booster_.save_model(str(output_dir / "model.txt"))

    metrics = {
        "spearman_rho": 0.05,  # very low
        "top5_recall": 0.0,
        "per_ha_bucket_rho": {},
        "n_train": 50,
        "n_holdout": 5,
        "trained_at": "2026-05-15T00:00:00Z",
        "target": None,
    }
    with open(output_dir / "holdout_metrics.json", "w") as f:
        json.dump(metrics, f)

    s = SurrogateScorer(str(output_dir), min_spearman=0.5)
    assert not s.is_ready
    assert "spearman" in s.fallback_reason
