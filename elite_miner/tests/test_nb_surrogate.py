"""Tests for nanobody surrogate (synthetic-label training validation)."""

import json
import os
import random

import numpy as np
import pandas as pd
import pytest

from elite_miner.nanobody.features import nb_features_batch
from elite_miner.nanobody.scorer import ProxyNanobodyScorer, METRIC_DIRECTIONS
from elite_miner.nanobody.surrogate import NanobodySurrogateScorer
from elite_miner.nanobody.templates import TEMPLATES
from elite_miner.nanobody.generator import NanobodyGenerator, GenerationConfig
from elite_miner.nanobody.filters import NanobodyValidityConfig


def _build_synthetic_labels(n_per_template: int = 8, target: str = "Q_SYNTH") -> pd.DataFrame:
    """Generate sequences from each template and synthesize raw BoltzGen metrics.

    Synthetic labels are deterministic functions of features so we know there
    IS a signal to learn.
    """
    vcfg = NanobodyValidityConfig.from_config({})
    rng = np.random.default_rng(0)
    rows = []
    for i, t in enumerate(TEMPLATES):
        gen = NanobodyGenerator(
            vcfg,
            gen_cfg=GenerationConfig(min_mutations=1, max_mutations=3),
            templates=[t],
            rng=random.Random(i),
        )
        batch = gen.generate_batch(n_per_template)
        for seq in batch:
            feats, _ = nb_features_batch([seq])
            v = feats[0]
            # Synthetic signal: combine a few feature indices
            # length (idx 25), cys_count (26), AA freq[A] (idx 0), AA freq[V] (idx 18)
            signal = v[0] * 2.0 - v[18] * 1.5 + v[26] * 0.5
            # Synthesize each metric — sign-aligned to directions
            metric_values = {}
            for metric, direction in METRIC_DIRECTIONS.items():
                # +1 metrics: higher is better → make them correlate with signal
                # -1 metrics: lower is better → make them anti-correlate
                base = direction * signal + rng.normal(0, 0.05)
                metric_values[metric] = base
            row = {
                "sequence": seq,
                "target": target,
                "seq_length": len(seq),
                **metric_values,
                "run_id": "synth",
                "timestamp": "2026-05-16T00:00:00Z",
            }
            rows.append(row)
    return pd.DataFrame(rows)


def test_surrogate_falls_back_when_no_model(tmp_path):
    s = NanobodySurrogateScorer(str(tmp_path / "missing"))
    assert not s.is_ready
    assert "not found" in s.fallback_reason
    result = s.score("EVELLASGGDLVQPGGSLRLSCAASGFTFSTYAMSWVRQAPGKGLERVSRVNQNGGTTTYADAMKGRFTISRDNAKNTLYLQMINVKPEDTAIYYCARWDGGSWSTDPWGRGTLVTVS")
    assert result is not None


def test_synthetic_training_learns_signal(tmp_path):
    """Synthesize labels with known signal; verify surrogate learns it."""
    df = _build_synthetic_labels(n_per_template=10)
    # Replicate with noise so we have enough rows
    parts = []
    rng = np.random.default_rng(1)
    for _ in range(6):
        df_k = df.copy()
        for m in METRIC_DIRECTIONS:
            df_k[m] = df_k[m] + rng.normal(0, 0.03, len(df_k))
        parts.append(df_k)
    big = pd.concat(parts, ignore_index=True)

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    big.to_parquet(labels_dir / "synth.parquet")

    # Cache the protein features so trainer can find them
    from elite_miner.protein.features import target_features
    target_features("Q_SYNTH", sequence="MKVLLAVLLATAATALAQQNTCEPVCDPLQCDLAVPVCNTLD")

    output_dir = tmp_path / "model"
    from argparse import Namespace
    from elite_miner.scripts.train_nb_surrogate import train

    args = Namespace(
        labels_glob=str(labels_dir / "*.parquet"),
        output_dir=str(output_dir),
        target="Q_SYNTH",
        no_protein=False,
        holdout_frac=0.20,
        seed=42,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=16,
        min_child_samples=2,
        early_stopping_rounds=20,
    )
    rho = train(args)
    # Synthetic signal is strong → expect rho > 0.3
    assert rho > 0.3, f"surrogate should learn synthetic signal, got rho={rho:.3f}"

    assert (output_dir / "model.txt").exists()
    assert (output_dir / "holdout_metrics.json").exists()


def test_surrogate_roundtrip(tmp_path):
    """Train + load via NanobodySurrogateScorer + score new sequences."""
    df = _build_synthetic_labels(n_per_template=12)
    parts = [df.copy() for _ in range(5)]
    big = pd.concat(parts, ignore_index=True)

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    big.to_parquet(labels_dir / "synth.parquet")

    from elite_miner.protein.features import target_features
    target_features("Q_SYNTH", sequence="MKVLLAVLLATAATALAQQNTCEPVCDPLQCDLAVPVCNTLD")

    output_dir = tmp_path / "model"
    from argparse import Namespace
    from elite_miner.scripts.train_nb_surrogate import train

    args = Namespace(
        labels_glob=str(labels_dir / "*.parquet"),
        output_dir=str(output_dir),
        target="Q_SYNTH",
        no_protein=False,
        holdout_frac=0.20,
        seed=42,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=16,
        min_child_samples=2,
        early_stopping_rounds=20,
    )
    train(args)

    tgt_feats = target_features("Q_SYNTH")
    scorer = NanobodySurrogateScorer(str(output_dir), target_features=tgt_feats)
    assert scorer.is_ready

    # Score new sequences (subset of training set is fine for this round-trip check)
    sample_seqs = df["sequence"].head(5).tolist()
    results = scorer.score_batch(sample_seqs)
    assert len(results) == 5
    for r in results:
        # Should be a finite float
        assert r.score == r.score  # not NaN
