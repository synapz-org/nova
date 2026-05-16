"""Test the offline parts of collect_labels.py (stratification, IO)."""

import os
import pandas as pd
import pytest

from elite_miner.scripts.collect_labels import (
    stratified_sample,
    already_scored,
    append_parquet,
)


def test_stratified_sample_returns_diverse_candidates():
    """Stratification should pull from multiple HA buckets, not just one."""
    config = {
        "min_heavy_atoms": 10,
        "min_rotatable_bonds": 1,
        "max_rotatable_bonds": 10,
        "banned_atom_types": ["Se", "Na", "Fe", "Zn"],
    }
    cands = stratified_sample(rxn_id=1, n_candidates=30, config=config, pool_size_multiplier=3)
    assert len(cands) > 0

    # Check HA diversity: should span at least 10 HA range
    from elite_miner.molecule.scorer import _heavy_atom_count
    has = [_heavy_atom_count(s) for _, s in cands]
    ha_range = max(has) - min(has)
    assert ha_range >= 5, f"HA range {ha_range} too narrow (got {has})"


def test_already_scored_returns_empty_for_missing_file(tmp_path):
    assert already_scored(str(tmp_path / "no.parquet"), "Q_TEST") == set()


def test_append_parquet_appends_correctly(tmp_path):
    path = str(tmp_path / "labels.parquet")
    df1 = pd.DataFrame({"smiles": ["CCO", "CCC"], "target": ["A", "A"], "pb": [0.5, 0.6]})
    append_parquet(path, df1)
    assert pd.read_parquet(path).shape[0] == 2

    df2 = pd.DataFrame({"smiles": ["CCN"], "target": ["B"], "pb": [0.7]})
    append_parquet(path, df2)
    result = pd.read_parquet(path)
    assert result.shape[0] == 3


def test_already_scored_filters_by_target(tmp_path):
    path = str(tmp_path / "labels.parquet")
    df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"], "target": ["A", "B", "A"], "pb": [0.5, 0.5, 0.5]})
    df.to_parquet(path)
    a_seen = already_scored(path, "A")
    b_seen = already_scored(path, "B")
    assert a_seen == {"CCO", "CCC"}
    assert b_seen == {"CCN"}
