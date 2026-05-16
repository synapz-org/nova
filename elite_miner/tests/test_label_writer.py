"""Tests for the online label streaming writer."""

import os
import pandas as pd
import pytest

from elite_miner.molecule.label_writer import append_labels, default_path


def test_append_creates_file(tmp_path):
    path = str(tmp_path / "labels.parquet")
    n = append_labels(path, [
        {"smiles": "CCO", "name": "x:1", "heavy_atoms": 3, "target": "Q1",
         "pb": 0.5, "pv": 0.3, "raw_score": 0.067},
    ])
    assert n == 1
    df = pd.read_parquet(path)
    assert df.shape[0] == 1
    assert "timestamp" in df.columns
    assert "run_id" in df.columns


def test_append_appends_existing(tmp_path):
    path = str(tmp_path / "labels.parquet")
    append_labels(path, [{"smiles": "A", "name": "n1", "heavy_atoms": 1,
                         "target": "Q1", "pb": 0.5, "pv": 0.3, "raw_score": 0.2}])
    append_labels(path, [{"smiles": "B", "name": "n2", "heavy_atoms": 2,
                         "target": "Q1", "pb": 0.6, "pv": 0.2, "raw_score": 0.4}])
    df = pd.read_parquet(path)
    assert df.shape[0] == 2


def test_append_empty_noop(tmp_path):
    path = str(tmp_path / "labels.parquet")
    n = append_labels(path, [])
    assert n == 0
    assert not os.path.exists(path)


def test_default_path_uses_target(tmp_path):
    p = default_path("Q6P6W3")
    assert p.endswith("Q6P6W3.parquet")
    assert "cache/labels" in p
