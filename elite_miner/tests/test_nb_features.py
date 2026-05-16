"""Tests for nanobody featurization."""

import numpy as np
import pytest

from elite_miner.nanobody.features import (
    nb_features,
    nb_features_batch,
    feature_dim,
)


def test_feature_dim_consistency():
    seq = "EVELLASGGDLVQPGGSLRLSCAASGFTFSTYAMSWVRQAPGKGLERVSRVNQNGGTTTYADAMKGRFTISRDNAKNTLYLQMINVKPEDTAIYYCARWDGGSWSTDPWGRGTLVTVS"
    v = nb_features(seq)
    assert v is not None
    assert v.shape == (feature_dim(),)
    assert v.dtype == np.float32


def test_returns_none_for_invalid_smiles():
    assert nb_features("") is None
    # Invalid AAs
    assert nb_features("EVQLXBZ") is None


def test_batch_drops_invalids():
    valid_seq = "EVELLASGGDLVQPGGSLRLSCAASGFTFSTYAMSWVRQAPGKGLERVSRVNQNGGTTTYADAMKGRFTISRDNAKNTLYLQMINVKPEDTAIYYCARWDGGSWSTDPWGRGTLVTVS"
    seqs = [valid_seq, "EVQLXBZ", "", valid_seq[:50]]
    mat, kept = nb_features_batch(seqs)
    # First valid sequence, possibly 4th (short seq passes basic AA check)
    assert 0 in kept
    assert 1 not in kept  # invalid AAs
    assert 2 not in kept  # empty
    assert mat.shape[0] == len(kept)
    assert mat.shape[1] == feature_dim()


def test_features_deterministic():
    seq = "EVELLASGGDLVQPGGSLRLSCAASGFTFSTYAMSWVRQAPGKGLERVSRVNQNGGTTTYADAMKGRFTISRDNAKNTLYLQMINVKPEDTAIYYCARWDGGSWSTDPWGRGTLVTVS"
    a = nb_features(seq)
    b = nb_features(seq)
    assert np.array_equal(a, b)


def test_length_feature_at_correct_index():
    """The length feature should match len(sequence)."""
    seq = "EVELLASGGDLVQPGGSLRLSCAASGFTFSTYAMSWVRQAPGKGLERVSRVNQNGGTTTYADAMKGRFTISRDNAKNTLYLQMINVKPEDTAIYYCARWDGGSWSTDPWGRGTLVTVS"
    v = nb_features(seq)
    # Layout: 20 AA freq + 5 physicochemical + 8 structural; index 25 is the length
    assert int(v[25]) == len(seq)
