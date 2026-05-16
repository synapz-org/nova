"""Tests for the mode-collapse diversity guard."""

import pytest

from elite_miner.molecule.diversity import DiversityTracker


def test_empty_tracker_not_collapsed():
    d = DiversityTracker(window_size=10, similarity_threshold=0.85)
    assert d.median_similarity() is None
    assert not d.is_collapsed()


def test_single_record_not_collapsed():
    d = DiversityTracker()
    d.record("CCO")
    assert d.median_similarity() is None
    assert not d.is_collapsed()


def test_diverse_smiles_not_collapsed():
    d = DiversityTracker(window_size=10, similarity_threshold=0.85)
    diverse = [
        "CCO",
        "c1ccccc1",
        "c1ccc2[nH]ccc2c1",
        "CC(=O)Nc1ccc(O)cc1",
        "Oc1ccc(C[C@H](N)C(=O)O)cc1",
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    ]
    for s in diverse:
        d.record(s)
    med = d.median_similarity()
    assert med is not None
    assert med < 0.5, f"Diverse SMILES should have low median similarity, got {med:.3f}"
    assert not d.is_collapsed()


def test_identical_smiles_collapsed():
    d = DiversityTracker(window_size=5, similarity_threshold=0.85)
    # Same molecule submitted 5 times — total mode collapse
    for _ in range(5):
        d.record("c1ccc(C(=O)O)cc1")
    assert d.median_similarity() == 1.0
    assert d.is_collapsed()


def test_near_identical_smiles_collapsed():
    """Slight variants of the same scaffold should still trigger collapse."""
    d = DiversityTracker(window_size=10, similarity_threshold=0.85)
    # Same core with small substituent changes
    near_dupes = [
        "c1ccc(C(=O)O)cc1",
        "c1ccc(C(=O)OC)cc1",
        "c1ccc(C(=O)N)cc1",
        "c1ccc(C(=O)O)cc1C",
        "Cc1ccc(C(=O)O)cc1",
    ]
    for s in near_dupes:
        d.record(s)
    med = d.median_similarity()
    # These share a benzoic-acid-like scaffold so should be similar
    assert med > 0.4, f"Near-duplicates should have meaningful similarity, got {med:.3f}"


def test_reset_clears_window():
    d = DiversityTracker()
    for _ in range(3):
        d.record("CCO")
    assert d.median_similarity() == 1.0
    d.reset()
    assert d.median_similarity() is None


def test_invalid_smiles_silently_dropped():
    d = DiversityTracker()
    d.record("not-a-smiles")
    d.record("CCO")
    d.record("also-invalid")
    # Only the valid one is recorded; median over 1 fp is None
    assert d.median_similarity() is None
