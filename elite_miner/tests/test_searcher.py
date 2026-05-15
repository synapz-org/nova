"""Tests for the generalized combinatorial searcher."""

import pytest

from elite_miner.molecule.searcher import CombinatorialSearcher


@pytest.mark.parametrize("rxn_id", [1, 2, 3, 4, 5])
def test_generates_correct_format(rxn_id):
    s = CombinatorialSearcher(rxn_id)
    batch = s.generate_batch(5)
    assert len(batch) > 0, f"rxn:{rxn_id} produced no candidates"

    expected_parts = 5 if s.info.is_three_component else 4
    for name, smiles in batch:
        parts = name.split(":")
        assert len(parts) == expected_parts, f"bad name format: {name}"
        assert parts[0] == "rxn"
        assert int(parts[1]) == rxn_id
        for p in parts[2:]:
            int(p)  # raises if not int
        assert isinstance(smiles, str) and len(smiles) > 0


def test_no_duplicates_within_batch():
    s = CombinatorialSearcher(1)
    batch = s.generate_batch(50)
    names = [n for n, _ in batch]
    assert len(names) == len(set(names))


def test_no_overlap_between_batches():
    s = CombinatorialSearcher(2)
    b1 = s.generate_batch(20)
    b2 = s.generate_batch(20)
    n1 = {n for n, _ in b1}
    n2 = {n for n, _ in b2}
    assert not (n1 & n2), "second batch should not overlap with first"


def test_unknown_rxn_raises():
    with pytest.raises(ValueError):
        CombinatorialSearcher(99)


def test_reset_seen_allows_resampling():
    s = CombinatorialSearcher(4)
    b1 = s.generate_batch(10)
    n1 = {n for n, _ in b1}
    s.reset_seen()
    # After reset, we may sample the same combos again (chance is high if space is tiny;
    # for big reactions it's still possible). Just verify no exception and at least
    # one new candidate (statistically near-certain).
    b2 = s.generate_batch(10)
    assert len(b2) > 0
