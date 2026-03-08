"""Tests for ProxyScorer."""

from elite_miner.scorer import ProxyScorer


def test_rank_by_proxy():
    scored = [
        ("mol_a", "CCO", 0.8, 3),   # proxy = 0.267
        ("mol_b", "CCCCC", 0.9, 5), # proxy = 0.180
        ("mol_c", "CC", 0.5, 2),    # proxy = 0.250
    ]
    ranked = ProxyScorer.rank_by_proxy(scored)
    assert ranked[0][0] == "mol_a"
    assert ranked[1][0] == "mol_c"
    assert ranked[2][0] == "mol_b"


def test_rank_by_proxy_values():
    scored = [("mol_a", "CCO", 0.8, 3)]
    ranked = ProxyScorer.rank_by_proxy(scored)
    assert abs(ranked[0][4] - 0.8 / 3) < 1e-9


def test_rank_by_proxy_zero_heavy_atoms():
    scored = [("mol_a", "CCO", 0.8, 0), ("mol_b", "CC", 0.5, 2)]
    ranked = ProxyScorer.rank_by_proxy(scored)
    assert ranked[0][0] == "mol_b"
    assert ranked[1][4] == float("-inf")


def test_rank_with_antitarget():
    molecules = [
        ("mol_a", "CCO", 0.8, 0.2, 3),   # (0.8-0.9*0.2)/3 = 0.207
        ("mol_b", "CCCCC", 0.9, 0.8, 5), # (0.9-0.9*0.8)/5 = 0.036
    ]
    ranked = ProxyScorer.rank_with_antitarget(molecules, antitarget_weight=0.9)
    assert ranked[0][0] == "mol_a"
    assert ranked[1][0] == "mol_b"


def test_rank_with_antitarget_values():
    molecules = [("mol_a", "CCO", 0.8, 0.2, 3)]
    ranked = ProxyScorer.rank_with_antitarget(molecules, antitarget_weight=0.9)
    expected = (0.8 - 0.9 * 0.2) / 3
    assert abs(ranked[0][5] - expected) < 1e-9


def test_rank_with_antitarget_zero_heavy_atoms():
    molecules = [("mol_a", "CCO", 0.8, 0.2, 0)]
    ranked = ProxyScorer.rank_with_antitarget(molecules)
    assert ranked[0][5] == float("-inf")
