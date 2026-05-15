"""Tests for molecule scorer (proxy mode)."""

import math
import random

from elite_miner.molecule.scorer import ProxyScorer, rank, _combined_score


def test_combined_score_formula():
    # (boltz_metric[0] - boltz_metric[1]) / heavy_atoms
    assert _combined_score(0.8, 0.2, 20) == (0.8 - 0.2) / 20
    assert _combined_score(0.5, 0.5, 10) == 0.0


def test_combined_score_zero_ha():
    assert _combined_score(0.5, 0.2, 0) == -math.inf


def test_proxy_score_is_deterministic_with_seed():
    rng = random.Random(42)
    s = ProxyScorer(rng)
    a = s.score("rxn:1:1:2", "CCO")
    rng2 = random.Random(42)
    s2 = ProxyScorer(rng2)
    b = s2.score("rxn:1:1:2", "CCO")
    assert a.score == b.score


def test_rank_descending():
    rng = random.Random(0)
    s = ProxyScorer(rng)
    candidates = [
        ("rxn:1:1:1", "CCCCCCCCCC"),
        ("rxn:1:1:2", "CCCCCCCCCCCC"),
        ("rxn:1:1:3", "CCCCCCCC"),
    ]
    scored = s.score_batch(candidates)
    ranked = rank(scored)
    for i in range(len(ranked) - 1):
        assert ranked[i].score >= ranked[i + 1].score


def test_proxy_prefers_drug_like_size():
    """Proxy should NOT systematically prefer the smallest molecule (validator bug).
    Drug-like mid-range (20-40 HA) should outrank very small (5 HA) on average."""
    rng = random.Random(0)
    s = ProxyScorer(rng)
    # Score the same SMILES many times to average out noise
    tiny = "C" * 5  # 5 heavy atoms
    midrange = "c1ccccc1" * 4  # ~24 heavy atoms
    n_trials = 100
    tiny_scores = [s.score("a", tiny).score for _ in range(n_trials)]
    mid_scores = [s.score("b", midrange).score for _ in range(n_trials)]
    avg_tiny = sum(tiny_scores) / n_trials
    avg_mid = sum(mid_scores) / n_trials
    assert avg_mid > avg_tiny, (
        f"Drug-like size {midrange} (avg={avg_mid:.4f}) should outrank tiny "
        f"{tiny} (avg={avg_tiny:.4f}) — proxy is biased toward small mols"
    )
