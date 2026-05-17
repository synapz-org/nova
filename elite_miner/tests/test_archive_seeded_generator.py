"""Tests for ArchiveSeededGenerator — seed from real winners, mutate 1-3 positions."""

import json
import random
import pytest

from elite_miner.nanobody.filters import NanobodyFilter, NanobodyValidityConfig
from elite_miner.nanobody.archive_seeded_generator import (
    ArchiveSeededGenerator,
    ArchiveSeededConfig,
)


SAMPLE_SEEDS = [
    # Two real archive winners (length 120)
    ("DVTLAESGGGLVQAGGSLRLSCAATGFTYSSYAMSWVRQRPGKEREWVSDITSQGDQTDYADFVKGRFTISRDNAKNTLYLQMTSLRPEDTAVYYCSKFPYRFMQTFAQHGQGTTVTVSA", 0.82),
    ("DVTLSESGGGLVQAGGSLRLSCAASGFTYSSYAMSWVRQRAGKEREWVSDVTSSGDQTDYADFVKGRFTISRDNAKQTLYLQMTSLRPEDTAVYYCSKWPYRFMKQYASHGQGTTVTVSS", 0.83),
]

SAMPLE_PSSM = {
    2: {"T": 0.5, "V": 0.25, "L": 0.16, "I": 0.04, "A": 0.03, "M": 0.02},
    100: {"Y": 0.72, "W": 0.19, "F": 0.07, "K": 0.02},
    104: {"Q": 0.40, "P": 0.29, "K": 0.21, "R": 0.06, "N": 0.02, "S": 0.01, "T": 0.01},
}


def test_generator_loads_explicit_data():
    vcfg = NanobodyValidityConfig.from_config({})
    cfg = ArchiveSeededConfig(min_mutations=1, max_mutations=2, pssm_path="<unused>", seeds_path="<unused>")
    gen = ArchiveSeededGenerator(vcfg, gen_cfg=cfg, seeds=SAMPLE_SEEDS, pssm=SAMPLE_PSSM)
    # Should have at least the 3 sample positions as variable
    assert 2 in gen.variable_positions
    assert 100 in gen.variable_positions
    assert 104 in gen.variable_positions


def test_generates_valid_sequences():
    vcfg = NanobodyValidityConfig.from_config({})
    cfg = ArchiveSeededConfig(min_mutations=1, max_mutations=2, pssm_path="<unused>", seeds_path="<unused>")
    gen = ArchiveSeededGenerator(vcfg, gen_cfg=cfg, rng=random.Random(42), seeds=SAMPLE_SEEDS, pssm=SAMPLE_PSSM)
    batch = gen.generate_batch(5)
    assert len(batch) > 0
    f = NanobodyFilter(vcfg)
    for s in batch:
        ok, reason = f.is_valid(s)
        assert ok, f"Invalid seq: {reason}\n  {s}"
        assert len(s) == 120


def test_dedup_across_batches():
    vcfg = NanobodyValidityConfig.from_config({})
    cfg = ArchiveSeededConfig(min_mutations=1, max_mutations=2, pssm_path="<unused>", seeds_path="<unused>")
    gen = ArchiveSeededGenerator(vcfg, gen_cfg=cfg, rng=random.Random(13), seeds=SAMPLE_SEEDS, pssm=SAMPLE_PSSM)
    b1 = gen.generate_batch(3)
    b2 = gen.generate_batch(3)
    assert not (set(b1) & set(b2)), "Should not produce duplicates across batches"


def test_seeds_weighted_by_iiptm():
    """Higher-iiptm seeds should be picked more often."""
    vcfg = NanobodyValidityConfig.from_config({})
    # Skewed weights — second seed much higher
    seeds = [
        ("A" * 120, 0.01),
        ("B" * 120, 100.0),  # impossible iiptm value, but biases sampling
    ]
    # Invalid sequences (no V/L/etc) but we're testing the seed selection itself
    cfg = ArchiveSeededConfig(min_mutations=0, max_mutations=0, pssm_path="<unused>", seeds_path="<unused>")
    gen = ArchiveSeededGenerator(vcfg, gen_cfg=cfg, rng=random.Random(0), seeds=seeds, pssm={})
    # Manually pick seeds 100 times to see distribution
    picks = [gen._pick_seed() for _ in range(100)]
    a_count = sum(1 for s in picks if s.startswith("A"))
    b_count = sum(1 for s in picks if s.startswith("B"))
    assert b_count > a_count * 50, f"weight bias not working: A={a_count} B={b_count}"


def test_load_from_files(tmp_path):
    """Round-trip: write JSON files, load via paths."""
    pssm_path = tmp_path / "pssm.json"
    seeds_path = tmp_path / "seeds.json"
    pssm_path.write_text(json.dumps(SAMPLE_PSSM))
    seeds_path.write_text(json.dumps([{"sequence": s, "iiptm": w} for s, w in SAMPLE_SEEDS]))

    vcfg = NanobodyValidityConfig.from_config({})
    cfg = ArchiveSeededConfig(
        min_mutations=1, max_mutations=2,
        pssm_path=str(pssm_path), seeds_path=str(seeds_path),
    )
    gen = ArchiveSeededGenerator(vcfg, gen_cfg=cfg, rng=random.Random(42))
    assert len(gen.seeds) == 2
    assert 100 in gen.variable_positions

    batch = gen.generate_batch(3)
    f = NanobodyFilter(vcfg)
    for s in batch:
        ok, _ = f.is_valid(s)
        assert ok
