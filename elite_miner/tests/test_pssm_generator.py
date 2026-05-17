"""Tests for PSSM-guided nanobody generator."""

import json
import os
import random
from pathlib import Path

import pytest

from elite_miner.nanobody.filters import NanobodyFilter, NanobodyValidityConfig
from elite_miner.nanobody.pssm_generator import (
    PSSMGenerator,
    PSSMGenerationConfig,
    DEFAULT_CONSENSUS_Q9NZQ7,
)


@pytest.fixture
def synthetic_pssm(tmp_path):
    """Build a tiny synthetic PSSM with a few variable positions."""
    pssm = {
        100: {"F": 0.7, "Y": 0.2, "W": 0.1},
        104: {"Q": 0.5, "P": 0.3, "K": 0.2},
        108: {"Q": 0.6, "T": 0.2, "S": 0.2},
    }
    p = tmp_path / "pssm.json"
    p.write_text(json.dumps(pssm))
    return str(p)


def test_pssm_generator_loads(synthetic_pssm):
    vcfg = NanobodyValidityConfig.from_config({})
    cfg = PSSMGenerationConfig(min_mutations=1, max_mutations=2, pssm_path=synthetic_pssm)
    gen = PSSMGenerator(vcfg, gen_cfg=cfg)
    assert len(gen.variable_positions) == 3


def test_pssm_generator_produces_valid_seqs(synthetic_pssm):
    vcfg = NanobodyValidityConfig.from_config({})
    cfg = PSSMGenerationConfig(min_mutations=1, max_mutations=3, pssm_path=synthetic_pssm)
    gen = PSSMGenerator(vcfg, gen_cfg=cfg, rng=random.Random(42))
    batch = gen.generate_batch(5)
    assert len(batch) > 0
    f = NanobodyFilter(vcfg)
    for s in batch:
        ok, reason = f.is_valid(s)
        assert ok, f"PSSM generated invalid: {reason}\n  {s}"
        assert len(s) == len(DEFAULT_CONSENSUS_Q9NZQ7)


def test_pssm_generator_dedupes(synthetic_pssm):
    vcfg = NanobodyValidityConfig.from_config({})
    cfg = PSSMGenerationConfig(min_mutations=1, max_mutations=3, pssm_path=synthetic_pssm)
    gen = PSSMGenerator(vcfg, gen_cfg=cfg, rng=random.Random(13))
    b1 = gen.generate_batch(5)
    b2 = gen.generate_batch(5)
    assert not (set(b1) & set(b2)), "Should not produce duplicates across batches"


def test_pssm_generator_explicit_pssm():
    """Pass pssm dict directly (avoid file IO)."""
    vcfg = NanobodyValidityConfig.from_config({})
    pssm = {100: {"F": 0.5, "Y": 0.5}, 104: {"Q": 0.5, "P": 0.5}}
    cfg = PSSMGenerationConfig(min_mutations=1, max_mutations=2, pssm_path="<unused>")
    gen = PSSMGenerator(vcfg, gen_cfg=cfg, pssm=pssm)
    assert 100 in gen.variable_positions
    assert 104 in gen.variable_positions
