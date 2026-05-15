"""Tests for the template-mutator nanobody generator."""

import random

from elite_miner.nanobody.filters import NanobodyFilter, NanobodyValidityConfig
from elite_miner.nanobody.generator import NanobodyGenerator, GenerationConfig
from elite_miner.nanobody.templates import TEMPLATES


def test_all_templates_pass_validity():
    f = NanobodyFilter(NanobodyValidityConfig.from_config({}))
    for tmpl in TEMPLATES:
        ok, reason = f.is_valid(tmpl.sequence)
        assert ok, f"template {tmpl.name} fails validator: {reason}"


def test_generator_produces_valid_unique_sequences():
    vcfg = NanobodyValidityConfig.from_config({})
    rng = random.Random(7)
    gen = NanobodyGenerator(vcfg, rng=rng)
    batch = gen.generate_batch(20)
    assert len(batch) > 0

    # All should pass validity
    f = NanobodyFilter(vcfg)
    for seq in batch:
        ok, reason = f.is_valid(seq)
        assert ok, f"generated sequence fails: {reason}\n  seq: {seq}"

    # All unique
    assert len(set(batch)) == len(batch)


def test_generator_respects_seen_dedup():
    vcfg = NanobodyValidityConfig.from_config({})
    rng = random.Random(13)
    gen = NanobodyGenerator(vcfg, rng=rng)
    b1 = gen.generate_batch(10)
    b2 = gen.generate_batch(10)
    assert not (set(b1) & set(b2))


def test_reset_seen_allows_resampling():
    vcfg = NanobodyValidityConfig.from_config({})
    rng = random.Random(31)
    gen = NanobodyGenerator(vcfg, gen_cfg=GenerationConfig(min_mutations=1, max_mutations=2), rng=rng)
    gen.generate_batch(5)
    initial_seen = len(gen._seen)
    gen.reset_seen()
    assert len(gen._seen) == 0
