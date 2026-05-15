"""End-to-end pipeline test: search → filter → score → rank (no network, no GPU)."""

import random

from elite_miner.molecule import (
    CombinatorialSearcher,
    ValidityFilter as MolFilter,
    ProxyScorer as MolProxyScorer,
    rank as mol_rank,
)
from elite_miner.nanobody import (
    NanobodyFilter,
    NanobodyValidityConfig,
    NanobodyGenerator,
    ProxyNanobodyScorer,
    rank as nb_rank,
)


def test_molecule_pipeline_all_reactions():
    for rxn_id in [1, 2, 3, 4, 5]:
        searcher = CombinatorialSearcher(rxn_id)
        vf = MolFilter.from_config({
            "min_heavy_atoms": 10,
            "min_rotatable_bonds": 1,
            "max_rotatable_bonds": 10,
            "banned_atom_types": ["Se", "Na", "Fe", "Zn"],
        })
        scorer = MolProxyScorer(random.Random(rxn_id))

        raw = searcher.generate_batch(20)
        assert len(raw) > 0

        valid = vf.filter_batch(raw)
        assert len(valid) > 0, f"rxn:{rxn_id} produced no valid candidates"

        scored = scorer.score_batch(valid)
        ranked = mol_rank(scored)
        assert ranked[0].score >= ranked[-1].score


def test_nanobody_pipeline():
    vcfg = NanobodyValidityConfig.from_config({})
    gen = NanobodyGenerator(vcfg, rng=random.Random(11))
    scorer = ProxyNanobodyScorer(random.Random(11))

    batch = gen.generate_batch(20)
    assert len(batch) > 0

    f = NanobodyFilter(vcfg)
    for seq in batch:
        ok, _ = f.is_valid(seq)
        assert ok

    scored = scorer.score_batch(batch)
    ranked = nb_rank(scored)
    # Lower is better for nanobody scores
    assert ranked[0].score <= ranked[-1].score
