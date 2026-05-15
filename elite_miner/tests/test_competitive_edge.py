"""Quantify the validity-filter edge over the reference miner.

The reference miner samples randomly and submits the result without checking
local validity. We sample, filter, and only submit valid candidates. This test
quantifies the gap: at what rate does each approach produce validator-acceptable
candidates?
"""

import random

from elite_miner.molecule import CombinatorialSearcher, ValidityFilter as MolFilter
from elite_miner.nanobody.filters import NanobodyFilter, NanobodyValidityConfig

REPO_MOLECULE_VALIDATOR_CONFIG = {
    "min_heavy_atoms": 10,
    "min_rotatable_bonds": 1,
    "max_rotatable_bonds": 10,
    "banned_atom_types": ["Se", "Na", "Fe", "Zn"],
}


def test_molecule_filter_passes_majority_of_random_samples():
    """Sanity: random sampling has a non-trivial pass rate (so filter is a
    real but bounded edge). If random pass rate is 100% the filter does
    nothing; if it's 0% something is wrong with sampling."""
    f = MolFilter.from_config(REPO_MOLECULE_VALIDATOR_CONFIG)
    total_raw = 0
    total_valid = 0
    for rxn_id in [1, 2, 4, 5]:  # skip 3 (lower yield, slower test)
        s = CombinatorialSearcher(rxn_id, rng=random.Random(42))
        raw = s.generate_batch(40)
        valid = f.filter_batch(raw)
        total_raw += len(raw)
        total_valid += len(valid)
    pass_rate = total_valid / total_raw if total_raw else 0
    # Reasonable: filter catches some but not all
    assert 0.4 <= pass_rate <= 0.99, (
        f"Filter pass rate {pass_rate:.2%} is outside [40%, 99%] — "
        f"check sampling or filter logic"
    )


def test_nanobody_filter_catches_mutated_failures():
    """Validity filter should reject failures from heavy mutation but accept
    mild mutations from valid templates. Uses fixed seed for determinism."""
    from elite_miner.nanobody.templates import TEMPLATES
    from elite_miner.nanobody.filters import ALLOWED_AAS, normalize_seq

    f = NanobodyFilter(NanobodyValidityConfig.from_config({}))
    rng = random.Random(20251115)  # deterministic

    # Heavy random mutations (80 positions) — local filter catches some but
    # not all. The rest will fail validator's IgBLAST/developability checks
    # which we can't run locally. Document this gap.
    template = TEMPLATES[0].sequence
    heavy_fails = 0
    n_trials = 30
    for _ in range(n_trials):
        seq = list(template)
        for _ in range(80):
            i = rng.randrange(len(seq))
            seq[i] = rng.choice(list(ALLOWED_AAS - {seq[i]}))
        if not f.is_valid("".join(seq))[0]:
            heavy_fails += 1
    local_catch_rate = heavy_fails / n_trials
    # Empirically: ~30-50% of heavy-mutation candidates fail our local checks.
    # The rest pass our filter but will be rejected by IgBLAST (nativeness,
    # framework, hallmarks at Kabat-aligned positions). This is a known gap.
    assert 0.20 <= local_catch_rate <= 0.80, (
        f"Local filter catches {local_catch_rate:.0%} of heavy mutations. "
        f"Expected 20-80% — outside range suggests filter regression"
    )

    # Light mutations (2-3 positions) — should mostly pass
    light_passes = 0
    for _ in range(20):
        seq = list(template)
        for _ in range(2):
            i = rng.randrange(len(seq))
            seq[i] = rng.choice(list(ALLOWED_AAS - {seq[i]}))
        if f.is_valid("".join(seq))[0]:
            light_passes += 1
    assert light_passes >= 15, (
        f"Light mutations (2 positions) should usually pass, but only "
        f"{light_passes}/20 passed — filter may be too strict"
    )
