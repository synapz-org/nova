"""Per-epoch hyperparameter variants for A/B-testing nb generation strategies.

Each epoch the miner samples a variant from VARIANTS and applies it to the
nanobody track. The variant id is logged to the submission queue, so the
labeling worker's results can be grouped by variant.

To compare variants, look at `cache/submission_labels/Q9NZQ7.parquet` joined
with `cache/submission_queue/nb.jsonl` on sequence, then `groupby variant_id`.

The current set is targeted: we know n=5000 from ArchiveSeededGenerator
1-3 muts gives high-variance real iiptm. Variants here change knobs that
might shift either the mean or the variance.

Adding a new variant: append to VARIANTS. Don't remove old ones until they're
clearly bad — labels for retired variants stay valuable as a baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class NbVariant:
    """A single hyperparameter assignment for the nb track.

    Each variant has:
    - id: short stable name. Logged into the submission queue. Don't reuse.
    - description: one-line human-readable summary.
    - fast_batch_size_nb: surrogate-only candidate pool size.
    - min_mutations / max_mutations: ArchiveSeededGenerator knobs.
    - generator: which generator to use. "archive_seeded" (default) or "pssm".

    Variants with the same generator + (min,max) but different fast_batch_size
    let us measure batch-size effect. Variants with different (min,max) let us
    measure neighborhood-width effect.
    """
    id: str
    description: str
    fast_batch_size_nb: int = 5000
    min_mutations: int = 1
    max_mutations: int = 3
    generator: str = "archive_seeded"


# Initial set. Keep small — labeling is slow (~85 min/each), so each variant
# needs several epochs of submissions to get N labels for a stable mean.
VARIANTS: list[NbVariant] = [
    NbVariant(
        id="baseline_n5000_m13",
        description="ArchiveSeededGenerator, n=5000, 1-3 mutations. The current default.",
        fast_batch_size_nb=5000,
        min_mutations=1,
        max_mutations=3,
    ),
    NbVariant(
        id="wide_n5000_m38",
        description="ArchiveSeededGenerator, n=5000, 3-8 mutations. Tests if a wider mutation neighborhood escapes the saturated competitor zone.",
        fast_batch_size_nb=5000,
        min_mutations=3,
        max_mutations=8,
    ),
    NbVariant(
        id="tight_n5000_m01",
        description="ArchiveSeededGenerator, n=5000, 0-1 mutations. Tests if staying closer to known winners has different variance.",
        fast_batch_size_nb=5000,
        min_mutations=0,
        max_mutations=1,
    ),
    NbVariant(
        id="labelseeded_n5000_m13",
        description="LabelSeededGenerator (mutates from sequences we've MEASURED at real iiptm >= 0.80), n=5000, 1-3 mutations. Tests if our offline-labeled winners give better seeds than the public archive top.",
        fast_batch_size_nb=5000,
        min_mutations=1,
        max_mutations=3,
        generator="label_seeded",
    ),
]


def pick_variant(rng: Optional[random.Random] = None) -> NbVariant:
    """Uniform-random pick. Replace with a smarter scheduler later (UCB1, etc)."""
    r = rng or random.Random()
    return r.choice(VARIANTS)


def get_variant(variant_id: str) -> Optional[NbVariant]:
    for v in VARIANTS:
        if v.id == variant_id:
            return v
    return None
