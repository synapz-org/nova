"""Nanobody candidate generator: template + CDR-biased mutation.

Strategy: pick a template, mutate positions concentrated in CDRs (especially
CDR3), reject candidates that fail local validity. Sample until we hit our
budget or run out of attempts.

This is a baseline — far from a competitive generator. The point for v1 is
to produce *valid* unique sequences. v2 will add (CDR grafting, IgLM/AbLang
language-model sampling, BoltzGen-guided RL).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Optional

from .filters import (
    NanobodyFilter,
    NanobodyValidityConfig,
    ALLOWED_AAS,
    normalize_seq,
    seq_hash,
)
from .templates import VHHTemplate, list_templates


# Sampling weights for picking which region to mutate.
# CDR3 is the most impactful for binding affinity → highest weight.
DEFAULT_REGION_WEIGHTS = {
    "cdr1": 0.20,
    "cdr2": 0.20,
    "cdr3": 0.55,
    "framework": 0.05,  # rare framework mutations preserve VHH-ness
}


@dataclass
class GenerationConfig:
    min_mutations: int = 1
    max_mutations: int = 8
    region_weights: dict | None = None
    max_attempts_per_candidate: int = 20  # tries before giving up on this template


class NanobodyGenerator:
    def __init__(
        self,
        validity_cfg: NanobodyValidityConfig,
        gen_cfg: Optional[GenerationConfig] = None,
        templates: Optional[list[VHHTemplate]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.filter = NanobodyFilter(validity_cfg)
        self.gen_cfg = gen_cfg or GenerationConfig()
        self.gen_cfg.region_weights = self.gen_cfg.region_weights or DEFAULT_REGION_WEIGHTS
        self.templates = templates or list_templates()
        if not self.templates:
            raise ValueError("Generator needs at least one template")
        self.rng = rng or random.Random()
        self._seen: set[str] = set()
        self._aa_choices = list(ALLOWED_AAS)

    def _pick_template(self) -> VHHTemplate:
        return self.rng.choice(self.templates)

    def _pick_region(self, t: VHHTemplate) -> tuple[int, int]:
        """Pick (start, end_exclusive) for a region to mutate, weighted by region_weights."""
        regions = [
            ("cdr1", t.cdr1_range),
            ("cdr2", t.cdr2_range),
            ("cdr3", t.cdr3_range),
            ("framework", (0, len(t.sequence))),
        ]
        weights = [self.gen_cfg.region_weights[name] for name, _ in regions]
        return self.rng.choices([r[1] for r in regions], weights=weights, k=1)[0]

    def _mutate(self, t: VHHTemplate) -> str:
        """Mutate `n` positions in selected region(s) of template sequence."""
        seq = list(t.sequence)
        n_mut = self.rng.randint(self.gen_cfg.min_mutations, self.gen_cfg.max_mutations)
        positions_used = set()
        for _ in range(n_mut):
            start, end = self._pick_region(t)
            if end <= start or start >= len(seq):
                continue
            # Pick a position in the region that we haven't used yet (allow reuse if region is small)
            for _try in range(8):
                pos = self.rng.randrange(start, min(end, len(seq)))
                if pos not in positions_used:
                    break
            positions_used.add(pos)
            current = seq[pos]
            new_aa = self.rng.choice([a for a in self._aa_choices if a != current])
            seq[pos] = new_aa
        return "".join(seq)

    def generate_one(self) -> Optional[str]:
        """Try to generate one valid unique sequence within budget. Returns None on failure."""
        for _ in range(self.gen_cfg.max_attempts_per_candidate):
            t = self._pick_template()
            seq = self._mutate(t)
            seq = normalize_seq(seq)
            h = seq_hash(seq)
            if h in self._seen:
                continue
            ok, _ = self.filter.is_valid(seq)
            if not ok:
                continue
            self._seen.add(h)
            return seq
        return None

    def generate_batch(self, batch_size: int) -> list[str]:
        """Generate up to batch_size valid unique sequences. May return fewer if budget exhausted."""
        out: list[str] = []
        budget = batch_size * self.gen_cfg.max_attempts_per_candidate
        attempts = 0
        while len(out) < batch_size and attempts < budget:
            attempts += 1
            seq = self.generate_one()
            if seq is not None:
                out.append(seq)
        return out

    def iter_samples(self) -> Iterator[str]:
        """Yield valid unique sequences forever (subject to template exhaustion)."""
        while True:
            seq = self.generate_one()
            if seq is None:
                # all templates exhausted at current mutation budget; widen
                self.gen_cfg.max_mutations = min(self.gen_cfg.max_mutations + 2, 25)
                continue
            yield seq

    def reset_seen(self) -> None:
        self._seen.clear()
