"""Nanobody scoring: proxy (offline) + BoltzGen (real inference, gated).

The validator's BoltzGen scoring produces per-(seq, target) components with
confidence_rank_sum / physical_interaction_rank_sum / developability_rank_sum,
combined to a final score via min-max weighted rank. Lower is better.

For v1 (no real inference), the proxy is a length-normalized random score
so we still produce a deterministic ranking. The competitive edge in v1
comes entirely from the validity filter passing more candidates through.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoredNanobody:
    sequence: str
    score: float  # lower is better (matches validator's rank_sum semantics)
    raw: Optional[dict] = None  # full component dict when from real BoltzGen


class ProxyNanobodyScorer:
    """Offline proxy for v1. Score ~ -length + noise to weakly prefer
    moderate-length sequences that exercise CDR3 surface area.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def score(self, seq: str) -> ScoredNanobody:
        # Pseudo rank_sum: lower is better. Add noise so equal-length sequences
        # don't tie, which would let block_submitted tiebreaker dominate.
        proxy = -len(seq) + self.rng.uniform(0, 5)
        return ScoredNanobody(sequence=seq, score=proxy)

    def score_batch(self, sequences: list[str]) -> list[ScoredNanobody]:
        return [self.score(s) for s in sequences]


class BoltzGenScorer:
    """Wraps vendored BoltzGen for miner-side scoring.

    Lazy-imports the wrapper so this module is importable without the heavy
    boltzgen deps installed. Activate via --use-inference at the run.py level.
    """

    def __init__(self, target: str, clip_interval=None):
        self.target = target
        self.clip_interval = clip_interval
        self._wrapper = None  # lazy

    def _ensure_loaded(self):
        if self._wrapper is not None:
            return
        from external_tools.boltzgen.boltzgen_wrapper import BoltzgenWrapper
        self._wrapper = BoltzgenWrapper()

    def score_batch(self, sequences: list[str], subnet_config: dict) -> list[ScoredNanobody]:
        """Run BoltzGen on a batch of candidate sequences against self.target."""
        self._ensure_loaded()
        # Validator-shape input: pretend one uid=0 owns all our candidates
        nanobodies_by_uid = {0: {"sequences": sequences, "hashes": [str(i) for i in range(len(sequences))]}}
        per_nb_components = self._wrapper.run_nanobody_inference(nanobodies_by_uid, subnet_config)

        out: list[ScoredNanobody] = []
        uid_components = (per_nb_components or {}).get(0, {})
        for i, seq in enumerate(sequences):
            seq_components = uid_components.get(str(i), {})
            target_components = seq_components.get(self.target, {})
            if not target_components:
                out.append(ScoredNanobody(sequence=seq, score=math.inf))
                continue
            # Conservative: use confidence_rank_sum if available, else sum of all rank_sums
            confidence_rs = target_components.get("confidence_rank_sum")
            physical_rs = target_components.get("physical_interaction_rank_sum")
            developability_rs = target_components.get("developability_rank_sum")
            parts = [v for v in (confidence_rs, physical_rs, developability_rs) if v is not None]
            score = sum(parts) if parts else math.inf
            out.append(ScoredNanobody(sequence=seq, score=score, raw=target_components))
        return out


def rank(scored: list[ScoredNanobody]) -> list[ScoredNanobody]:
    """Sort ascending by score (lower rank_sum is better; validator boltzgen_rank_mode='min')."""
    return sorted(scored, key=lambda s: s.score)
