"""Nanobody scoring: proxy (offline) + BoltzGen (real inference, gated).

The validator's BoltzGen scoring produces per-(seq, target) components — a flat
dict of raw metric values (design_iiptm, plip_hbonds_refolded, etc.). The
validator then ranks across the whole submission pool and aggregates to a
rank_sum (lower is better). We can't predict rank_sum directly (it's pool-
relative), but we CAN predict raw metric values and combine them into a
pseudo-rank-sum at score time using signed normalization.

For v1 (no real inference), the proxy is a length-penalized random score.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional


# Validator's metric directions (mirrors config/boltzgen_config.yaml).
# +1 = higher is better, -1 = lower is better.
METRIC_DIRECTIONS = {
    "design_iiptm": +1,
    "design_ptm": +1,
    "design_to_target_iptm": +1,
    "min_design_to_target_pae": -1,
    "interaction_pae": -1,
    "plip_hbonds_refolded": +1,
    "plip_saltbridge_refolded": +1,
    "delta_sasa_refolded": +1,
    "liability_score": -1,
    "liability_num_violations": -1,
}


@dataclass
class ScoredNanobody:
    sequence: str
    score: float  # lower is better (matches validator's rank_sum semantics)
    raw: Optional[dict] = None  # full metric dict (when from real BoltzGen)


# The primary signal we train the surrogate to predict. BoltzGen returns 10
# metrics across 3 categories; design_iiptm (interface ipTM) is the single
# strongest binding-quality signal. Empirically: training a LightGBM to predict
# design_iiptm on a small label set (~120 rows) gives Spearman ρ ≈ 0.80
# vs ~0.19 for the naive multi-metric combined score (delta_sasa dominates).
#
# Validator's rank_sum is pool-relative and aggregates all 10 metrics. We
# can't match it directly without the pool. But design_iiptm strongly
# correlates with the metrics most relevant to binding (design_to_target_iptm,
# physical_interaction.*), so high-iiptm candidates also tend to rank well overall.
SURROGATE_TARGET_METRIC = "design_iiptm"


def combined_score_from_metrics(metrics: dict) -> float:
    """Map raw BoltzGen metrics to a 'lower-is-better' scalar.

    For Phase 3, this is just `-design_iiptm` (validator: higher iiptm = better,
    so negate to get lower-is-better). All other metrics are ignored in scoring
    but still streamed to label_writer for future training improvements.

    Returns +inf if the target metric is missing.
    """
    v = metrics.get(SURROGATE_TARGET_METRIC)
    if v is None:
        return math.inf
    return -float(v)
    return total / n  # average to keep magnitude stable across missing metrics


class ProxyNanobodyScorer:
    """Offline proxy. Penalty for length extremes (sweet spot 110-130)."""

    SWEET_LOW = 110
    SWEET_HIGH = 130

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def _length_penalty(self, length: int) -> float:
        if length < self.SWEET_LOW:
            return (self.SWEET_LOW - length) ** 2 * 0.1
        if length > self.SWEET_HIGH:
            return (length - self.SWEET_HIGH) ** 2 * 0.1
        return 0.0

    def score(self, seq: str) -> ScoredNanobody:
        noise = self.rng.uniform(0, 5)
        proxy = self._length_penalty(len(seq)) + noise
        return ScoredNanobody(sequence=seq, score=proxy)

    def score_batch(self, sequences: list[str]) -> list[ScoredNanobody]:
        return [self.score(s) for s in sequences]


class BoltzGenScorer:
    """Wraps vendored BoltzGen for miner-side scoring.

    Streams raw metrics to cache/nb_labels/{target}.parquet (free training data).
    Returns ScoredNanobody whose `score` is the combined min-is-better scalar.
    """

    def __init__(
        self,
        target: str,
        clip_interval=None,
        stream_labels: bool = True,
        labels_path: Optional[str] = None,
    ):
        self.target = target
        self.clip_interval = clip_interval
        self.stream_labels = stream_labels
        self._labels_path = labels_path
        self._wrapper = None  # lazy

    def _ensure_loaded(self):
        if self._wrapper is not None:
            return
        # boltzgen is pip-installed from external_tools/boltzgen (src layout),
        # so the module is just `boltzgen.boltzgen_wrapper`, not `external_tools.*`
        from boltzgen.boltzgen_wrapper import BoltzgenWrapper
        self._wrapper = BoltzgenWrapper()

    def score_batch(self, sequences: list[str], subnet_config: dict) -> list[ScoredNanobody]:
        self._ensure_loaded()
        from .filters import seq_hash
        # Validator-shape input: pretend uid=0 owns all candidates.
        # Pass sequence hashes so the wrapper's seq_idx → seq map works.
        hashes = [seq_hash(s) for s in sequences]
        nanobodies_by_uid = {0: {"sequences": sequences, "hashes": hashes}}
        per_nb_components = self._wrapper.run_nanobody_inference(nanobodies_by_uid, subnet_config)

        out: list[ScoredNanobody] = []
        label_rows: list[dict] = []
        # per_nanobody_components keyed [uid][seq][target] — same bug class as Boltz2
        uid_components = (per_nb_components or {}).get(0, {})
        for seq in sequences:
            seq_components = uid_components.get(seq, {})
            target_components = seq_components.get(self.target, {})
            if not target_components:
                out.append(ScoredNanobody(sequence=seq, score=math.inf))
                continue
            score = combined_score_from_metrics(target_components)
            out.append(ScoredNanobody(sequence=seq, score=score, raw=target_components))
            label_rows.append((seq, target_components))

        if self.stream_labels and label_rows:
            try:
                from .label_writer import append_labels, default_path, make_label_row
                path = self._labels_path or default_path(self.target)
                rows = [make_label_row(s, self.target, m) for s, m in label_rows]
                append_labels(path, rows)
            except Exception:
                pass

        return out


def rank(scored: list[ScoredNanobody]) -> list[ScoredNanobody]:
    """Sort ascending by score (lower is better)."""
    return sorted(scored, key=lambda s: s.score)
