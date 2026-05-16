"""Mode-collapse guard for nanobody track.

Tracks recent submitted nanobody sequences; flags if pairwise sequence
identity exceeds threshold (a different kind of "near-duplicate" than for
molecules — substring overlap matters more than fingerprint similarity).

Identity metric: simple pairwise residue-match fraction over aligned positions.
For sequences of differing length, align by trimming both to the shorter.
"""

from __future__ import annotations

import statistics
from collections import deque
from typing import Optional


def _pairwise_identity(a: str, b: str) -> float:
    """Fraction of identical residues at aligned positions (truncated to min length)."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n


class NanobodyDiversityTracker:
    """Sliding window of submitted nanobodies → median pairwise identity."""

    def __init__(self, window_size: int = 10, identity_threshold: float = 0.95):
        self.window_size = window_size
        self.threshold = identity_threshold
        self._seqs: deque = deque(maxlen=window_size)

    def record(self, sequence: str) -> None:
        if sequence:
            self._seqs.append(sequence)

    def median_identity(self) -> Optional[float]:
        if len(self._seqs) < 2:
            return None
        ids: list[float] = []
        seqs = list(self._seqs)
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                ids.append(_pairwise_identity(seqs[i], seqs[j]))
        return statistics.median(ids) if ids else None

    def is_collapsed(self) -> bool:
        med = self.median_identity()
        return med is not None and med >= self.threshold

    def reset(self) -> None:
        self._seqs.clear()
