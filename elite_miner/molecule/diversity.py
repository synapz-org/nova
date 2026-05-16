"""Mode-collapse guard: track ECFP4 fingerprint similarity across submissions.

When the surrogate over-fits, top-K candidates drift toward a single scaffold;
we submit the same compound (or near-duplicates) week after week. Validator's
uniqueness check rejects exact duplicates by InChIKey but not near-duplicates,
so we still submit but earn nothing useful.

We track the last N submitted SMILES, compute pairwise Tanimoto, and flag when
median similarity exceeds a threshold. The flag is a signal for run.py to
fall back to ProxyScorer until a retrain.
"""

from __future__ import annotations

import statistics
from collections import deque
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity


class DiversityTracker:
    """Sliding window of submitted SMILES → median pairwise Tanimoto."""

    def __init__(self, window_size: int = 20, similarity_threshold: float = 0.85):
        self.window_size = window_size
        self.threshold = similarity_threshold
        self._fps: deque = deque(maxlen=window_size)
        self._smiles: deque = deque(maxlen=window_size)

    def record(self, smiles: str) -> None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        self._fps.append(fp)
        self._smiles.append(smiles)

    def median_similarity(self) -> Optional[float]:
        """None if fewer than 2 fingerprints stored."""
        if len(self._fps) < 2:
            return None
        sims: list[float] = []
        for i in range(len(self._fps)):
            for j in range(i + 1, len(self._fps)):
                sims.append(TanimotoSimilarity(self._fps[i], self._fps[j]))
        return statistics.median(sims) if sims else None

    def is_collapsed(self) -> bool:
        """True if the recent submissions look like mode collapse."""
        med = self.median_similarity()
        return med is not None and med >= self.threshold

    def reset(self) -> None:
        self._fps.clear()
        self._smiles.clear()
