"""Nanobody featurization for the BoltzGen surrogate.

Vector = nanobody sequence features (25d seqstat or 320d ESM2)
       + target features (already exposed by elite_miner.protein.features)
       + a few hand-crafted features (length, hydrophobicity, charge, CDR3 length proxy)

The hand-crafted features are cheap, fast to compute, and capture the kinds of
sequence properties that affect BoltzGen's confidence/physical_interaction/
developability metrics — features ESM2 may not explicitly surface.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from .filters import ALLOWED_AAS, HYDROPHOBIC, normalize_seq

ALL_AAS = "ACDEFGHIKLMNPQRSTVWY"
POLAR = set("STNQH")
CHARGED_POS = set("KR")
CHARGED_NEG = set("DE")
AROMATIC = set("FYW")


def _aa_composition(seq: str) -> np.ndarray:
    """20-d AA frequency vector."""
    n = max(1, len(seq))
    counts = {aa: 0 for aa in ALL_AAS}
    for aa in seq:
        if aa in counts:
            counts[aa] += 1
    return np.array([counts[aa] / n for aa in ALL_AAS], dtype=np.float32)


def _physicochemical(seq: str) -> np.ndarray:
    """5-d bucket fractions: hydrophobic, polar, +charged, -charged, aromatic."""
    n = max(1, len(seq))
    return np.array([
        sum(1 for c in seq if c in HYDROPHOBIC) / n,
        sum(1 for c in seq if c in POLAR) / n,
        sum(1 for c in seq if c in CHARGED_POS) / n,
        sum(1 for c in seq if c in CHARGED_NEG) / n,
        sum(1 for c in seq if c in AROMATIC) / n,
    ], dtype=np.float32)


def _structural_proxies(seq: str) -> np.ndarray:
    """8-d hand-crafted features that proxy structural / developability traits:
        length, cys_count, cys_pair_separation, max_homopolymer_run,
        max_di_repeat_pairs, n_term_hydro_frac, c_term_hydro_frac, cdr3_window_charge
    """
    if not seq:
        return np.zeros(8, dtype=np.float32)

    length = len(seq)
    cys_count = seq.count("C")
    cys_positions = [i for i, aa in enumerate(seq) if aa == "C"]
    cys_pair_sep = (cys_positions[1] - cys_positions[0]) if len(cys_positions) >= 2 else 0

    # Reuse normalized seq computation for homopolymer / di-repeat
    best_run = 1
    cur = 1
    for i in range(1, length):
        if seq[i] == seq[i - 1]:
            cur += 1
            best_run = max(best_run, cur)
        else:
            cur = 1

    di_repeats = 0
    for i in range(length - 3):
        a, b = seq[i], seq[i + 1]
        if a == b:
            continue
        pairs = 1
        j = i + 2
        while j + 1 < length and seq[j] == a and seq[j + 1] == b:
            pairs += 1
            j += 2
        di_repeats = max(di_repeats, pairs)

    # N-terminal hydrophobicity (first 20 residues)
    nterm = seq[:20]
    nterm_hydro = sum(1 for c in nterm if c in HYDROPHOBIC) / max(1, len(nterm))
    # C-terminal hydrophobicity (last 20)
    cterm = seq[-20:]
    cterm_hydro = sum(1 for c in cterm if c in HYDROPHOBIC) / max(1, len(cterm))

    # CDR3 window charge: positions 95-115 (typical CDR3 range)
    cdr3_window = seq[95:115] if length >= 115 else seq[max(0, length-20):]
    cdr3_charge = (
        sum(1 for c in cdr3_window if c in CHARGED_POS)
        - sum(1 for c in cdr3_window if c in CHARGED_NEG)
    )

    return np.array([
        length, cys_count, cys_pair_sep, best_run,
        di_repeats, nterm_hydro, cterm_hydro, cdr3_charge,
    ], dtype=np.float32)


def nb_features(sequence: str) -> Optional[np.ndarray]:
    """Featurize a single nanobody sequence.

    Returns float32 vector of shape (33,) = 20 (AA freq) + 5 (physicochemical) + 8 (structural proxies).
    Returns None for invalid sequences (empty, all-invalid AAs).
    """
    seq = normalize_seq(sequence)
    if not seq or set(seq) - ALLOWED_AAS:
        # Sequences with invalid AAs are unlikely to ever pass validity anyway —
        # but if we featurize them via filter_batch, they get None and are dropped
        return None
    return np.concatenate([_aa_composition(seq), _physicochemical(seq), _structural_proxies(seq)])


def nb_features_batch(sequences: list[str]) -> tuple[np.ndarray, list[int]]:
    rows = []
    kept = []
    for i, s in enumerate(sequences):
        v = nb_features(s)
        if v is not None:
            rows.append(v)
            kept.append(i)
    if not rows:
        return np.zeros((0, feature_dim()), dtype=np.float32), []
    return np.stack(rows), kept


def feature_dim() -> int:
    return 20 + 5 + 8


def seq_cache_key(sequence: str) -> str:
    seq = normalize_seq(sequence)
    return hashlib.sha256(seq.encode()).hexdigest()[:16]
