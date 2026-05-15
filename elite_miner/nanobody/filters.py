"""Nanobody validity filter — mirrors neurons/validator/nanobody_validity.py
stages 1-3 (the cheap checks). Skips IgBLAST nativeness and developability
since those are heavy and require external tools.

Functions are inlined (not imported from utils.nanobodies) so this module
doesn't drag in the metanano/abnumber/anarci stack at import time. The miner
can stay light-weight.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

ALLOWED_AAS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWV")


def normalize_seq(seq: str) -> str:
    return seq.strip().upper()


def seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def max_run_length(s: str) -> int:
    if not s:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def max_di_repeat_pairs(s: str) -> int:
    best = 0
    n = len(s)
    for i in range(n - 3):
        a, b = s[i], s[i + 1]
        if a == b:
            continue
        pairs = 1
        j = i + 2
        while j + 1 < n and s[j] == a and s[j + 1] == b:
            pairs += 1
            j += 2
        if pairs > best:
            best = pairs
    return best


def has_plausible_cys_pair(seq: str, min_sep: int, max_sep: int) -> bool:
    cys_positions = [i for i, aa in enumerate(seq) if aa == "C"]
    for i in range(len(cys_positions) - 1):
        for j in range(i + 1, len(cys_positions)):
            sep = cys_positions[j] - cys_positions[i]
            if sep < min_sep:
                continue
            if sep > max_sep:
                break
            return True
    return False


def looks_like_signal_peptide(seq: str, window: int, hydro_min: int, scan_prefix: int) -> bool:
    prefix = seq[: min(len(seq), scan_prefix)]
    if len(prefix) < window:
        return False
    count = sum(1 for aa in prefix[:window] if aa in HYDROPHOBIC)
    if count >= hydro_min:
        return True
    for i in range(window, len(prefix)):
        if prefix[i - window] in HYDROPHOBIC:
            count -= 1
        if prefix[i] in HYDROPHOBIC:
            count += 1
        if count >= hydro_min:
            return True
    return False


def has_vhh_hallmarks(seq: str) -> bool:
    """Heuristic VHH FR2 hallmark check.

    The real validator check (in IgBLAST-aligned Kabat numbering) requires
    position 49 in {E,Q} and position 50 = R. CDR1 length varies, so raw
    sequence positions don't map cleanly. We use a loose heuristic: look for
    an `[EQ]R` motif anywhere in the FR2 raw-position range [40, 60].

    True VHH templates from anti-target literature reliably contain this
    motif somewhere in that window. False positives are fine — they just
    pass our pre-filter and get rejected by the real IgBLAST stage. False
    negatives (rejecting real VHHs) are what we must avoid; this heuristic
    has none across the templates we've tested.
    """
    if len(seq) < 60:
        return False
    fr2_region = seq[40:60]
    for i in range(len(fr2_region) - 1):
        if fr2_region[i] in "EQ" and fr2_region[i + 1] == "R":
            return True
    return False


@dataclass
class NanobodyValidityConfig:
    """Mirrors config.yaml `nanobody_requirements` section."""

    min_sequence_length: int = 90
    max_sequence_length: int = 150
    min_cysteines: int = 1
    cys_pair_min_separation: int = 35
    cys_pair_max_separation: int = 80
    max_homopolymer_run: int = 6
    max_di_repeat_pairs: int = 4
    reject_signal_peptides: bool = True
    sp_window: int = 12
    sp_hydro_min_in_window: int = 8
    sp_scan_prefix: int = 30
    enforce_vhh_hallmarks: bool = True

    @classmethod
    def from_config(cls, cfg: dict) -> "NanobodyValidityConfig":
        return cls(
            min_sequence_length=cfg.get("min_sequence_length", 90),
            max_sequence_length=cfg.get("max_sequence_length", 150),
            min_cysteines=cfg.get("min_cysteines", 1),
            cys_pair_min_separation=cfg.get("cys_pair_min_separation", 35),
            cys_pair_max_separation=cfg.get("cys_pair_max_separation", 80),
            max_homopolymer_run=cfg.get("max_homopolymer_run", 6),
            max_di_repeat_pairs=cfg.get("max_di_repeat_pairs", 4),
            reject_signal_peptides=cfg.get("reject_signal_peptides", True),
            sp_window=cfg.get("sp_window", 12),
            sp_hydro_min_in_window=cfg.get("sp_hydro_min_in_window", 8),
            sp_scan_prefix=cfg.get("sp_scan_prefix", 30),
            enforce_vhh_hallmarks=cfg.get("enforce_vhh_hallmarks", True),
        )


class NanobodyFilter:
    def __init__(self, cfg: NanobodyValidityConfig):
        self.cfg = cfg

    def is_valid(self, seq: str) -> tuple[bool, str]:
        """Returns (passes, reason). reason is empty on pass."""
        seq = normalize_seq(seq)

        if not (self.cfg.min_sequence_length <= len(seq) <= self.cfg.max_sequence_length):
            return False, f"length {len(seq)} not in [{self.cfg.min_sequence_length},{self.cfg.max_sequence_length}]"

        if set(seq) - ALLOWED_AAS:
            bad = set(seq) - ALLOWED_AAS
            return False, f"invalid AAs: {sorted(bad)}"

        if max_run_length(seq) > self.cfg.max_homopolymer_run:
            return False, f"homopolymer run > {self.cfg.max_homopolymer_run}"

        if max_di_repeat_pairs(seq) > self.cfg.max_di_repeat_pairs:
            return False, f"di-repeat pairs > {self.cfg.max_di_repeat_pairs}"

        if seq.count("C") < self.cfg.min_cysteines:
            return False, f"cysteine count < {self.cfg.min_cysteines}"

        if self.cfg.min_cysteines > 1 and not has_plausible_cys_pair(
            seq, self.cfg.cys_pair_min_separation, self.cfg.cys_pair_max_separation
        ):
            return False, "no plausible cysteine pair"

        if self.cfg.reject_signal_peptides and looks_like_signal_peptide(
            seq, self.cfg.sp_window, self.cfg.sp_hydro_min_in_window, self.cfg.sp_scan_prefix
        ):
            return False, "looks like signal peptide"

        if self.cfg.enforce_vhh_hallmarks and not has_vhh_hallmarks(seq):
            return False, "missing VHH FR2 hallmarks"

        return True, ""

    def filter_batch(self, sequences: list[str]) -> list[str]:
        return [s for s in sequences if self.is_valid(s)[0]]
