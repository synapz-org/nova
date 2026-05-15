"""Nanobody sequence-hash uniqueness check against Metanova/Submission-Archive."""

from __future__ import annotations

from .filters import seq_hash, normalize_seq


def _check_fn():
    """Defer import to avoid utils/__init__ heavy deps at module import time."""
    from utils.challenge import entry_unique_for_protein_hf
    return entry_unique_for_protein_hf


def is_nanobody_unique(target: str, sequence: str) -> bool:
    """True if SHA-256 hash of normalized sequence is not in archive for target."""
    h = seq_hash(normalize_seq(sequence))
    return _check_fn()(target, h, entity_type="nanobodies")


def warm_archive_cache(target: str) -> None:
    """Prefetch the sequence-hash set for target."""
    _check_fn()(target, "", entity_type="nanobodies")


def filter_unique(target: str, sequences: list[str]) -> list[str]:
    """Return only sequences whose hash is not in the archive."""
    warm_archive_cache(target)
    check = _check_fn()
    return [s for s in sequences if check(target, seq_hash(normalize_seq(s)), entity_type="nanobodies")]
