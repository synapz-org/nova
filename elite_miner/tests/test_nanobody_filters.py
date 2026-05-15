"""Tests for nanobody filter — mirrors validator's nanobody_validity stages 1-3."""

import pytest

from elite_miner.nanobody.filters import (
    NanobodyFilter,
    NanobodyValidityConfig,
    max_run_length,
    max_di_repeat_pairs,
    has_plausible_cys_pair,
    looks_like_signal_peptide,
    has_vhh_hallmarks,
    seq_hash,
    normalize_seq,
)


def _vcfg():
    return NanobodyValidityConfig.from_config({})


def test_max_run_length():
    assert max_run_length("") == 0
    assert max_run_length("AAAA") == 4
    assert max_run_length("ABCDEFG") == 1
    assert max_run_length("ABBCCCD") == 3


def test_max_di_repeat():
    assert max_di_repeat_pairs("GSGSGS") == 3
    assert max_di_repeat_pairs("GSGSGSGSGS") == 5
    assert max_di_repeat_pairs("AAAAAA") == 0  # homopolymer, not di-repeat


def test_cys_pair():
    seq = "X" * 10 + "C" + "X" * 40 + "C" + "X" * 10
    # cys pair separation = 41
    assert has_plausible_cys_pair(seq, 35, 80)
    assert not has_plausible_cys_pair(seq, 42, 80)  # min too strict
    assert not has_plausible_cys_pair(seq, 35, 40)  # max too strict


def test_signal_peptide():
    # First 12 of "AILMFWV" are all hydrophobic — should trigger
    assert looks_like_signal_peptide("AILMFWVAILMA" + "X" * 80, window=12, hydro_min=8, scan_prefix=30)
    # Sequence with no hydrophobic stretch
    assert not looks_like_signal_peptide("EVQLVESGGGLVQPGGSLRLSCAA" + "Y" * 80, window=12, hydro_min=8, scan_prefix=30)


def test_vhh_hallmarks_present():
    # Reference miner template has the [EQ]R motif in FR2 region
    ref = "EVELLASGGDLVQPGGSLRLSCAASGFTFSTYAMSWVRQAPGKGLERVSRVNQNGGTTTYADAMKGRFTISRDNAKNTLYLQMINVKPEDTAIYYCARWDGGSWSTDPWGRGTLVTVS"
    assert has_vhh_hallmarks(ref)


def test_vhh_hallmarks_absent():
    # All-A sequence has no E or Q
    assert not has_vhh_hallmarks("A" * 130)


def test_full_filter_accepts_valid_vhh():
    seq = "EVELLASGGDLVQPGGSLRLSCAASGFTFSTYAMSWVRQAPGKGLERVSRVNQNGGTTTYADAMKGRFTISRDNAKNTLYLQMINVKPEDTAIYYCARWDGGSWSTDPWGRGTLVTVS"
    f = NanobodyFilter(_vcfg())
    ok, reason = f.is_valid(seq)
    assert ok, f"valid VHH rejected: {reason}"


def test_full_filter_rejects_short():
    f = NanobodyFilter(_vcfg())
    ok, _ = f.is_valid("EVQLVESGGG")
    assert not ok


def test_full_filter_rejects_invalid_aa():
    f = NanobodyFilter(_vcfg())
    seq = "EV" + "X" * 100 + "ER" + "WGQ" * 5
    ok, reason = f.is_valid(seq)
    assert not ok
    assert "invalid" in reason.lower() or "AA" in reason


def test_seq_hash_deterministic():
    s = "EVQLVESGGGLVQ"
    assert seq_hash(s) == seq_hash(s)
    assert seq_hash(normalize_seq("  evqlvesgggl  vq  ".replace(" ", ""))) == seq_hash("EVQLVESGGGLVQ")
