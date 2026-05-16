"""Tests for nanobody mode-collapse guard."""

from elite_miner.nanobody.diversity import NanobodyDiversityTracker, _pairwise_identity


REF = "EVELLASGGDLVQPGGSLRLSCAASGFTFSTYAMSWVRQAPGKGLERVSRVNQNGGTTTYADAMKGRFTISRDNAKNTLYLQMINVKPEDTAIYYCARWDGGSWSTDPWGRGTLVTVS"


def test_pairwise_identity_identical():
    assert _pairwise_identity(REF, REF) == 1.0


def test_pairwise_identity_disjoint():
    a = "AAAAAA"
    b = "VVVVVV"
    assert _pairwise_identity(a, b) == 0.0


def test_pairwise_identity_partial():
    a = "AAAAVV"
    b = "AAVVVV"
    # 2 matches at indices 0,1 (A,A) plus matches at 4,5 (V,V) — depending on alignment
    # Aligned positions: 0:A=A, 1:A=A, 2:A vs V (no), 3:A vs V (no), 4:V=V, 5:V=V → 4/6
    expected = 4 / 6
    assert _pairwise_identity(a, b) == expected


def test_pairwise_different_lengths():
    a = "AAAAAA"
    b = "AAA"
    assert _pairwise_identity(a, b) == 1.0  # all 3 aligned positions match


def test_empty_tracker():
    t = NanobodyDiversityTracker()
    assert t.median_identity() is None
    assert not t.is_collapsed()


def test_one_sequence_not_collapsed():
    t = NanobodyDiversityTracker()
    t.record(REF)
    assert t.median_identity() is None
    assert not t.is_collapsed()


def test_identical_sequences_collapsed():
    t = NanobodyDiversityTracker(identity_threshold=0.95)
    for _ in range(5):
        t.record(REF)
    assert t.median_identity() == 1.0
    assert t.is_collapsed()


def test_diverse_sequences_not_collapsed():
    t = NanobodyDiversityTracker(identity_threshold=0.95)
    # Different VHH templates have ~50-70% identity to each other
    seqs = [
        "EVELLASGGDLVQPGGSLRLSCAASGFTFSTYAMSWVRQAPGKGLERVSRVNQNGGTTTYADAMKGRFTISRDNAKNTLYLQMINVKPEDTAIYYCARWDGGSWSTDPWGRGTLVTVS",
        "QVQLVESGGGLVQPGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAGISWSGGSTYYTDSVKGRFTISRDNAKNTLYLQMNSLKPEDTAVYYCAAAGSAWYGTLYEYDYWGQGTQVTVSS",
        "EVQLQASGGGLVQPGGSLRLSCAASGRTLSDYAMGWFRQAPGKEREFVAAIYWSGGYAYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAATTPYTGSARFFNRYRSDYDYWGQGTQVTVSS",
    ]
    for s in seqs:
        t.record(s)
    med = t.median_identity()
    assert med is not None
    assert med < 0.8, f"Different templates should have <80% median identity, got {med:.3f}"
    assert not t.is_collapsed()


def test_reset_clears_window():
    t = NanobodyDiversityTracker()
    for _ in range(3):
        t.record(REF)
    assert t.median_identity() == 1.0
    t.reset()
    assert t.median_identity() is None
