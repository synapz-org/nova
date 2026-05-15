"""End-to-end mocked epoch test.

Exercises run.run_epoch() with mocked subtensor + wallet + scorer + submission,
verifying the search → score → submit path produces a valid validator-format
submission message.

Doesn't require a live subtensor, GPU, or registered wallet. Catches wiring
regressions that the per-module tests would miss.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elite_miner.run import MoleculeTrack, NanobodyTrack, _do_submit, run_epoch
from elite_miner.timing import EpochState, SubmissionRateLimiter
from elite_miner.submission import SubmissionResult


def _build_config():
    """Build a minimal config namespace shaped like config.yaml + argparse."""
    return SimpleNamespace(
        netuid=379,
        small_molecule_target=["Q6P6W3"],
        small_molecule_target_clip_interval=[None],
        nanobody_target=["Q9NZQ7"],
        nanobody_target_clip_interval=[(17, 132)],
        no_submission_blocks=10,
        nanobody_weight=0.6,
        num_molecules=1,
        num_sequences=1,
        min_heavy_atoms=10,
        min_rotatable_bonds=1,
        max_rotatable_bonds=10,
        banned_atom_types=["Se", "Na", "Fe", "Zn"],
        min_sequence_length=90,
        max_sequence_length=150,
        min_cysteines=1,
        cys_pair_min_separation=35,
        cys_pair_max_separation=80,
        max_homopolymer_run=6,
        max_di_repeat_pairs=4,
        reject_signal_peptides=True,
        sp_window=12,
        sp_hydro_min_in_window=8,
        sp_scan_prefix=30,
        enforce_vhh_hallmarks=True,
        random_valid_reaction=True,
        allowed_reactions=["rxn:1"],
        boltz_metric=["affinity_probability_binary", "affinity_pred_value"],
        combination_strategy="heavy_atom_normalization",
        boltz_mode="max",
        boltzgen_rank_mode="min",
        boltzgen_rank_by="rank_sum",
        batch_size=20,
        max_batches_per_epoch=2,
        use_inference=False,
        no_uniqueness_check=True,   # skip HF lookups in tests
        disable_molecule_track=False,
        disable_nanobody_track=False,
    )


def test_molecule_track_finds_best_in_epoch():
    config = _build_config()
    track = MoleculeTrack(config)
    track.reset_for_epoch(rxn_id=1)
    cand = track.search_one_batch()
    assert cand is not None
    track.update_best(cand)
    assert track.best is not None
    assert track.best.name.startswith("rxn:1:")
    assert track.candidates_scored > 0


def test_nanobody_track_finds_best_in_epoch():
    config = _build_config()
    track = NanobodyTrack(config)
    track.reset_for_epoch()
    cand = track.search_one_batch()
    assert cand is not None
    track.update_best(cand)
    assert track.best is not None
    assert 90 <= len(track.best.sequence) <= 150


def test_track_keeps_best_across_batches():
    config = _build_config()
    track = MoleculeTrack(config)
    track.reset_for_epoch(rxn_id=2)

    track.update_best(track.search_one_batch())
    first_best = track.best
    assert first_best is not None

    # Run more batches; best may improve but should never regress
    for _ in range(3):
        track.update_best(track.search_one_batch())
        assert track.best.score >= first_best.score


@pytest.mark.asyncio
async def test_do_submit_produces_correct_message():
    config = _build_config()
    mt = MoleculeTrack(config)
    mt.reset_for_epoch(rxn_id=1)
    mt.update_best(mt.search_one_batch())
    assert mt.best is not None

    nt = NanobodyTrack(config)
    nt.reset_for_epoch()
    nt.update_best(nt.search_one_batch())
    assert nt.best is not None

    # Mock subtensor / wallet / bdt / upload
    fake_subtensor = MagicMock()
    fake_subtensor.get_current_block = AsyncMock(return_value=100)
    fake_subtensor.set_commitment = AsyncMock(return_value=True)

    captured_message = {}

    class FakeBDT:
        def encrypt(self, uid, message, block):
            captured_message["msg"] = message
            captured_message["uid"] = uid
            captured_message["block"] = block
            return "encrypted-blob"

    fake_upload = MagicMock(return_value=True)
    state = {
        "subtensor": fake_subtensor,
        "wallet": MagicMock(),
        "config": config,
        "miner_uid": 42,
        "bdt": FakeBDT(),
        "github_path": "test/repo/main",
    }

    # Patch the upload function reference used inside _do_submit
    with patch("elite_miner.run._get_upload_github_fn", return_value=fake_upload):
        result = await _do_submit(state, mt, nt)

    assert result.success, f"submission should have succeeded: {result.reason}"
    msg = captured_message["msg"]
    assert "|" in msg, f"message should be molecule|nanobody format, got: {msg}"
    mol_part, nb_part = msg.split("|", 1)
    assert mol_part.startswith("rxn:1:")
    assert len(nb_part) >= 90
    assert captured_message["uid"] == 42
    fake_subtensor.set_commitment.assert_awaited_once()
    fake_upload.assert_called_once()


@pytest.mark.asyncio
async def test_run_epoch_submits_at_least_once():
    """Full run_epoch loop with mocked subtensor — verifies the orchestration."""
    config = _build_config()
    config.max_batches_per_epoch = 3

    blocks = [100, 100, 101, 102, 103, 104, 200]
    block_iter = iter(blocks)

    fake_subtensor = MagicMock()

    async def get_current_block():
        try:
            return next(block_iter)
        except StopIteration:
            return 999

    fake_subtensor.get_current_block = get_current_block
    fake_subtensor.set_commitment = AsyncMock(return_value=True)

    class FakeBDT:
        def encrypt(self, uid, message, block):
            return f"enc({message[:50]})"

    fake_upload = MagicMock(return_value=True)

    state = {
        "subtensor": fake_subtensor,
        "wallet": MagicMock(),
        "config": config,
        "miner_uid": 42,
        "bdt": FakeBDT(),
        "github_path": "test/repo/main",
        "epoch_length": 360,
        "rate_limiter": SubmissionRateLimiter(no_submission_blocks=10),
    }

    epoch_state = EpochState(
        epoch_length=360,
        current_block=100,
        last_boundary=0,
        next_boundary=360,
        block_hash="0x" + "ab" * 32,
    )

    mt = MoleculeTrack(config)
    nt = NanobodyTrack(config)

    # Patch the lazy upload getter and the challenge function
    fake_challenge = MagicMock(return_value={
        "small_molecule_target": ["Q6P6W3"],
        "nanobody_target": ["Q9NZQ7"],
        "allowed_reaction": "rxn:1",
    })

    with patch("elite_miner.run._get_upload_github_fn", return_value=fake_upload), \
         patch("elite_miner.run._get_challenge_fn", return_value=fake_challenge):
        await run_epoch(state, mt, nt, epoch_state)

    # At least one chain commit should have happened
    assert fake_subtensor.set_commitment.await_count >= 1
    # And a github upload paired with each successful chain commit
    assert fake_upload.call_count >= 1
