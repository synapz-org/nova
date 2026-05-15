"""Tests for timing primitives."""

from elite_miner.timing import EpochState, SubmissionRateLimiter


def test_epoch_state_blocks_remaining():
    es = EpochState(epoch_length=360, current_block=100, last_boundary=0, next_boundary=360)
    assert es.blocks_into_epoch == 100
    assert es.blocks_remaining == 260


def test_epoch_state_at_boundary():
    es = EpochState(epoch_length=360, current_block=360, last_boundary=360, next_boundary=720)
    assert es.is_at_boundary
    assert es.blocks_into_epoch == 0


def test_epoch_state_near_end():
    es = EpochState(epoch_length=360, current_block=355, last_boundary=0, next_boundary=360)
    assert es.is_near_end


def test_rate_limiter_initial_allows():
    rl = SubmissionRateLimiter(no_submission_blocks=10)
    assert rl.can_submit(100)


def test_rate_limiter_blocks_within_window():
    rl = SubmissionRateLimiter(no_submission_blocks=10)
    rl.record_submission(100)
    assert not rl.can_submit(105)
    assert not rl.can_submit(109)
    assert rl.can_submit(110)


def test_rate_limiter_blocks_until_window():
    rl = SubmissionRateLimiter(no_submission_blocks=10)
    rl.record_submission(100)
    assert rl.blocks_until_next_submission(105) == 5
    assert rl.blocks_until_next_submission(110) == 0
    assert rl.blocks_until_next_submission(200) == 0
