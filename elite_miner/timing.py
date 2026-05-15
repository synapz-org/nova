"""Epoch boundary detection and submission rate-limit tracking."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import bittensor as bt


@dataclass
class EpochState:
    epoch_length: int
    current_block: int
    last_boundary: int        # block at the start of current epoch
    next_boundary: int        # block at the start of next epoch
    block_hash: Optional[str] = None  # hash of last_boundary block

    @property
    def blocks_into_epoch(self) -> int:
        return self.current_block - self.last_boundary

    @property
    def blocks_remaining(self) -> int:
        return self.next_boundary - self.current_block

    @property
    def is_at_boundary(self) -> bool:
        return self.current_block % self.epoch_length == 0

    @property
    def is_near_end(self) -> bool:
        """Whether we're too close to epoch end for new work to be useful."""
        return self.blocks_remaining < 20


async def get_epoch_state(subtensor, epoch_length: int) -> EpochState:
    current = await subtensor.get_current_block()
    last_boundary = (current // epoch_length) * epoch_length
    next_boundary = last_boundary + epoch_length
    block_hash = await subtensor.determine_block_hash(last_boundary)
    return EpochState(
        epoch_length=epoch_length,
        current_block=current,
        last_boundary=last_boundary,
        next_boundary=next_boundary,
        block_hash=block_hash,
    )


async def wait_until_block(subtensor, target_block: int, poll_seconds: float = 12.0) -> None:
    """Sleep in increments until subtensor reports current_block >= target_block."""
    while True:
        current = await subtensor.get_current_block()
        if current >= target_block:
            return
        bt.logging.debug(f"timing: waiting for block {target_block}, currently at {current}")
        await asyncio.sleep(poll_seconds)


class SubmissionRateLimiter:
    """Tracks last submission block to enforce no_submission_blocks rate limit.

    The chain enforces this server-side too (set_commitment raises MetadataError),
    so this is just to avoid wasting cycles on doomed attempts.
    """

    def __init__(self, no_submission_blocks: int):
        self.no_submission_blocks = no_submission_blocks
        self.last_submitted_block: int = 0

    def can_submit(self, current_block: int) -> bool:
        if self.last_submitted_block == 0:
            return True
        return (current_block - self.last_submitted_block) >= self.no_submission_blocks

    def record_submission(self, block: int) -> None:
        self.last_submitted_block = block

    def blocks_until_next_submission(self, current_block: int) -> int:
        if self.last_submitted_block == 0:
            return 0
        return max(0, self.no_submission_blocks - (current_block - self.last_submitted_block))
