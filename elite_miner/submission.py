"""Submission: drand-encrypt + chain commit + GitHub upload.

Mirrors neurons/miner/miner.py's submit_response with cleaner separation of
concerns: the encryption / commit / upload primitives are unit-testable, the
orchestration is in run.py.
"""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import bittensor as bt
from bittensor.core.errors import MetadataError


def build_message(molecule_name: Optional[str], nanobody_sequence: Optional[str]) -> str:
    """Validator-expected format: 'molecule|nanobody'.

    The validator's parse_decrypted_submission rejects submissions where either
    side of `|` is empty (`""`). Use `~` as the documented null placeholder when
    we only have one of the two — that way the present side is still scored.
    """
    m = molecule_name or "~"
    n = nanobody_sequence or "~"
    return f"{m}|{n}"


def encode_for_commit(encrypted_response) -> tuple[str, str]:
    """Take the drand-encrypted blob and return (filename, base64_content)."""
    tmp = tempfile.NamedTemporaryFile(delete=True)
    try:
        with open(tmp.name, "w+") as f:
            f.write(str(encrypted_response))
            f.flush()
            f.seek(0)
            content_str = f.read()
    finally:
        tmp.close()
    encoded = base64.b64encode(content_str.encode()).decode()
    filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
    return filename, encoded


def build_github_path() -> str:
    """Construct miner's GitHub path from env vars. Validates <= 100 chars."""
    owner = os.environ.get("GITHUB_REPO_OWNER")
    name = os.environ.get("GITHUB_REPO_NAME")
    branch = os.environ.get("GITHUB_REPO_BRANCH")
    path = os.environ.get("GITHUB_REPO_PATH")

    if not (owner and name and branch):
        raise ValueError("GITHUB_REPO_OWNER, GITHUB_REPO_NAME, GITHUB_REPO_BRANCH must be set")

    if not path:
        gh_path = f"{owner}/{name}/{branch}"
    else:
        gh_path = f"{owner}/{name}/{branch}/{path}"

    if len(gh_path) > 100:
        raise ValueError(f"GitHub path too long ({len(gh_path)} > 100): {gh_path}")
    return gh_path


@dataclass
class SubmissionResult:
    success: bool
    block_submitted: Optional[int] = None
    commit_content: Optional[str] = None
    reason: Optional[str] = None


async def submit(
    *,
    subtensor,
    wallet,
    netuid: int,
    miner_uid: int,
    bdt,                     # QuicknetBittensorDrandTimelock instance
    molecule_name: Optional[str],
    nanobody_sequence: Optional[str],
    github_path: str,
    upload_github_fn,        # callable(filename, encoded_content) -> bool
) -> SubmissionResult:
    """End-to-end: encrypt → chain commit → GitHub upload.

    Returns SubmissionResult. Success means chain commitment landed; GitHub
    upload failure is non-fatal but logged.
    """
    if not molecule_name and not nanobody_sequence:
        return SubmissionResult(success=False, reason="nothing to submit")

    message = build_message(molecule_name, nanobody_sequence)
    current_block = await subtensor.get_current_block()
    try:
        encrypted = bdt.encrypt(miner_uid, message, current_block)
    except Exception as e:
        bt.logging.error(f"submission: encryption failed: {e}")
        return SubmissionResult(success=False, reason=f"encrypt failed: {e}")

    filename, encoded_content = encode_for_commit(encrypted)
    commit_content = f"{github_path}/{filename}.txt"
    bt.logging.info(f"submission: prepared commit_content={commit_content}")

    try:
        commitment_status = await subtensor.set_commitment(
            wallet=wallet,
            netuid=netuid,
            data=commit_content,
        )
    except MetadataError:
        bt.logging.info("submission: rate-limited by chain, will retry later")
        return SubmissionResult(success=False, reason="rate-limited")
    except Exception as e:
        bt.logging.error(f"submission: set_commitment failed: {e}")
        return SubmissionResult(success=False, reason=f"chain commit failed: {e}")

    if not commitment_status:
        return SubmissionResult(success=False, reason="chain returned non-success")

    bt.logging.info(f"submission: chain commit successful for {commit_content}")

    # GitHub upload is non-fatal — chain commitment is what wins the epoch
    try:
        gh_ok = upload_github_fn(filename, encoded_content)
        if gh_ok:
            bt.logging.info(f"submission: github upload ok for {filename}")
        else:
            bt.logging.error(f"submission: github upload failed for {filename}")
    except Exception as e:
        bt.logging.error(f"submission: github upload exception for {filename}: {e}")

    return SubmissionResult(success=True, block_submitted=current_block, commit_content=commit_content)
