"""End-to-end verification that our submissions are valid from the validator's perspective.

Replays the validator's exact fetch → hash-check → decrypt → parse path on one of
our own committed submissions. If every step passes, our pipeline is correct;
if any step fails, the failure mode tells us exactly which gate the validator
is rejecting us at.

Usage:
    python -m elite_miner.scripts.verify_submission_e2e \
        --hotkey 5EnzmPAvpRaQeDdmixf47Xvvt8N5E1wz5cC54CfmDrpyXPJY \
        --netuid 68

Exit code 0 if everything passes, 1 otherwise. Prints a checklist of each step.

Notes:
- Drand signatures for the target round must be published before decrypt can
  succeed. If the round is in the future (epoch still in progress), the script
  will report that as a pending-decrypt result, NOT a failure.
- Uses no authentication for the GitHub fetch — exactly what the validator does
  when it doesn't have access to our private repo.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from ast import literal_eval
from typing import Optional

import requests

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import bittensor as bt


# Match validator's MAX_RESPONSE_SIZE for the Range header
MAX_RESPONSE_SIZE = 5 * 1024 * 1024


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    line = f"  {mark} {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotkey", required=True, help="Miner hotkey to verify")
    parser.add_argument("--netuid", type=int, default=68)
    parser.add_argument("--network", default="finney")
    args = parser.parse_args()

    print(f"E2E submission verification for hotkey={args.hotkey[:12]}…")
    print()

    # Step 1: read on-chain commitment ---------------------------------------
    print("STEP 1: read on-chain commitment")
    sub = bt.subtensor(network=args.network)
    info = sub.query_module("Commitments", "CommitmentOf", params=[args.netuid, args.hotkey])
    if not info:
        check("commitment present", False, "no commitment for this hotkey")
        return 1
    commit_block = info["block"]
    # Decode the commitment data — same path as validator's decode_metadata
    try:
        commit_info = info["info"]
        fields = commit_info["fields"][0][0]
        # fields is something like {"Raw{N}": (...,)} — find the Raw key
        raw_key = next(k for k in fields if k.startswith("Raw"))
        raw_bytes = bytes(fields[raw_key][0])
        commit_data = raw_bytes.decode("utf-8")
    except Exception as e:
        check("commitment decode", False, str(e))
        print(f"    raw info: {info}")
        return 1
    check("commitment present", True, f"block={commit_block} path={commit_data}")

    # Step 2: derive UID -----------------------------------------------------
    print("STEP 2: derive UID")
    mg = sub.metagraph(netuid=args.netuid)
    try:
        uid = mg.hotkeys.index(args.hotkey)
    except ValueError:
        check("hotkey registered on netuid", False)
        return 1
    check("hotkey → UID", True, f"uid={uid}")

    # Step 3: fetch via raw.githubusercontent.com (unauthenticated) ----------
    print("STEP 3: fetch via raw.githubusercontent.com (unauthenticated, validator-style)")
    content_url = f"https://raw.githubusercontent.com/{commit_data}"
    try:
        resp = requests.get(content_url, headers={"Range": f"bytes=0-{MAX_RESPONSE_SIZE}"}, timeout=30)
    except Exception as e:
        check("raw GitHub fetch", False, str(e))
        return 1
    if resp.status_code not in (200, 206):
        check("raw GitHub fetch", False, f"HTTP {resp.status_code} — validator can't see this submission")
        return 1
    content_bytes = resp.content
    check("raw GitHub fetch", True, f"HTTP {resp.status_code}, {len(content_bytes)} bytes")

    # Step 4: content-hash verification --------------------------------------
    print("STEP 4: content-hash verification")
    content_str = content_bytes.decode("utf-8", errors="replace")
    content_hash = hashlib.sha256(content_str.encode("utf-8")).hexdigest()[:20]
    expected_suffix = f"/{content_hash}.txt"
    if not commit_data.endswith(expected_suffix):
        check("sha256(content)[:20] matches chain commit", False,
              f"content_hash={content_hash} but commit ends with {commit_data[-30:]!r}")
        return 1
    check("sha256(content)[:20] matches chain commit", True, f"hash={content_hash}")

    # Step 5: literal_eval into (target_round, ciphertext) -------------------
    print("STEP 5: parse encrypted payload")
    try:
        decoded = literal_eval(content_str)
    except Exception as e:
        check("literal_eval(content)", False, str(e))
        print(f"    content head: {content_str[:120]!r}")
        return 1
    if not (isinstance(decoded, tuple) and len(decoded) == 2):
        check("payload is a 2-tuple", False, f"got {type(decoded).__name__} len={len(decoded) if hasattr(decoded,'__len__') else '?'}")
        return 1
    target_round, ciphertext = decoded
    if not isinstance(target_round, int):
        check("target_round is int", False, f"got {type(target_round).__name__}")
        return 1
    if not isinstance(ciphertext, (bytes, bytearray)):
        check("ciphertext is bytes", False, f"got {type(ciphertext).__name__}")
        return 1
    check("payload structure", True, f"target_round={target_round} ciphertext={len(ciphertext)} bytes")

    # Step 6: check whether the drand round is unlockable yet ----------------
    print("STEP 6: drand round availability")
    from utils.btdr import QuicknetBittensorDrandTimelock, DrandClient
    btd = QuicknetBittensorDrandTimelock()
    try:
        current_round = btd.get_current_round()
    except Exception as e:
        check("drand current_round fetch", False, str(e))
        return 1
    rounds_to_wait = target_round - current_round
    if rounds_to_wait > 0:
        check("drand round unlocked", False, f"target={target_round} current={current_round} — wait {rounds_to_wait} rounds (~{rounds_to_wait*3}s)")
        print()
        print("    (Pipeline up to encryption is correct. Decrypt will succeed once epoch closes.)")
        return 0
    check("drand round unlocked", True, f"target={target_round} current={current_round}")

    # Step 7: decrypt --------------------------------------------------------
    print("STEP 7: decrypt with drand signature")
    try:
        decrypted = btd.decrypt(uid, bytes(ciphertext), target_round)
    except Exception as e:
        check("decrypt", False, str(e))
        return 1
    if decrypted is None:
        check("decrypt", False, "returned None — likely UID-prefix mismatch")
        return 1
    check("decrypt", True, f"plaintext={decrypted!r}")

    # Step 8: parse_decrypted_submission -------------------------------------
    print("STEP 8: validator-side parse")
    from neurons.validator.commitments import parse_decrypted_submission
    # We don't know num_molecules / num_sequences from this script — use 1, 1 which is current config
    parsed = parse_decrypted_submission(uid, decrypted, num_molecules=1, num_sequences=1)
    if parsed is None:
        check("parse_decrypted_submission", False, "rejected by validator format gate")
        return 1
    mols, seqs = parsed
    check("parse_decrypted_submission", True, f"mols={mols} seqs={[s[:30]+'…' for s in seqs]}")

    print()
    print("ALL CHECKS PASS — validator can fetch, hash-verify, decrypt, and parse this submission.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
