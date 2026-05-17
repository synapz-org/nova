"""Run the validator's actual decrypt_submissions() against our own submission.

Unlike verify_submission_e2e.py which replays the validator's logic step by
step, this script imports and calls the literal validator code path —
`neurons.validator.commitments.decrypt_submissions`. If that function
returns our UID in its parsed-submissions dict, the validator would
accept us. If it doesn't, the validator would reject us in the exact
same way.

Usage:
    python -m elite_miner.scripts.run_validator_decrypt \
        --hotkey 5EnzmPAvpRaQeDdmixf47Xvvt8N5E1wz5cC54CfmDrpyXPJY \
        --netuid 68

Exit 0 = validator code accepts our submission.
Exit 1 = validator code rejects (or drand round not yet unlockable).

The drand round must be signed before decrypt can succeed. If the epoch
is still in progress, the script reports "pending" not "fail".
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import bittensor as bt

from neurons.validator.commitments import decrypt_submissions
from utils.btdr import QuicknetBittensorDrandTimelock
from elite_miner.config import subnet_config_dict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotkey", required=True)
    parser.add_argument("--netuid", type=int, default=68)
    parser.add_argument("--network", default="finney")
    args = parser.parse_args()

    print(f"Running validator's decrypt_submissions() against hotkey={args.hotkey[:12]}…")
    print()

    # 1) Read our on-chain commitment and stuff it into the SimpleNamespace shape
    #    that decrypt_submissions() expects.
    sub = bt.subtensor(network=args.network)
    info = sub.query_module("Commitments", "CommitmentOf", params=[args.netuid, args.hotkey])
    if not info:
        print("✗ no on-chain commitment for this hotkey")
        return 1

    mg = sub.metagraph(netuid=args.netuid)
    try:
        uid = mg.hotkeys.index(args.hotkey)
    except ValueError:
        print("✗ hotkey not registered on this netuid")
        return 1

    # decode_metadata path — same as get_commitments() does
    from bittensor.core.chain_data.utils import decode_metadata
    commit_data = decode_metadata(info)
    block_submitted = info["block"]
    print(f"  uid={uid}")
    print(f"  block={block_submitted}")
    print(f"  commitment data={commit_data}")
    print()

    # decrypt_submissions wants a {hotkey: SimpleNamespace(uid, hotkey, block, data)} dict
    current_commitments = {
        args.hotkey: SimpleNamespace(
            uid=uid,
            hotkey=args.hotkey,
            block=block_submitted,
            data=commit_data,
        )
    }

    # 2) Validator's github auth: it only has its own token, NOT ours. To honestly
    #    test the validator's path, we send NO Authorization header at all — the
    #    raw.githubusercontent.com fetch must work as a public read.
    github_headers = {"Accept": "application/vnd.github+json"}

    # 3) Build the subnet config the way the validator constructs it.
    #    For our purposes we just need num_molecules and num_sequences.
    cfg_dict = {"num_molecules": 1, "num_sequences": 1}

    # 4) Drand timelock object
    btd = QuicknetBittensorDrandTimelock()

    # 5) Run the literal validator function
    print("Calling validator's decrypt_submissions()…")
    print()
    parsed, push_timestamps = decrypt_submissions(
        current_commitments, github_headers, btd, cfg_dict
    )

    print(f"push_timestamps: {push_timestamps}")
    print(f"parsed (validator's accepted submissions): {parsed}")
    print()

    if uid in parsed:
        sub = parsed[uid]
        print(f"✓ VALIDATOR ACCEPTS our submission")
        print(f"  molecules: {sub.get('molecules')}")
        print(f"  sequences: {[(s[:30] + '…') for s in sub.get('sequences', [])]}")
        return 0
    else:
        # Could be: drand round not yet unlocked, decrypt None, parse rejected, etc.
        # decrypt_submissions silently drops failures; we have to figure out why.
        print("✗ Our UID is NOT in the validator's accepted submissions.")
        print()
        print("Possible reasons (in order of likelihood):")
        print("  - Drand round target_round isn't signed yet (epoch in progress)")
        print("  - raw.githubusercontent.com returned non-200 (private repo / CDN cache)")
        print("  - content_str sha256[:20] != filename in chain commit")
        print("  - literal_eval of the payload failed")
        print("  - Decrypt returned None (UID-prefix mismatch)")
        print("  - parse_decrypted_submission rejected (empty side of |, etc.)")
        print()
        print("Run elite_miner.scripts.verify_submission_e2e for step-by-step diagnosis.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
