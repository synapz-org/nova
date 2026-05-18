"""Pre-flight verifier — run this before starting the miner.

Codifies `kb/strategy/03-deployment-playbook.md` Phase 1 as a single go/no-go
command. Exits 0 if all checks pass and the miner is safe to start; exits 1
with a specific failed-step message otherwise.

Yesterday's session would have caught the private-repo bug at step 3 (10
seconds), instead of after 4 hours of wasted compute.

Usage:
    python -m elite_miner.scripts.preflight \
        --hotkey 5EnzmPAvpRaQeDdmixf47Xvvt8N5E1wz5cC54CfmDrpyXPJY \
        --wallet-name omron_miner --wallet-hotkey metanova_miner \
        --netuid 68
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    line = f"  {mark} {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def step1_env() -> bool:
    print("STEP 1: env vars")
    required = ["GITHUB_TOKEN", "GITHUB_REPO_OWNER", "GITHUB_REPO_NAME", "GITHUB_REPO_BRANCH"]
    ok = True
    for k in required:
        ok &= check(f"{k} set", bool(os.environ.get(k)))
    return ok


def step2_repo_public() -> bool:
    print("STEP 2: GitHub repo public (validator must fetch via raw.githubusercontent.com)")
    owner = os.environ["GITHUB_REPO_OWNER"]
    name = os.environ["GITHUB_REPO_NAME"]
    branch = os.environ["GITHUB_REPO_BRANCH"]
    url = f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/README.md"
    try:
        r = requests.get(url, timeout=10)
        # Use README.md as a probe — if private, raw returns 404 even for valid paths
        ok = r.status_code == 200
        return check(f"raw.githubusercontent.com fetch", ok, f"HTTP {r.status_code} from {url}")
    except Exception as e:
        return check("raw fetch", False, str(e))


def step3_hotkey_registered(hotkey: str, netuid: int, network: str) -> tuple[bool, int]:
    print(f"STEP 3: hotkey registered on netuid {netuid}")
    try:
        import bittensor as bt
        sub = bt.subtensor(network=network)
        mg = sub.metagraph(netuid=netuid)
        if hotkey not in mg.hotkeys:
            check("hotkey in metagraph", False, f"hotkey {hotkey[:12]}… not on subnet {netuid}")
            return False, -1
        uid = mg.hotkeys.index(hotkey)
        check("hotkey in metagraph", True, f"uid={uid}")
        return True, uid
    except Exception as e:
        check("metagraph query", False, str(e))
        return False, -1


def step4_coldkey_runway(coldkey_ss58: str, netuid: int, network: str) -> bool:
    print(f"STEP 4: coldkey runway (≥ 2× recycle cost)")
    try:
        import bittensor as bt
        sub = bt.subtensor(network=network)
        balance = sub.get_balance(coldkey_ss58)
        recycle = sub.recycle(netuid=netuid)
        balance_tao = float(balance) if not hasattr(balance, "tao") else balance.tao
        recycle_tao = float(recycle) if not hasattr(recycle, "tao") else recycle.tao
        ok = balance_tao >= 2 * recycle_tao
        return check(
            "balance >= 2x recycle",
            ok,
            f"balance={balance_tao:.4f}τ recycle={recycle_tao:.4f}τ",
        )
    except Exception as e:
        return check("balance query", False, str(e))


def step5_e2e_verifier(hotkey: str, netuid: int, network: str) -> bool:
    print("STEP 5: end-to-end submission verifier")
    # Import inline so this script can run even if some deps are broken.
    try:
        from elite_miner.scripts.verify_submission_e2e import main as e2e_main
    except Exception as e:
        return check("import e2e verifier", False, str(e))
    # We don't want to mutate sys.argv for the subprocess interface; call directly.
    import argparse as _ap
    saved = sys.argv
    sys.argv = ["verify_submission_e2e", "--hotkey", hotkey, "--netuid", str(netuid), "--network", network]
    try:
        rc = e2e_main()
    finally:
        sys.argv = saved
    return check("e2e verifier exit code", rc == 0, f"exit={rc}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hotkey", required=True)
    p.add_argument("--coldkey", help="coldkey ss58 — if omitted, skips runway check")
    p.add_argument("--netuid", type=int, default=68)
    p.add_argument("--network", default="finney")
    args = p.parse_args()

    print(f"PREFLIGHT for hotkey={args.hotkey[:12]}… netuid={args.netuid}")
    print()

    # Load env from .env if present (matches miner behaviour)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    s1 = step1_env()
    if not s1:
        print("\n✗ FAILED at step 1 — fix .env and retry")
        return 1

    s2 = step2_repo_public()
    if not s2:
        print("\n✗ FAILED at step 2 — repo must be public (see kb/gotchas/github-submissions-repo-must-be-public.md)")
        return 1

    s3, _ = step3_hotkey_registered(args.hotkey, args.netuid, args.network)
    if not s3:
        print("\n✗ FAILED at step 3 — register the hotkey first")
        return 1

    if args.coldkey:
        s4 = step4_coldkey_runway(args.coldkey, args.netuid, args.network)
        if not s4:
            print("\n⚠ WARNING at step 4 — runway tight but not blocking")
            # don't fail on this — it's a warning
    else:
        print("STEP 4: skipped (no --coldkey provided)")

    s5 = step5_e2e_verifier(args.hotkey, args.netuid, args.network)
    # Step 5 may pass at "drand-pending" which is OK — that's still a green light
    # because step 1-5 of verify_submission_e2e all passed and we can't do better
    # without a signed drand round. If the verifier exits non-zero at steps 1-5,
    # we fail. If it exits 0 (either fully pass or pending), we succeed.
    if not s5:
        print("\n✗ FAILED at step 5 — pipeline broken, see e2e output above")
        return 1

    print()
    print("✅ ALL CHECKS PASS — safe to start the miner")
    print()
    print("Don't forget: hourly out-of-band sanity check (see kb/strategy/03-deployment-playbook.md Phase 3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
