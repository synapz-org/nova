"""Elite miner main loop with combinatorial search strategy.

Ties together CombinatorialSearcher, ValidityFilter, and ProxyScorer
with Bittensor infrastructure for epoch-based molecule submission.
"""

import os
import sys
import json
import argparse
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib

from typing import Any, Dict

from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

import bittensor as bt
from bittensor.core.errors import MetadataError
from substrateinterface import SubstrateInterface

from config.config_loader import load_config
from utils import (
    get_sequence_from_protein_code,
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    get_heavy_atom_count,
)
from utils.molecules import molecule_unique_for_protein_hf
from PSICHIC.wrapper import PsichicWrapper
from btdr import QuicknetBittensorDrandTimelock

from elite_miner.searcher import CombinatorialSearcher
from elite_miner.filters import ValidityFilter
from elite_miner.scorer import ProxyScorer

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
STATE_FILE = os.path.join(BASE_DIR, "elite_miner", "miner_state.json")


# ---------------------------------------------------------------------------
# 1. CONFIG & ARGUMENT PARSING
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments and merge with config defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default=os.getenv("SUBTENSOR_NETWORK"), help="Network to use")
    parser.add_argument("--netuid", type=int, default=68, help="The chain subnet uid.")
    parser.add_argument("--batch_size", type=int, default=500, help="Molecules per search batch.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    config = bt.config(parser)
    config.update(load_config())

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            "elite_miner",
        )
    )
    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    """Build GitHub path from env vars; validate <= 100 chars."""
    github_repo_name = os.environ.get("GITHUB_REPO_NAME")
    github_repo_branch = os.environ.get("GITHUB_REPO_BRANCH")
    github_repo_owner = os.environ.get("GITHUB_REPO_OWNER")
    github_repo_path = os.environ.get("GITHUB_REPO_PATH")

    if github_repo_name is None or github_repo_branch is None or github_repo_owner is None:
        raise ValueError("Missing one or more GitHub environment variables (GITHUB_REPO_*)")

    if github_repo_path == "" or github_repo_path is None:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path is too long. Please shorten it to 100 characters or less.")
    return github_path


# ---------------------------------------------------------------------------
# 2. LOGGING SETUP
# ---------------------------------------------------------------------------

def setup_logging(config: argparse.Namespace) -> None:
    """Set up Bittensor logging."""
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running elite miner for subnet: {config.netuid} on network: {config.subtensor.network}")
    bt.logging.info(config)


# ---------------------------------------------------------------------------
# 3. BITTENSOR SETUP
# ---------------------------------------------------------------------------

async def setup_bittensor(config: argparse.Namespace):
    """Initialize wallet, subtensor, metagraph; return (wallet, metagraph, miner_uid, epoch_length)."""
    bt.logging.info("Setting up Bittensor objects.")
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    try:
        async with bt.async_subtensor(network=config.network) as subtensor:
            bt.logging.info(f"Connected to subtensor network: {config.network}")
            metagraph = await subtensor.metagraph(config.netuid)
            await metagraph.sync()
            bt.logging.info("Metagraph synced successfully.")

            miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            bt.logging.info(f"Miner UID: {miner_uid}")

            node = SubstrateInterface(url=config.network)
            epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1
            bt.logging.info(f"Epoch length: {epoch_length} blocks")

        return wallet, metagraph, miner_uid, epoch_length
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        raise


# ---------------------------------------------------------------------------
# 4. PERSISTENT STATE
# ---------------------------------------------------------------------------

def load_persistent_state() -> set:
    """Load set of previously submitted mol_names from disk."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            submitted = set(data.get("submitted", []))
            bt.logging.info(f"Loaded {len(submitted)} previously submitted molecules from state file.")
            return submitted
        except Exception as e:
            bt.logging.warning(f"Could not load state file: {e}")
    return set()


def save_persistent_state(submitted: set) -> None:
    """Save set of submitted mol_names to disk."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"submitted": list(submitted)}, f)
        bt.logging.debug(f"Saved {len(submitted)} submitted molecules to state file.")
    except Exception as e:
        bt.logging.warning(f"Could not save state file: {e}")


# ---------------------------------------------------------------------------
# 5. SUBMISSION
# ---------------------------------------------------------------------------

async def submit_response(state: Dict[str, Any]) -> None:
    """Encrypt candidate with DRAND, commit to chain, upload to GitHub."""
    candidate_product = state["candidate_product"]
    if not candidate_product:
        bt.logging.warning("No candidate product to submit.")
        return

    bt.logging.info(f"Starting submission for: {candidate_product}")

    # Encrypt
    current_block = await state["subtensor"].get_current_block()
    encrypted_response = state["bdt"].encrypt(state["miner_uid"], candidate_product, current_block)
    bt.logging.info("Encrypted response generated.")

    # Write to temp file, base64 encode
    tmp_file = tempfile.NamedTemporaryFile(delete=True)
    with open(tmp_file.name, "w+") as f:
        f.write(str(encrypted_response))
        f.flush()
        f.seek(0)
        content_str = f.read()
        encoded_content = base64.b64encode(content_str.encode()).decode()

        filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
        commit_content = f"{state['github_path']}/{filename}.txt"
        bt.logging.info(f"Commit content path: {commit_content}")

        # Chain commitment
        try:
            commitment_status = await state["subtensor"].set_commitment(
                wallet=state["wallet"],
                netuid=state["config"].netuid,
                data=commit_content,
            )
            bt.logging.info(f"Chain commitment status: {commitment_status}")
        except MetadataError:
            bt.logging.info("Too soon to commit again. Will retry later.")
            return

        # Upload to GitHub
        if commitment_status:
            try:
                github_status = upload_file_to_github(filename, encoded_content)
                if github_status:
                    bt.logging.info(f"Uploaded to GitHub: {commit_content}")
                    state["last_submitted_product"] = candidate_product
                    state["submitted_names"].add(candidate_product)
                    save_persistent_state(state["submitted_names"])
                else:
                    bt.logging.error("GitHub upload failed.")
            except Exception as e:
                bt.logging.error(f"GitHub upload error: {e}")


# ---------------------------------------------------------------------------
# 6. SEARCH & SCORE
# ---------------------------------------------------------------------------

async def search_and_score(state: Dict[str, Any]) -> None:
    """Run one search cycle: generate -> filter -> uniqueness -> PSICHIC score -> rank."""
    searcher = state["searcher"]
    validity_filter = state["validity_filter"]
    config = state["config"]

    # 1. Generate batch
    batch = searcher.generate_batch(batch_size=config.batch_size)
    if not batch:
        bt.logging.warning("Empty batch from searcher.")
        return
    bt.logging.info(f"Generated batch of {len(batch)} molecules.")

    # 2. Filter
    valid = validity_filter.filter_batch(batch)
    bt.logging.info(f"After validity filter: {len(valid)}/{len(batch)} passed.")
    if not valid:
        return

    # 3. Check uniqueness against HF archive (for each target protein)
    unique = []
    for mol_name, smiles in valid:
        # Check against all target proteins
        is_unique = True
        for target_protein in state["current_challenge_targets"]:
            if not molecule_unique_for_protein_hf(target_protein, smiles):
                is_unique = False
                break
        if is_unique:
            unique.append((mol_name, smiles))

    bt.logging.info(f"After uniqueness check: {len(unique)}/{len(valid)} passed.")
    if not unique:
        return

    # 4. Score with PSICHIC against targets and antitargets
    smiles_list = [smi for _, smi in unique]

    # Target scores
    target_scores_all = []
    for target_protein in state["current_challenge_targets"]:
        model = state["psichic_models"].get(target_protein)
        if model is None:
            continue
        results_df = model.run_validation(smiles_list)
        target_scores_all.append(results_df["predicted_binding_affinity"].tolist())

    if not target_scores_all:
        bt.logging.warning("No target scores computed.")
        return

    # Average target scores across all targets
    avg_target = [
        sum(scores[i] for scores in target_scores_all) / len(target_scores_all)
        for i in range(len(smiles_list))
    ]

    # Antitarget scores
    avg_anti = [0.0] * len(smiles_list)
    antitarget_scores_all = []
    for anti_protein in state["current_challenge_antitargets"]:
        model = state["psichic_models"].get(anti_protein)
        if model is None:
            continue
        results_df = model.run_validation(smiles_list)
        antitarget_scores_all.append(results_df["predicted_binding_affinity"].tolist())

    if antitarget_scores_all:
        avg_anti = [
            sum(scores[i] for scores in antitarget_scores_all) / len(antitarget_scores_all)
            for i in range(len(smiles_list))
        ]

    # 5. Build tuples for ProxyScorer
    scored_molecules = []
    for idx, (mol_name, smiles) in enumerate(unique):
        ha = get_heavy_atom_count(smiles)
        scored_molecules.append((mol_name, smiles, avg_target[idx], avg_anti[idx], ha))

    ranked = ProxyScorer.rank_with_antitarget(
        scored_molecules, antitarget_weight=config.antitarget_weight
    )

    # 6. Update best candidate if improved
    if ranked:
        best = ranked[0]
        best_name, best_smiles, best_target, best_anti, best_ha, best_proxy = best
        bt.logging.info(
            f"Top candidate: {best_name} | proxy={best_proxy:.6f} "
            f"target={best_target:.4f} anti={best_anti:.4f} HA={best_ha}"
        )

        if best_proxy > state["best_proxy_score"]:
            state["best_proxy_score"] = best_proxy
            state["candidate_product"] = best_name
            state["candidate_smiles"] = best_smiles
            bt.logging.info(f"*** New best! proxy={best_proxy:.6f} mol={best_name}")


# ---------------------------------------------------------------------------
# 7. MAIN MINING LOOP
# ---------------------------------------------------------------------------

async def run_miner(config: argparse.Namespace) -> None:
    """Main mining loop: setup, search, score, submit on epoch cadence."""

    # 1. Setup BT objects
    wallet, metagraph, miner_uid, epoch_length = await setup_bittensor(config)

    # 2. Create long-lived subtensor
    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()
    bt.logging.info("Long-lived subtensor initialized.")

    # 3. Persistent state, searcher, filter
    submitted_names = load_persistent_state()

    searcher = CombinatorialSearcher(DB_PATH)
    searcher.load_rxn5_building_blocks()
    bt.logging.info(
        f"Loaded {len(searcher.scaffolds)} scaffolds, {len(searcher.boronic_acids)} boronic acids."
    )

    validity_filter = ValidityFilter(
        min_heavy_atoms=config.min_heavy_atoms,
        min_rotatable_bonds=config.min_rotatable_bonds,
        max_rotatable_bonds=config.max_rotatable_bonds,
        banned_atoms=config.banned_atom_types,
    )

    github_path = load_github_path()
    bdt = QuicknetBittensorDrandTimelock()

    # Shared state dict
    state: Dict[str, Any] = {
        "config": config,
        "wallet": wallet,
        "subtensor": subtensor,
        "metagraph": metagraph,
        "miner_uid": miner_uid,
        "epoch_length": epoch_length,
        "github_path": github_path,
        "bdt": bdt,
        "searcher": searcher,
        "validity_filter": validity_filter,
        "psichic_models": {},
        "current_challenge_targets": [],
        "current_challenge_antitargets": [],
        "candidate_product": None,
        "candidate_smiles": None,
        "best_proxy_score": float("-inf"),
        "last_submitted_product": None,
        "submitted_names": submitted_names,
    }

    # 4. Get initial challenge from last epoch boundary
    current_block = await subtensor.get_current_block()
    last_boundary = (current_block // epoch_length) * epoch_length

    bt.logging.info(f"Current block: {current_block}, last epoch boundary: {last_boundary}")

    block_hash = await subtensor.determine_block_hash(last_boundary)
    challenge = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        weekly_target=config.weekly_target,
        num_antitargets=config.num_antitargets,
    )

    if challenge:
        state["current_challenge_targets"] = challenge["targets"]
        state["current_challenge_antitargets"] = challenge["antitargets"]
        bt.logging.info(f"Targets: {challenge['targets']}, Antitargets: {challenge['antitargets']}")

        # Initialize PSICHIC models for all proteins
        for protein in challenge["targets"] + challenge["antitargets"]:
            try:
                seq = get_sequence_from_protein_code(protein)
                model = PsichicWrapper()
                model.run_challenge_start(seq)
                state["psichic_models"][protein] = model
                bt.logging.info(f"Initialized PSICHIC model for {protein}")
            except Exception as e:
                bt.logging.error(f"Failed to init model for {protein}: {e}")

    # 5. Main loop
    bt.logging.info("Entering main mining loop...")
    last_sync_block = 0

    while True:
        try:
            current_block = await subtensor.get_current_block()
            next_boundary = ((current_block // epoch_length) + 1) * epoch_length
            blocks_remaining = next_boundary - current_block

            # --- New epoch boundary ---
            if current_block % epoch_length == 0:
                bt.logging.info(f"=== EPOCH BOUNDARY at block {current_block} ===")

                # Refresh config
                config.update(load_config())

                # Get new challenge
                block_hash = await subtensor.determine_block_hash(current_block)
                new_challenge = get_challenge_params_from_blockhash(
                    block_hash=block_hash,
                    weekly_target=config.weekly_target,
                    num_antitargets=config.num_antitargets,
                )

                if new_challenge:
                    state["current_challenge_targets"] = new_challenge["targets"]
                    state["current_challenge_antitargets"] = new_challenge["antitargets"]
                    bt.logging.info(
                        f"New targets: {new_challenge['targets']}, "
                        f"antitargets: {new_challenge['antitargets']}"
                    )

                    # Init models for any new proteins
                    for protein in new_challenge["targets"] + new_challenge["antitargets"]:
                        if protein not in state["psichic_models"]:
                            try:
                                seq = get_sequence_from_protein_code(protein)
                                model = PsichicWrapper()
                                model.run_challenge_start(seq)
                                state["psichic_models"][protein] = model
                                bt.logging.info(f"Initialized PSICHIC model for {protein}")
                            except Exception as e:
                                bt.logging.error(f"Failed to init model for {protein}: {e}")

                # Reset per-epoch state
                state["candidate_product"] = None
                state["candidate_smiles"] = None
                state["best_proxy_score"] = float("-inf")
                state["last_submitted_product"] = None

                # Reset searcher iterator for fresh shuffle
                searcher._combo_iter = None
                bt.logging.info("Per-epoch state reset. Starting new search.")

            # --- Search phase: >15 blocks remaining ---
            if blocks_remaining > 15:
                try:
                    await search_and_score(state)
                except Exception as e:
                    bt.logging.error(f"Search error: {e}")
                    traceback.print_exc()

            # --- Submit phase: 10-15 blocks remaining, have candidate ---
            elif 10 <= blocks_remaining <= 15 and state["candidate_product"]:
                if state["candidate_product"] != state["last_submitted_product"]:
                    bt.logging.info(
                        f"Submission window ({blocks_remaining} blocks left). "
                        f"Submitting: {state['candidate_product']}"
                    )
                    try:
                        await submit_response(state)
                    except Exception as e:
                        bt.logging.error(f"Submission error: {e}")
                        traceback.print_exc()
                else:
                    bt.logging.debug("Already submitted current best candidate.")

            # --- Sync metagraph every 60 blocks ---
            if current_block - last_sync_block >= 60:
                try:
                    await metagraph.sync()
                    last_sync_block = current_block
                    bt.logging.info(
                        f"Metagraph synced at block {current_block} | "
                        f"Blocks remaining in epoch: {blocks_remaining}"
                    )
                except Exception as e:
                    bt.logging.warning(f"Metagraph sync failed: {e}")

            await asyncio.sleep(1.5)

        except asyncio.CancelledError:
            bt.logging.warning("Task cancelled, reconnecting subtensor...")
            try:
                subtensor = bt.async_subtensor(network=config.network)
                await subtensor.initialize()
                state["subtensor"] = subtensor
                bt.logging.info("Subtensor reconnected.")
            except Exception as e:
                bt.logging.error(f"Reconnection failed: {e}")
                await asyncio.sleep(10)

        except KeyboardInterrupt:
            bt.logging.info("Keyboard interrupt detected. Shutting down.")
            save_persistent_state(state["submitted_names"])
            break

        except Exception as e:
            bt.logging.error(f"Unexpected error in main loop: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

async def main() -> None:
    """Parse args, setup logging, run miner."""
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
