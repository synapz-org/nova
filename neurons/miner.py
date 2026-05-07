import os
import sys
import math
import random
import argparse
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib
import sqlite3

from rdkit import Chem
from rdkit.Chem import Descriptors

from typing import Any, Dict, List, Optional, Tuple, cast
from types import SimpleNamespace

from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata
from bittensor.core.errors import MetadataError
from substrateinterface import SubstrateInterface
from datasets import load_dataset
from huggingface_hub import list_repo_files
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

BOLTZ_CACHE_DB = os.path.join(BASE_DIR, "boltz_score_cache.db")

from config.config_loader import load_config
from utils import (
    get_sequence_from_protein_code,
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    get_heavy_atom_count,
    compute_maccs_entropy,
)
from utils.molecules import is_boltz_safe_smiles, get_canonical_smiles
from utils.msa import ensure_msa
from utils.salsa import run_salsa_search
from utils.genetic import run_gradient_ga
from PSICHIC.wrapper import PsichicWrapper
from boltz.wrapper import BoltzWrapper
from btdr import QuicknetBittensorDrandTimelock

# ----------------------------------------------------------------------------
# 0. PERSISTENT BOLTZ SCORE CACHE (SQLite)
# ----------------------------------------------------------------------------

def _init_boltz_cache_db(db_path: str) -> None:
    """Create the boltz_cache table if it doesn't exist."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS boltz_cache (
                smiles  TEXT NOT NULL,
                protein TEXT NOT NULL,
                score   REAL NOT NULL,
                ts      INTEGER DEFAULT (strftime('%s','now')),
                PRIMARY KEY (smiles, protein)
            )
        """)
        # Add product_name column when upgrading from older schema that lacked it.
        # SQLite raises OperationalError if the column already exists; we swallow it.
        try:
            conn.execute("ALTER TABLE boltz_cache ADD COLUMN product_name TEXT")
        except Exception:
            pass


def _disk_cache_get(db_path: str, smiles: str, protein: str) -> Optional[float]:
    """Return cached Boltz score for (smiles, protein), or None on miss/error."""
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT score FROM boltz_cache WHERE smiles=? AND protein=?",
                (smiles, protein),
            ).fetchone()
        return float(row[0]) if row else None
    except Exception:
        return None


def _disk_cache_put(db_path: str, smiles: str, protein: str, score: float, product_name: Optional[str] = None) -> None:
    """Upsert a Boltz score into the persistent cache (silently ignores errors)."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO boltz_cache (smiles, protein, score, product_name) VALUES (?,?,?,?)",
                (smiles, protein, score, product_name),
            )
    except Exception:
        pass


def _cleanup_boltz_cache(db_path: str, keep_protein: str, max_age_days: int = 14) -> None:
    """
    Prune stale entries from the persistent Boltz cache.

    Removes rows where:
    - The protein does not match the current weekly target (old target entries).
    - The entry is older than max_age_days (catches any leftover same-protein entries
      from weeks when the target happened to recur).

    Called once at miner startup. Silently ignores errors.
    """
    try:
        cutoff_ts = int(__import__('time').time()) - max_age_days * 86400
        with sqlite3.connect(db_path) as conn:
            deleted = conn.execute(
                "DELETE FROM boltz_cache WHERE protein != ? OR ts < ?",
                (keep_protein, cutoff_ts),
            ).rowcount
        if deleted:
            bt.logging.info(f"Boltz cache cleanup: removed {deleted} stale entries (non-{keep_protein} or >{max_age_days}d old).")
    except Exception:
        pass


def _disk_cache_get_best(db_path: str, protein: str) -> Optional[Tuple[float, str, str]]:
    """
    Return (score, smiles, product_name) for the highest-scoring cached entry for
    *protein* that has a submittable product_name (non-NULL).

    Used by _apply_warm_start to seed candidate_product at epoch start.
    Returns None on cache miss or error.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT score, smiles, product_name FROM boltz_cache "
                "WHERE protein=? AND product_name IS NOT NULL "
                "ORDER BY score DESC LIMIT 1",
                (protein,),
            ).fetchone()
        if row:
            return float(row[0]), str(row[1]), str(row[2])
        return None
    except Exception:
        return None


def _apply_warm_start(state: Dict[str, Any], db_path: str, protein: str) -> None:
    """
    Pre-populate state['candidate_product'] from the best Boltz-scored molecule
    stored in the disk cache for *protein*.

    Called at epoch start so the miner has a valid fallback submission from
    block 1 — before PSICHIC streaming has had time to find new candidates.

    Does not set best_score (left at -inf), so the first valid PSICHIC result
    supersedes this molecule automatically within the first few minutes of streaming.
    On epoch 2+ this guarantees we always have something Boltz-validated to submit
    even if streaming is slow or the miner restarts late in an epoch.
    """
    best = _disk_cache_get_best(db_path, protein)
    if best is None:
        return
    score, smiles, product_name = best
    state['candidate_product'] = product_name
    bt.logging.info(
        f"[WarmStart] Seeded candidate_product from disk cache: "
        f"{product_name} (boltz_score={score:.4f}, target={protein})"
    )


# ----------------------------------------------------------------------------
# 1. CONFIG & ARGUMENT PARSING
# ----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments and merges with config defaults.

    Returns:
        argparse.Namespace: The combined configuration object.
    """
    parser = argparse.ArgumentParser()
    # Add override arguments for network.
    parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
    # Adds override arguments for netuid.
    parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
    # Bittensor standard argument additions.
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    # Parse combined config
    config = bt.config(parser)

    # Load protein selection params
    config.update(load_config())

    # Final logging dir
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            'miner',
        )
    )

    # Ensure the logging directory exists.
    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    """
    Constructs the path for GitHub operations from environment variables.
    
    Returns:
        str: The fully qualified GitHub path (owner/repo/branch/path).
    Raises:
        ValueError: If the final path exceeds 100 characters.
    """
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')  # e.g., "nova"
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')  # e.g., "main"
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER')  # e.g., "metanova-labs"
    github_repo_path = os.environ.get('GITHUB_REPO_PATH')  # e.g., "data/results" or ""

    if github_repo_name is None or github_repo_branch is None or github_repo_owner is None:
        raise ValueError("Missing one or more GitHub environment variables (GITHUB_REPO_*)")

    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path is too long. Please shorten it to 100 characters or less.")

    return github_path


# ----------------------------------------------------------------------------
# 2. LOGGING SETUP
# ----------------------------------------------------------------------------

def setup_logging(config: argparse.Namespace) -> None:
    """
    Sets up Bittensor logging.

    Args:
        config (argparse.Namespace): The miner configuration object.
    """
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid} on network: {config.subtensor.network} with config:")
    bt.logging.info(config)


# ----------------------------------------------------------------------------
# 3. BITTENSOR & NETWORK SETUP
# ----------------------------------------------------------------------------

async def setup_bittensor_objects(config: argparse.Namespace) -> Tuple[Any, Any, Any, int, int]:
    """
    Initializes wallet, subtensor, and metagraph. Fetches the epoch length
    and calculates the miner UID.

    Args:
        config (argparse.Namespace): The miner configuration object.

    Returns:
        tuple: A 5-element tuple of
            (wallet, subtensor, metagraph, miner_uid, epoch_length).
    """
    bt.logging.info("Setting up Bittensor objects.")

    # Initialize wallet
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # Initialize subtensor (asynchronously)
    try:
        async with bt.async_subtensor(network=config.network) as subtensor:
            bt.logging.info(f"Connected to subtensor network: {config.network}")
            
            # Sync metagraph
            metagraph = await subtensor.metagraph(config.netuid)
            await metagraph.sync()
            bt.logging.info(f"Metagraph synced successfully.")

            bt.logging.info(f"Subtensor: {subtensor}")
            bt.logging.info(f"Metagraph synced: {metagraph}")

            # Get miner UID
            miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            bt.logging.info(f"Miner UID: {miner_uid}")

            # Query epoch length
            node = SubstrateInterface(url=config.network)
            # Set epoch_length to tempo + 1
            epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1
            bt.logging.info(f"Epoch length query successful: {epoch_length} blocks")

        return wallet, subtensor, metagraph, miner_uid, epoch_length
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        bt.logging.error("Please check your network connection and the subtensor network status")
        raise

# ----------------------------------------------------------------------------
# 4. DATA SETUP
# ----------------------------------------------------------------------------

def stream_random_chunk_from_dataset(dataset_repo: str, chunk_size: int) -> Any:
    """
    Streams a random chunk from the specified Hugging Face dataset repo.

    Args:
        dataset_repo (str): Hugging Face dataset repository path (user/repo).
        chunk_size (int): Size of each chunk to stream.

    Returns:
        Any: A batched (chunked) dataset iterator.
    """
    files = list_repo_files(dataset_repo, repo_type='dataset')
    files = [file for file in files if file.endswith('.csv')]
    random_file = random.choice(files)

    dataset_dict = load_dataset(
        dataset_repo,
        data_files={'train': random_file},
        streaming=True,
    )
    dataset = dataset_dict['train']
    batched = dataset.batch(chunk_size)
    return batched


# ----------------------------------------------------------------------------
# 5. INFERENCE AND SUBMISSION LOGIC
# ----------------------------------------------------------------------------

async def run_psichic_model_loop(state: Dict[str, Any]) -> None:
    """
    Continuously runs the PSICHIC model on batches of molecules from Hugging Face dataset.
    Updates the best candidate whenever a higher score is found, but only submits when close to epoch end.

    Args:
        state (dict): A shared state dict containing references to:
            'chunk_size', 'hugging_face_dataset_repo', 'psichic_models', 'current_challenge_targets',
            'current_challenge_antitargets', 'psichic_result_column_name', 'best_score',
            'candidate_product', 'submission_interval', 'last_submission_time',
            'last_submitted_product', 'shutdown_event', etc.
    """
    bt.logging.info("Starting PSICHIC model inference loop.")

    # Defined once here so we don't recreate the closure on every chunk iteration.
    # Reads state['config'] on each call — picks up any live config changes correctly.
    def _pharma_ok(smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        # Validator-enforced: banned atom types (e.g. Se)
        _banned = getattr(state['config'], 'banned_atom_types', [])
        if _banned and any(a.GetSymbol() in _banned for a in mol.GetAtoms()):
            return False
        # Validator-enforced: rotatable bond bounds
        _rot = Descriptors.NumRotatableBonds(mol)
        _min_rb = getattr(state['config'], 'min_rotatable_bonds', 1)
        _max_rb = getattr(state['config'], 'max_rotatable_bonds', None)
        if _rot < _min_rb or (_max_rb is not None and _rot > _max_rb):
            return False
        # Lipinski-inspired drug-likeness (fast heuristic filter).
        # HBD minimum is 0 — aromatic scaffolds with no NH/OH groups (e.g.
        # N-alkylated heterocycles) are valid binders and must not be excluded.
        return (
            Descriptors.NumHDonors(mol) <= 5
            and 2 <= Descriptors.NumHAcceptors(mol) <= 10
            and -1.0 <= Descriptors.MolLogP(mol) <= 5.0
        )

    while not state['shutdown_event'].is_set():
        try:
            # Create a fresh iterator each outer cycle so that when one streaming
            # file is exhausted we immediately pick a new random file rather than
            # spinning on an empty iterator.
            dataset_iter = stream_random_chunk_from_dataset(
                dataset_repo=state['hugging_face_dataset_repo'],
                chunk_size=state['chunk_size']
            )
            for chunk in dataset_iter:
                if state['shutdown_event'].is_set():
                    break

                df = pd.DataFrame.from_dict(chunk)
                # Clean data
                df['product_name'] = df['product_name'].apply(lambda x: x.replace('"', ''))
                df['product_smiles'] = df['product_smiles'].apply(lambda x: x.replace('"', ''))

                # Filter by heavy atom count (min and optional max for Boltz efficiency)
                df['heavy_atoms'] = df['product_smiles'].apply(lambda x: get_heavy_atom_count(x))
                df = df[df['heavy_atoms'] >= state['config'].min_heavy_atoms]
                max_ha = getattr(state['config'], 'max_heavy_atoms', None)
                if max_ha:
                    df = df[df['heavy_atoms'] <= max_ha]
                if df.empty or len(df) < state['config'].num_molecules:
                    continue

                # Filter for Boltz-2 compatibility (validator scores with Boltz-2)
                df = df[df['product_smiles'].apply(lambda x: is_boltz_safe_smiles(x)[0])]
                if df.empty or len(df) < state['config'].num_molecules:
                    continue

                # Combined pre-filter: validator constraints + Lipinski drug-likeness.
                # _pharma_ok is defined once at function entry (above the while loop).
                df = df[df['product_smiles'].apply(_pharma_ok)]
                if df.empty or len(df) < state['config'].num_molecules:
                    continue

                # Run inference for all targets and antitargets
                target_scores = []
                antitarget_scores = []

                # Score against all target proteins
                for target_protein in state['current_challenge_targets']:
                    if target_protein not in state['psichic_models']:
                        try:
                            target_sequence = get_sequence_from_protein_code(target_protein)
                            model = PsichicWrapper()
                            model.run_challenge_start(target_sequence)
                            state['psichic_models'][target_protein] = model
                            bt.logging.info(f"Initialized model for target: {target_protein}")
                        except Exception as e:
                            bt.logging.error(f"Error initializing model for target {target_protein}: {e}")
                            continue

                    scores = state['psichic_models'][target_protein].run_validation(df['product_smiles'].tolist())
                    target_scores.append(scores[state['psichic_result_column_name']])

                # Score against all antitarget proteins
                for antitarget_protein in state['current_challenge_antitargets']:
                    if antitarget_protein not in state['psichic_models']:
                        try:
                            antitarget_sequence = get_sequence_from_protein_code(antitarget_protein)
                            model = PsichicWrapper()
                            model.run_challenge_start(antitarget_sequence)
                            state['psichic_models'][antitarget_protein] = model
                            bt.logging.info(f"Initialized model for antitarget: {antitarget_protein}")
                        except Exception as e:
                            bt.logging.error(f"Error initializing model for antitarget {antitarget_protein}: {e}")
                            continue

                    scores = state['psichic_models'][antitarget_protein].run_validation(df['product_smiles'].tolist())
                    antitarget_scores.append(scores[state['psichic_result_column_name']])

                # Calculate average scores
                df['target_affinity'] = pd.DataFrame(target_scores).mean(axis=0)
                # Guard: if no antitargets are configured, default penalty to 0.
                df['antitarget_affinity'] = (
                    pd.DataFrame(antitarget_scores).mean(axis=0)
                    if antitarget_scores else 0.0
                )
                # Ligand-efficiency scoring: divide by heavy atom count so the PSICHIC
                # pre-filter uses the same per-atom normalisation as the Boltz-2 scoring
                # formula (affinity_probability_binary - affinity_pred_value) / heavy_atoms.
                # This makes the top-N candidates sent to Boltz more likely to win.
                df['combined_score'] = (
                    df['target_affinity'] - state['config'].antitarget_weight * df['antitarget_affinity']
                ) / df['heavy_atoms']

                # Sort by combined score
                df.sort_values(by=['combined_score'], ascending=[False], inplace=True)
                df.reset_index(drop=True, inplace=True)

                # Select top 10 molecules
                top_molecules = df.iloc[:10]

                # ---------------------------------------------------------------
                # Global candidate pool: accumulate the best molecules seen
                # across ALL chunks this epoch (not just the latest best batch).
                # Capped at 20 entries, sorted by ligand-efficiency combined_score.
                # Boltz pre-scoring draws from this pool, ensuring it always sees
                # the highest-quality candidates regardless of when they appeared.
                # ---------------------------------------------------------------
                if state.get('global_candidate_pool') is None or state['global_candidate_pool'].empty:
                    state['global_candidate_pool'] = top_molecules.copy()
                else:
                    combined_pool = pd.concat(
                        [state['global_candidate_pool'], top_molecules], ignore_index=True
                    )
                    combined_pool.drop_duplicates(subset=['product_name'], inplace=True)
                    combined_pool.sort_values(by=['combined_score'], ascending=False, inplace=True)
                    state['global_candidate_pool'] = combined_pool.head(20).reset_index(drop=True)

                # ---------------------------------------------------------------
                # SAVI stream pool: accumulates the top-5000 PSICHIC-scored
                # molecules seen this epoch, sorted by combined_score.
                # Keeping the highest-quality molecules (rather than the first
                # seen) ensures SALSA/GA nearest-neighbor search operates on the
                # best chemical space available at trigger time.
                # Once both SALSA and GA have fired (salsa_run_this_epoch AND
                # ga_run_this_epoch), the pool is no longer read — skip the
                # concat/sort to save CPU for the remaining streaming window.
                # ---------------------------------------------------------------
                if not (state.get('salsa_run_this_epoch') and state.get('ga_run_this_epoch')):
                    if state.get('savi_stream_pool') is None or state['savi_stream_pool'].empty:
                        state['savi_stream_pool'] = df.copy()
                    else:
                        _pool_combined = pd.concat(
                            [state['savi_stream_pool'], df], ignore_index=True
                        )
                        _pool_combined.drop_duplicates(subset=['product_name'], inplace=True)
                        _pool_combined.sort_values('combined_score', ascending=False, inplace=True)
                        state['savi_stream_pool'] = _pool_combined.head(5000).reset_index(drop=True)

                if not top_molecules.empty:
                    entropy = compute_maccs_entropy(top_molecules['product_smiles'].tolist())
                    scores_sum = top_molecules['combined_score'].sum()
                    
                    if scores_sum > state['config'].entropy_bonus_threshold:
                        final_score = scores_sum * (state['config'].entropy_start_weight + entropy)
                    else:
                        final_score = scores_sum

                    if final_score > state['best_score']:
                        state['best_score'] = final_score
                        state['candidate_molecules'] = top_molecules.copy()
                        state['candidate_product'] = ','.join(top_molecules['product_name'].tolist())
                        state['boltz_prescored'] = False  # Re-run Boltz for new best candidate
                        bt.logging.info(f"New best score: {state['best_score']}, Candidates: {state['candidate_product']}")

                    # Only submit if we're close to epoch end (20 blocks away)
                    # Check if we're close to epoch end (20 blocks away)
                    current_block = await state['subtensor'].get_current_block()
                    next_epoch_block = ((current_block // state['epoch_length']) + 1) * state['epoch_length']
                    blocks_until_epoch = next_epoch_block - current_block

                    bt.logging.debug(f"Current block: {current_block}, Epoch length: {state['epoch_length']}, Next epoch block: {next_epoch_block}, Blocks until epoch: {blocks_until_epoch}")

                    # ---------------------------------------------------------------
                    # SALSA trigger: run once per epoch when the stream pool is
                    # large enough to support meaningful NN search and we still have
                    # time before Boltz kicks in.
                    #   - Requires ≥500 molecules in the stream pool (enough for NN).
                    #   - Fires only when > boltz_trigger * 1.5 blocks remain, so
                    #     SALSA hits are in global_candidate_pool before Boltz starts.
                    #   - One-shot per epoch (salsa_run_this_epoch flag).
                    # ---------------------------------------------------------------
                    salsa_pool = state.get('savi_stream_pool')
                    salsa_pool_size = 0 if salsa_pool is None else len(salsa_pool)
                    boltz_trigger = state.get('boltz_trigger_blocks', 100)
                    salsa_threshold = int(boltz_trigger * 1.5)
                    if (
                        not state.get('salsa_run_this_epoch', False)
                        and salsa_pool_size >= 500
                        and blocks_until_epoch > salsa_threshold
                        and state.get('global_candidate_pool') is not None
                        and not state['global_candidate_pool'].empty
                    ):
                        state['salsa_run_this_epoch'] = True
                        # Multi-seed SALSA: run from up to top-3 candidates so we
                        # explore three distinct chemical neighbourhoods in one pass.
                        # Runtime: ~3 × 180 ms = ~540 ms CPU — negligible vs Boltz.
                        _n_seeds = min(3, len(state['global_candidate_pool']))
                        _seeds = state['global_candidate_pool'].head(_n_seeds)['product_smiles'].tolist()
                        bt.logging.info(
                            f"SALSA: triggering with {salsa_pool_size}-molecule pool, "
                            f"{blocks_until_epoch} blocks remaining (threshold={salsa_threshold}), "
                            f"{_n_seeds} seed(s)..."
                        )
                        try:
                            _all_salsa = []
                            for _seed_smiles in _seeds:
                                _hits = await asyncio.to_thread(
                                    run_salsa_search,
                                    _seed_smiles,
                                    salsa_pool,
                                    3,   # rounds
                                    60,  # n_perturb
                                    5,   # top_k
                                )
                                if not _hits.empty:
                                    _all_salsa.append(_hits)

                            if _all_salsa:
                                salsa_hits = pd.concat(_all_salsa, ignore_index=True)
                                salsa_hits.drop_duplicates(subset=['product_name'], inplace=True)
                                salsa_hits.sort_values('combined_score', ascending=False, inplace=True)
                                salsa_hits = salsa_hits.head(5 * _n_seeds).reset_index(drop=True)
                            else:
                                salsa_hits = pd.DataFrame()

                            if not salsa_hits.empty:
                                bt.logging.info(
                                    f"SALSA: {len(salsa_hits)} hits found across {_n_seeds} seed(s); "
                                    f"merging into global_candidate_pool."
                                )
                                combined_gcp = pd.concat(
                                    [state['global_candidate_pool'], salsa_hits],
                                    ignore_index=True
                                )
                                combined_gcp.drop_duplicates(subset=['product_name'], inplace=True)
                                combined_gcp.sort_values('combined_score', ascending=False, inplace=True)
                                state['global_candidate_pool'] = combined_gcp.head(20).reset_index(drop=True)
                                bt.logging.info(
                                    f"SALSA: global_candidate_pool now has "
                                    f"{len(state['global_candidate_pool'])} entries."
                                )
                            else:
                                bt.logging.info("SALSA: no hits found.")
                        except Exception as _salsa_err:
                            bt.logging.error(f"SALSA error: {_salsa_err}")

                    # ---------------------------------------------------------------
                    # GradientGA trigger: run once per epoch after SALSA has had a
                    # chance to populate the stream pool further.  Fires when:
                    #   - pool >= 500 molecules (same gate as SALSA)
                    #   - blocks_until_epoch > boltz_trigger + 20 (runs before Boltz)
                    #   - salsa_run_this_epoch (SALSA must have run first so its hits
                    #     are already in global_candidate_pool)
                    #   - ga_run_this_epoch == False (one-shot per epoch)
                    # Runtime: ~1-3 s CPU for 5 generations — negligible vs Boltz.
                    # ---------------------------------------------------------------
                    ga_pool = state.get('savi_stream_pool')
                    ga_pool_size = 0 if ga_pool is None else len(ga_pool)
                    boltz_trigger_ga = state.get('boltz_trigger_blocks', 100)
                    if (
                        not state.get('ga_run_this_epoch', False)
                        and state.get('salsa_run_this_epoch', False)
                        and ga_pool_size >= 500
                        and blocks_until_epoch > boltz_trigger_ga + 20
                        and state.get('global_candidate_pool') is not None
                        and not state['global_candidate_pool'].empty
                    ):
                        state['ga_run_this_epoch'] = True
                        bt.logging.info(
                            f"GradientGA: triggering with {ga_pool_size}-molecule pool, "
                            f"{blocks_until_epoch} blocks remaining..."
                        )
                        try:
                            ga_hits = await asyncio.to_thread(
                                run_gradient_ga,
                                state['global_candidate_pool'],
                                ga_pool,
                                5,   # n_generations
                                50,  # pop_size
                                5,   # top_k
                            )
                            if not ga_hits.empty:
                                bt.logging.info(
                                    f"GradientGA: {len(ga_hits)} hits found; "
                                    f"merging into global_candidate_pool."
                                )
                                combined_gcp = pd.concat(
                                    [state['global_candidate_pool'], ga_hits],
                                    ignore_index=True,
                                )
                                combined_gcp.drop_duplicates(
                                    subset=['product_name'], inplace=True
                                )
                                combined_gcp.sort_values(
                                    'combined_score', ascending=False, inplace=True
                                )
                                state['global_candidate_pool'] = (
                                    combined_gcp.head(20).reset_index(drop=True)
                                )
                                bt.logging.info(
                                    f"GradientGA: global_candidate_pool now has "
                                    f"{len(state['global_candidate_pool'])} entries."
                                )
                            else:
                                bt.logging.info("GradientGA: no hits found.")
                        except Exception as _ga_err:
                            bt.logging.error(f"GradientGA error: {_ga_err}")

                    # Trigger Boltz-2 pre-scoring when approaching epoch end.
                    # Threshold starts at 100 blocks (20 min); after the first run it is
                    # updated adaptively based on measured GPU time so fast hardware
                    # (A100 ~4 min) doesn't sit idle for 16 extra minutes.
                    boltz_trigger = state.get('boltz_trigger_blocks', 100)
                    if (
                        state['candidate_product']
                        and blocks_until_epoch <= boltz_trigger
                        and not state.get('boltz_prescored', False)
                    ):
                        bt.logging.info(
                            f"Triggering Boltz-2 pre-scoring with "
                            f"{blocks_until_epoch} blocks until epoch end..."
                        )
                        state['boltz_prescored'] = True
                        try:
                            # Dynamic candidate budget: fill the available epoch
                            # window rather than always scoring a fixed 5 molecules.
                            # Default to 150 s/mol (RTX 3090 worst-case) until the
                            # first real measurement is stored in state.
                            _t_per_mol = state.get('boltz_time_per_mol', 150.0)
                            _avail_secs = max(0.0, blocks_until_epoch * 12 - 240)
                            _dyn_max = max(3, min(20, int(_avail_secs / _t_per_mol)))
                            bt.logging.info(
                                f"Dynamic Boltz budget: {_dyn_max} candidates "
                                f"({_avail_secs:.0f}s available, ~{_t_per_mol:.0f}s/mol)"
                            )
                            await run_boltz_prescoring(state, max_candidates=_dyn_max)
                        except Exception as e:
                            bt.logging.error(f"Boltz-2 pre-scoring error: {e}")
                            traceback.print_exc()

                    if state['candidate_product'] and blocks_until_epoch <= 20:
                        bt.logging.info(f"Close to epoch end ({blocks_until_epoch} blocks remaining), attempting submission...")
                        if state['candidate_product'] != state['last_submitted_product']:
                            bt.logging.info("Attempting to submit new candidate...")
                            try:
                                await submit_response(state)
                            except Exception as e:
                                bt.logging.error(f"Error submitting response: {e}")
                        else:
                            bt.logging.info("Skipping submission - same product as last submission")

                await asyncio.sleep(2)

        except Exception as e:
            bt.logging.error(f"Error in PSICHIC model loop: {e}")
            traceback.print_exc()
            state['shutdown_event'].set()


async def submit_response(state: Dict[str, Any]) -> None:
    """
    Encrypts and submits the current candidate product as a chain commitment and uploads
    the encrypted response to GitHub. If the chain accepts the commitment, we finalize it.

    Args:
        state (dict): Shared state dictionary containing references to:
            'bdt', 'miner_uid', 'candidate_product', 'subtensor', 'wallet', 'config',
            'github_path', etc.
    """
    candidate_product = state['candidate_product']
    if not candidate_product:
        bt.logging.warning("No candidate product to submit")
        return

    bt.logging.info(f"Starting submission process for product: {candidate_product}")
    
    # 1) Encrypt the response
    current_block = await state['subtensor'].get_current_block()
    encrypted_response = state['bdt'].encrypt(state['miner_uid'], candidate_product, current_block)
    bt.logging.info(f"Encrypted response generated successfully")

    # 2) Create temp file, write content
    tmp_file = tempfile.NamedTemporaryFile(delete=True)
    with open(tmp_file.name, 'w+') as f:
        f.write(str(encrypted_response))
        f.flush()

        # Read, base64-encode
        f.seek(0)
        content_str = f.read()
        encoded_content = base64.b64encode(content_str.encode()).decode()

        # Generate short hash-based filename
        filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
        commit_content = f"{state['github_path']}/{filename}.txt"
        bt.logging.info(f"Prepared commit content: {commit_content}")

        # 3) Attempt chain commitment
        bt.logging.info(f"Attempting chain commitment...")
        try: 
            commitment_status = await state['subtensor'].set_commitment(
                wallet=state['wallet'],
                netuid=state['config'].netuid,
                data=commit_content
            )
            bt.logging.info(f"Chain commitment status: {commitment_status}")
        except MetadataError:
            bt.logging.info("Too soon to commit again. Will keep looking for better candidates.")
            return

        # 4) If chain commitment success, upload to GitHub
        if commitment_status:
            try:
                bt.logging.info(f"Commitment set successfully for {commit_content}")
                bt.logging.info("Attempting GitHub upload...")
                github_status = upload_file_to_github(filename, encoded_content)
                if github_status:
                    bt.logging.info(f"File uploaded successfully to {commit_content}")
                    state['last_submitted_product'] = candidate_product
                    state['last_submission_time'] = datetime.datetime.now()
                else:
                    bt.logging.error(f"Failed to upload file to GitHub for {commit_content}")
            except Exception as e:
                bt.logging.error(f"Failed to upload file for {commit_content}: {e}")


# ----------------------------------------------------------------------------
# 6. BOLTZ-2 PRE-SCORING
# ----------------------------------------------------------------------------

def _scaffold_diverse_candidates(
    df: pd.DataFrame,
    max_k: int,
    smiles_col: str = 'product_smiles',
    name_col: str = 'product_name',
) -> pd.DataFrame:
    """
    Select up to *max_k* rows from *df* (sorted by score desc) such that
    each selected molecule has a distinct Murcko scaffold from all previously
    selected ones.

    When SALSA or GA converges to a single chemical region, the top-N
    candidates by PSICHIC score can all share one scaffold.  Scoring 5
    near-identical molecules with Boltz-2 wastes the GPU budget: they'll
    receive similar affinity estimates and only one can be submitted.

    Strategy: greedy first-pass picks candidates with unseen scaffolds.  A
    fill-pass then appends remaining top-ranked candidates (scaffold repeats
    allowed) until *max_k* slots are filled — so if the pool is small or
    chemically homogeneous we still return max_k candidates.

    Fallback: any molecule whose SMILES can't be parsed gets a unique random
    key and is always admitted, so malformed entries don't block the fill.
    """
    from rdkit.Chem import MurckoScaffold

    selected_rows: list = []
    seen_scaffolds: set = set()
    seen_names: set = set()

    # First pass: greedy scaffold-diversity selection
    for _, row in df.iterrows():
        if len(selected_rows) >= max_k:
            break
        smiles = row.get(smiles_col, '')
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Unparseable — admit unconditionally (Boltz-safe filter upstream
            # would have already dropped truly invalid SMILES).
            selected_rows.append(row)
            seen_names.add(row.get(name_col, ''))
            continue
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold) if scaffold is not None else smiles
        except Exception:
            scaffold_smi = smiles

        if scaffold_smi not in seen_scaffolds:
            seen_scaffolds.add(scaffold_smi)
            selected_rows.append(row)
            seen_names.add(row.get(name_col, ''))

    # Fill-pass: if scaffold diversity exhausted the pool early, admit repeats
    if len(selected_rows) < max_k:
        for _, row in df.iterrows():
            if len(selected_rows) >= max_k:
                break
            name = row.get(name_col, '')
            if name not in seen_names:
                selected_rows.append(row)
                seen_names.add(name)

    if not selected_rows:
        return pd.DataFrame()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


async def run_boltz_prescoring(state: Dict[str, Any], max_candidates: int = 5) -> None:
    """
    Runs Boltz-2 affinity predictions on the top PSICHIC candidates and reorders
    state['candidate_product'] so the highest-scoring Boltz-2 molecule comes first.

    Implements an ANYTIME / incremental scoring strategy: candidates are scored
    one-by-one in PSICHIC-rank order and state['candidate_product'] is reordered
    immediately after each molecule is scored.  This means even if the epoch ends
    mid-run, the submission already reflects the best Boltz score seen so far —
    not just the raw PSICHIC ranking.

    Results are cached in state['boltz_score_cache'] keyed by
    (canonical_smiles, protein_code) so molecules already scored in this session
    skip inference entirely — saving 45-150 s of GPU time per cache hit.

    The validator uses sample_selection="first" with num_molecules_boltz=1, so the
    molecule placed first in the submission is the one that determines our Boltz score.
    Boltz scoring formula: (affinity_probability_binary - affinity_pred_value) / heavy_atom_count

    Args:
        state: Shared miner state dict.
        max_candidates: Maximum number of PSICHIC top-molecules to score with Boltz-2.
    """
    # Prefer global candidate pool (spans entire epoch, sorted by ligand efficiency)
    # so Boltz always evaluates the best molecules seen so far, not just those from
    # the most recent best-batch.  Fall back to current-batch candidates if needed.
    candidates = state.get('global_candidate_pool')
    if candidates is None or candidates.empty:
        candidates = state.get('candidate_molecules')
    if candidates is None or candidates.empty:
        bt.logging.warning("Boltz-2 pre-scoring: no candidate molecules available.")
        return

    # Pull a wider slice (3× budget) so the scaffold-diversity filter has
    # candidates to choose from even when SALSA has converged to one region.
    # global_candidate_pool is capped at 20 rows, so this is at most 20.
    candidates = candidates.head(max_candidates * 3).copy()
    safe_mask = candidates['product_smiles'].apply(lambda s: is_boltz_safe_smiles(s)[0])
    candidates = candidates[safe_mask].reset_index(drop=True)

    if candidates.empty:
        bt.logging.warning("Boltz-2 pre-scoring: no Boltz-safe candidates, keeping PSICHIC ranking.")
        return

    # Scaffold-diverse selection: prefer candidates with distinct Murcko
    # scaffolds so Boltz-2 explores different binding modes per epoch budget.
    pre_div = len(candidates)
    candidates = _scaffold_diverse_candidates(candidates, max_candidates)
    post_div = len(candidates)
    if pre_div != post_div:
        bt.logging.info(
            f"Boltz-2 scaffold diversity: selected {post_div} from {pre_div} Boltz-safe candidates "
            f"({pre_div - post_div} near-duplicate scaffold(s) deferred)"
        )

    protein = state['config'].weekly_target
    boltz_cache: Dict[Tuple[str, str], float] = state.setdefault('boltz_score_cache', {})
    db_path: str = state.get('boltz_cache_db', BOLTZ_CACHE_DB)

    # Subnet config reused for every single-molecule Boltz call
    subnet_config = {
        'weekly_target': protein,
        'binding_pocket': state['config'].binding_pocket,
        'max_distance': state['config'].max_distance,
        'force': state['config'].force,
        'num_molecules_boltz': 1,
        'sample_selection': 'first',
        'boltz_metric': state['config'].boltz_metric,
        'combination_strategy': state['config'].combination_strategy,
    }

    # One wrapper instance shared across all molecules to avoid repeated
    # directory setup; remove_files=true in boltz_config.yaml ensures each
    # call cleans up its YAML/output before the next one starts.
    wrapper = BoltzWrapper()

    # Accumulate scores as we go; allows immediate reorder after each hit.
    all_scores: Dict[str, float] = {}

    def _reorder_submission(scores: Dict[str, float]) -> None:
        """Put the best Boltz-scored molecule first in state['candidate_product']."""
        valid = {s: v for s, v in scores.items() if math.isfinite(v)}
        if not valid:
            return
        best_smiles = max(valid, key=valid.get)
        best_row = candidates[candidates['product_smiles'] == best_smiles]
        if best_row.empty:
            return
        best_name = best_row.iloc[0]['product_name']
        best_score = valid[best_smiles]
        original_names = state['candidate_product'].split(',')
        reordered = [best_name] + [n for n in original_names if n != best_name]
        state['candidate_product'] = ','.join(reordered)
        bt.logging.info(
            f"  → submission updated after {len(scores)}/{len(candidates)} scored: "
            f"best={best_name} (boltz_score={best_score:.4f})"
        )

    bt.logging.info(
        f"Boltz-2 anytime pre-scoring: {len(candidates)} candidates for target {protein}..."
    )

    for i, row in candidates.iterrows():
        smiles = row['product_smiles']
        canon = get_canonical_smiles(smiles)
        key = (canon, protein)

        # --- Cache lookup (in-memory → disk → GPU inference) ---
        if key in boltz_cache:
            score = boltz_cache[key]
            bt.logging.debug(f"[{i+1}/{len(candidates)}] in-memory cache hit: {score:.4f}")
        else:
            disk_score = _disk_cache_get(db_path, canon, protein)
            if disk_score is not None:
                boltz_cache[key] = disk_score
                score = disk_score
                bt.logging.debug(f"[{i+1}/{len(candidates)}] disk cache hit: {score:.4f}")
            else:
                # Before launching GPU inference, confirm the epoch hasn't ended.
                # The anytime guarantee means whatever we've already scored is at
                # position 0, so there's no value running Boltz past the boundary.
                try:
                    _curr_blk = await state['subtensor'].get_current_block()
                    _next_ep = ((_curr_blk // state['epoch_length']) + 1) * state['epoch_length']
                    if _next_ep - _curr_blk < 5:
                        bt.logging.info(
                            f"[{i+1}/{len(candidates)}] epoch ends in <5 blocks — "
                            f"stopping Boltz after {len(all_scores)}/{len(candidates)} scored."
                        )
                        break
                except Exception:
                    pass  # subtensor unavailable; proceed

                # Cache miss: run Boltz for this single molecule
                bt.logging.info(f"[{i+1}/{len(candidates)}] running Boltz-2 inference...")
                uid = 0
                valid_molecules_by_uid = {
                    uid: {"smiles": [smiles], "names": [row['product_name']]}
                }
                score_dict: Dict[str, Any] = {uid: {}}
                try:
                    await asyncio.to_thread(
                        wrapper.score_molecules_target,
                        valid_molecules_by_uid,
                        score_dict,
                        subnet_config,
                        '0x' + '0' * 64,
                    )
                    mol_scores = wrapper.per_molecule_metric.get(uid, {})
                    score = mol_scores.get(smiles, -math.inf)

                    # Persist to both cache layers (product_name enables warm-start §AA)
                    boltz_cache[key] = score
                    _pname = row.get('product_name')
                    if not isinstance(_pname, str):
                        _pname = None
                    _disk_cache_put(db_path, canon, protein, score, product_name=_pname)

                    # Adaptive trigger: one molecule gives the most accurate per-mol timing
                    elapsed = wrapper.last_inference_duration
                    if elapsed > 0:
                        state['boltz_time_per_mol'] = elapsed  # persist for dynamic budget calc
                        adaptive_trigger = int(elapsed * max_candidates / 12) + 20
                        state['boltz_trigger_blocks'] = max(adaptive_trigger, 30)
                        bt.logging.info(
                            f"  adaptive timing: {elapsed:.1f}s/mol → "
                            f"trigger={state['boltz_trigger_blocks']} blocks"
                        )
                except Exception as e:
                    bt.logging.error(f"  Boltz-2 inference failed: {e}")
                    traceback.print_exc()
                    score = -math.inf

        all_scores[smiles] = score

        # Reorder submission immediately — anytime guarantee: if epoch ends
        # after this molecule, the best Boltz score seen so far is at position 0.
        _reorder_submission(all_scores)

    # Final summary
    valid_scores = {s: v for s, v in all_scores.items() if v != -math.inf}
    bt.logging.info(
        f"Boltz-2 anytime pre-scoring complete: "
        f"{len(valid_scores)}/{len(candidates)} molecules scored."
    )

    # §FF: Boltz-guided SALSA — second SALSA pass seeded from the best Boltz molecule.
    # The main loop above uses the PSICHIC-ranked candidate pool as seeds; PSICHIC and
    # Boltz-2 correlate imperfectly, so the validated Boltz winner may occupy a different
    # region of chemical space.  By running SALSA from the actual best-Boltz SMILES we
    # explore its chemical neighbourhood and may find SAVI-2020 molecules that score
    # even better — without any additional PSICHIC overhead.
    # Only fires when the epoch has ≥2 mol-lengths + 2 min of runway remaining.
    _ff_best_smiles = max(all_scores, key=lambda s: all_scores.get(s, -math.inf), default=None)
    _savi_pool_ff = state.get('savi_stream_pool')
    if (
        _ff_best_smiles is not None
        and math.isfinite(all_scores.get(_ff_best_smiles, -math.inf))
        and _savi_pool_ff is not None
        and not _savi_pool_ff.empty
    ):
        try:
            _curr_blk_ff = await state['subtensor'].get_current_block()
            _next_ep_ff = ((_curr_blk_ff // state['epoch_length']) + 1) * state['epoch_length']
            _remaining_s_ff = (_next_ep_ff - _curr_blk_ff) * 12
            _t_per_mol_ff = state.get('boltz_time_per_mol', 150.0)
            if _remaining_s_ff > _t_per_mol_ff * 2 + 120:
                bt.logging.info(
                    f"§FF Boltz-guided SALSA: {_remaining_s_ff:.0f}s remaining — "
                    f"seeding 2-round SALSA from best Boltz molecule..."
                )
                ff_salsa_hits = await asyncio.to_thread(
                    run_salsa_search,
                    _ff_best_smiles,
                    _savi_pool_ff,
                    2,   # rounds (fewer — time is limited)
                    60,  # n_perturb
                    3,   # top_k
                )
                if not ff_salsa_hits.empty:
                    bt.logging.info(
                        f"§FF: {len(ff_salsa_hits)} Boltz-guided SALSA hits — scoring with Boltz..."
                    )
                    _ff_best_score = max(
                        (v for v in all_scores.values() if math.isfinite(v)), default=-math.inf
                    )
                    for _, _ff_row in ff_salsa_hits.iterrows():
                        _ff_smiles = _ff_row['product_smiles']
                        _ff_canon = get_canonical_smiles(_ff_smiles)
                        _ff_key = (_ff_canon, protein)

                        if _ff_key in boltz_cache:
                            _ff_score = boltz_cache[_ff_key]
                            bt.logging.debug(f"§FF cache hit: {_ff_score:.4f}")
                        else:
                            # Epoch guard before each GPU inference
                            try:
                                _curr_blk2 = await state['subtensor'].get_current_block()
                                _next_ep2 = ((_curr_blk2 // state['epoch_length']) + 1) * state['epoch_length']
                                if _next_ep2 - _curr_blk2 < 5:
                                    bt.logging.info("§FF: epoch ends in <5 blocks — stopping.")
                                    break
                            except Exception:
                                pass
                            _ff_uid = 0
                            _ff_vmbu = {_ff_uid: {"smiles": [_ff_smiles], "names": [_ff_row['product_name']]}}
                            _ff_sd = {_ff_uid: {}}
                            try:
                                await asyncio.to_thread(
                                    wrapper.score_molecules_target,
                                    _ff_vmbu, _ff_sd, subnet_config, '0x' + '0' * 64,
                                )
                                _ff_mol_scores = wrapper.per_molecule_metric.get(_ff_uid, {})
                                _ff_score = _ff_mol_scores.get(_ff_smiles, -math.inf)
                                boltz_cache[_ff_key] = _ff_score
                                _disk_cache_put(
                                    db_path, _ff_canon, protein, _ff_score,
                                    product_name=_ff_row.get('product_name'),
                                )
                                bt.logging.info(
                                    f"§FF scored: {_ff_row.get('product_name', '?')} "
                                    f"boltz={_ff_score:.4f}"
                                )
                            except Exception as _ff_e:
                                bt.logging.error(f"§FF Boltz inference error: {_ff_e}")
                                _ff_score = -math.inf

                        if math.isfinite(_ff_score) and _ff_score > _ff_best_score:
                            _ff_prev_score = _ff_best_score
                            _ff_best_score = _ff_score
                            _ff_pname = _ff_row.get('product_name', '')
                            if _ff_pname:
                                _orig = state['candidate_product'].split(',')
                                if _ff_pname in _orig:
                                    state['candidate_product'] = ','.join(
                                        [_ff_pname] + [n for n in _orig if n != _ff_pname]
                                    )
                                else:
                                    state['candidate_product'] = ','.join([_ff_pname] + _orig)
                                bt.logging.info(
                                    f"§FF: new best from Boltz-guided SALSA — "
                                    f"{_ff_pname} (boltz={_ff_score:.4f} > prev={_ff_prev_score:.4f})"
                                )
        except Exception as _ff_err:
            bt.logging.warning(f"§FF Boltz-guided SALSA failed (non-fatal): {_ff_err}")

    # Warm-start guard (§CC): after reordering by this epoch's Boltz scores,
    # check whether the best molecule in the persistent disk cache beats the
    # best new score.  This covers the scenario where the warm-start molecule
    # (scored in a prior session) was evicted from global_candidate_pool when
    # PSICHIC reset the pool at epoch start, so it never appeared in
    # `candidates` above — but its cached Boltz score is still valid and may
    # be higher than anything found this epoch.
    best_new = max((v for v in all_scores.values() if math.isfinite(v)), default=-math.inf)
    _ws_best = _disk_cache_get_best(db_path, protein)
    if (
        _ws_best is not None
        and state.get('candidate_product')
    ):
        _ws_score, _ws_smiles, _ws_pname = _ws_best
        # Only act if the cached molecule is not already in all_scores (i.e.
        # it was NOT evaluated during this run and thus not handled by
        # _reorder_submission above).
        _already_scored = any(
            get_canonical_smiles(s) == get_canonical_smiles(_ws_smiles)
            for s in all_scores
        )
        if (
            not _already_scored
            and math.isfinite(_ws_score)
            and _ws_score > best_new
            and _ws_pname
        ):
            _orig = state['candidate_product'].split(',')
            if _ws_pname in _orig:
                _reordered = [_ws_pname] + [n for n in _orig if n != _ws_pname]
            else:
                _reordered = [_ws_pname] + _orig
            state['candidate_product'] = ','.join(_reordered)
            bt.logging.info(
                f"[WarmGuard] Prior-session molecule retained at position 0: "
                f"{_ws_pname} (cached={_ws_score:.4f} > epoch_best={best_new:.4f})"
            )


# ----------------------------------------------------------------------------
# 6. MAIN MINING LOOP
# ----------------------------------------------------------------------------

async def run_miner(config: argparse.Namespace) -> None:
    """
    The main mining loop, orchestrating:
      - Bittensor objects initialization
      - Model initialization
      - Fetching new proteins each epoch
      - Running inference and submissions
      - Periodically syncing metagraph

    Args:
        config (argparse.Namespace): The miner configuration object.
    """

    # 1) Setup wallet, subtensor, metagraph, etc.
    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor_objects(config)

    # 2) Prepare shared state
    state: Dict[str, Any] = {
        # environment / config
        'config': config,
        'hugging_face_dataset_repo': 'Metanova/SAVI-2020',
        'psichic_result_column_name': 'predicted_binding_affinity',
        'chunk_size': 128,
        'submission_interval': 1200,

        # GitHub
        'github_path': load_github_path(),

        # Bittensor
        'wallet': wallet,
        'subtensor': subtensor,
        'metagraph': metagraph,
        'miner_uid': miner_uid,
        'epoch_length': epoch_length,

        # Models - one instance per protein
        'psichic_models': {},  # Dictionary mapping protein codes to their PSICHIC instances
        'bdt': QuicknetBittensorDrandTimelock(),

        # Inference state
        'candidate_product': None,
        'candidate_molecules': None,
        'global_candidate_pool': None,   # top-20 molecules across all chunks, by ligand efficiency
        'savi_stream_pool': None,        # all PSICHIC-scored molecules this epoch (capped at 5000)
        'salsa_run_this_epoch': False,   # prevent duplicate SALSA runs per epoch
        'ga_run_this_epoch': False,      # prevent duplicate GradientGA runs per epoch
        'best_score': float('-inf'),
        'boltz_prescored': False,
        'last_submitted_product': None,
        'last_submission_time': None,
        'shutdown_event': asyncio.Event(),

        # Boltz score cache: {(canonical_smiles, protein_code): float}
        # In-memory layer — persists across epochs within a session.
        'boltz_score_cache': {},
        # Path to persistent SQLite cache (survives process restarts).
        'boltz_cache_db': BOLTZ_CACHE_DB,

        # Challenges
        'current_challenge_targets': [],
        'last_challenge_targets': [],
        'current_challenge_antitargets': [],
        'last_challenge_antitargets': [],
    }

    # Ensure persistent Boltz cache DB exists
    _init_boltz_cache_db(state['boltz_cache_db'])
    _cleanup_boltz_cache(state['boltz_cache_db'], keep_protein=config.weekly_target)
    bt.logging.info(f"Boltz persistent cache initialised: {state['boltz_cache_db']}")

    # Warm epoch start (§AA): pre-populate candidate_product from best cached result.
    # On first run the cache is empty; on subsequent runs this gives an immediate
    # fallback submission before PSICHIC streaming finds new candidates.
    _apply_warm_start(state, state['boltz_cache_db'], config.weekly_target)

    # Ensure MSA file exists for the current weekly target (§S).
    # Boltz-2 predictions are significantly weaker without an MSA — this call
    # is a no-op when the file already exists and fetches it via ColabFold
    # API (~1–5 min) only when the target has rotated to a new protein.
    try:
        _target_seq = get_sequence_from_protein_code(config.weekly_target)
        if _target_seq:
            _msa_ok = ensure_msa(config.weekly_target, _target_seq)
            if not _msa_ok:
                bt.logging.warning(
                    f"[MSA] Could not obtain MSA for {config.weekly_target}. "
                    "Boltz-2 will run in single-sequence mode (weaker predictions)."
                )
        else:
            bt.logging.warning(f"[MSA] Could not retrieve sequence for {config.weekly_target} — skipping MSA fetch.")
    except Exception as _msa_exc:
        bt.logging.warning(f"[MSA] MSA check failed (non-fatal): {_msa_exc}")

    bt.logging.info("Entering main miner loop...")

    # 3) If we start mid-epoch, obtain most recent proteins from block hash
    current_block = await subtensor.get_current_block()
    last_boundary = (current_block // epoch_length) * epoch_length
    next_boundary = last_boundary + epoch_length

    # If we start too close to epoch end, wait for next epoch
    if next_boundary - current_block < 20:
        bt.logging.info(f"Too close to epoch end, waiting for next epoch to start...")
        block_to_check = next_boundary
        await asyncio.sleep(12*10)
    else:
        block_to_check = last_boundary

    block_hash = await subtensor.determine_block_hash(block_to_check)
    startup_proteins = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        weekly_target=config.weekly_target,
        num_antitargets=config.num_antitargets
    )

    if startup_proteins:
        state['current_challenge_targets'] = startup_proteins["targets"]
        state['last_challenge_targets'] = startup_proteins["targets"]
        state['current_challenge_antitargets'] = startup_proteins["antitargets"]
        state['last_challenge_antitargets'] = startup_proteins["antitargets"]
        bt.logging.info(f"Startup targets: {startup_proteins['targets']}, antitargets: {startup_proteins['antitargets']}")

        # Initialize models for all proteins
        try:
            for target_protein in startup_proteins["targets"]:
                target_sequence = get_sequence_from_protein_code(target_protein)
                model = PsichicWrapper()
                model.run_challenge_start(target_sequence)
                state['psichic_models'][target_protein] = model
                bt.logging.info(f"Initialized model for target: {target_protein}")

            for antitarget_protein in startup_proteins["antitargets"]:
                antitarget_sequence = get_sequence_from_protein_code(antitarget_protein)
                model = PsichicWrapper()
                model.run_challenge_start(antitarget_sequence)
                state['psichic_models'][antitarget_protein] = model
                bt.logging.info(f"Initialized model for antitarget: {antitarget_protein}")
        except Exception as e:
            try:
                os.system(
                    f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/TREAT1/model.pt')} "
                    f"https://huggingface.co/Metanova/TREAT-1/resolve/main/model.pt"
                )
                # Retry initialization after download
                for target_protein in state['current_challenge_targets']:
                    if target_protein not in state['psichic_models']:
                        target_sequence = get_sequence_from_protein_code(target_protein)
                        model = PsichicWrapper()
                        model.run_challenge_start(target_sequence)
                        state['psichic_models'][target_protein] = model
                        bt.logging.info(f"Initialized model for target: {target_protein}")

                for antitarget_protein in state['current_challenge_antitargets']:
                    if antitarget_protein not in state['psichic_models']:
                        antitarget_sequence = get_sequence_from_protein_code(antitarget_protein)
                        model = PsichicWrapper()
                        model.run_challenge_start(antitarget_sequence)
                        state['psichic_models'][antitarget_protein] = model
                        bt.logging.info(f"Initialized model for antitarget: {antitarget_protein}")
                bt.logging.info("Models re-downloaded and initialized successfully.")
            except Exception as e2:
                bt.logging.error(f"Error initializing models after re-download attempt: {e2}")

        # 4) Launch the inference loop
        try:
            state['inference_task'] = asyncio.create_task(run_psichic_model_loop(state))
            bt.logging.debug("Inference started on startup proteins.")
        except Exception as e:
            bt.logging.error(f"Error starting inference: {e}")

    # 5) Main epoch-based loop
    while True:
        try:
            current_block = await subtensor.get_current_block()

            # If we are at an epoch boundary, fetch new proteins
            if current_block % epoch_length == 0:
                bt.logging.info(f"Found epoch boundary at block {current_block}.")
                
                block_hash = await subtensor.determine_block_hash(current_block)
                
                new_proteins = get_challenge_params_from_blockhash(
                    block_hash=block_hash,
                    weekly_target=config.weekly_target,
                    num_antitargets=config.num_antitargets
                )
                if (new_proteins and 
                    (new_proteins["targets"] != state['last_challenge_targets'] or 
                     new_proteins["antitargets"] != state['last_challenge_antitargets'])):
                    state['current_challenge_targets'] = new_proteins["targets"]
                    state['last_challenge_targets'] = new_proteins["targets"]
                    state['current_challenge_antitargets'] = new_proteins["antitargets"]
                    state['last_challenge_antitargets'] = new_proteins["antitargets"]
                    bt.logging.info(f"New proteins - targets: {new_proteins['targets']}, antitargets: {new_proteins['antitargets']}")

                # Cancel old inference, reset relevant state
                if 'inference_task' in state and state['inference_task']:
                    if not state['inference_task'].done():
                        state['shutdown_event'].set()
                        bt.logging.debug("Shutdown event set for old inference task.")
                        await state['inference_task']

                # Reset best score and candidate
                state['candidate_product'] = None
                state['candidate_molecules'] = None
                state['global_candidate_pool'] = None
                state['savi_stream_pool'] = None
                state['salsa_run_this_epoch'] = False
                state['ga_run_this_epoch'] = False
                state['best_score'] = float('-inf')
                state['boltz_prescored'] = False
                state['last_submitted_product'] = None
                state['shutdown_event'] = asyncio.Event()

                # Warm-start (§AA): seed candidate_product from disk cache so
                # there is always a valid Boltz-validated submission available
                # even before PSICHIC streaming produces new candidates.
                _apply_warm_start(state, state['boltz_cache_db'], config.weekly_target)

                # Initialize models for new proteins
                try:
                    for target_protein in state['current_challenge_targets']:
                        if target_protein not in state['psichic_models']:
                            target_sequence = get_sequence_from_protein_code(target_protein)
                            model = PsichicWrapper()
                            model.run_challenge_start(target_sequence)
                            state['psichic_models'][target_protein] = model
                            bt.logging.info(f"Initialized model for target: {target_protein}")

                    for antitarget_protein in state['current_challenge_antitargets']:
                        if antitarget_protein not in state['psichic_models']:
                            antitarget_sequence = get_sequence_from_protein_code(antitarget_protein)
                            model = PsichicWrapper()
                            model.run_challenge_start(antitarget_sequence)
                            state['psichic_models'][antitarget_protein] = model
                            bt.logging.info(f"Initialized model for antitarget: {antitarget_protein}")
                except Exception as e:
                    try:
                        os.system(
                            f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/TREAT1/model.pt')} "
                            f"https://huggingface.co/Metanova/TREAT-1/resolve/main/model.pt"
                        )
                        # Retry initialization after download
                        for target_protein in state['current_challenge_targets']:
                            if target_protein not in state['psichic_models']:
                                target_sequence = get_sequence_from_protein_code(target_protein)
                                model = PsichicWrapper()
                                model.run_challenge_start(target_sequence)
                                state['psichic_models'][target_protein] = model
                                bt.logging.info(f"Initialized model for target: {target_protein}")

                        for antitarget_protein in state['current_challenge_antitargets']:
                            if antitarget_protein not in state['psichic_models']:
                                antitarget_sequence = get_sequence_from_protein_code(antitarget_protein)
                                model = PsichicWrapper()
                                model.run_challenge_start(antitarget_sequence)
                                state['psichic_models'][antitarget_protein] = model
                                bt.logging.info(f"Initialized model for antitarget: {antitarget_protein}")
                        bt.logging.info("Models re-downloaded and initialized successfully.")
                    except Exception as e2:
                        bt.logging.error(f"Error initializing models after re-download attempt: {e2}")

                # Start new inference
                try:
                    state['inference_task'] = asyncio.create_task(run_psichic_model_loop(state))
                    bt.logging.debug("New inference task started.")
                except Exception as e:
                    bt.logging.error(f"Error starting new inference: {e}")

            # Periodically update our knowledge of the network
            if current_block % 60 == 0:
                await metagraph.sync()
                log = (
                    f"Block: {metagraph.block.item()} | "
                    f"Number of nodes: {metagraph.n} | "
                    f"Current epoch: {metagraph.block.item() // epoch_length}"
                )
                bt.logging.info(log)

            await asyncio.sleep(1)

        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting miner.")
            break


# ----------------------------------------------------------------------------
# 7. ENTRY POINT
# ----------------------------------------------------------------------------

async def main() -> None:
    """
    Main entry point for asynchronous execution of the miner logic.
    """
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
