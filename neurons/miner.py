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
from utils.chembl import get_chembl_seeds
from utils.surrogate import fit_surrogate, rank_pool_by_surrogate
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


def _disk_cache_get_candidates(db_path: str, protein: str, limit: int = 20) -> list:
    """
    Return up to *limit* cached entries for *protein*, sorted by score desc.

    Used as a synthetic candidate pool in run_boltz_prescoring when PSICHIC
    has not yet produced candidates (e.g., early-epoch miner restart).  All
    returned entries will be instant in-memory or disk cache hits — zero GPU
    time — but the §CC warm-start guard and _reorder_submission will still run
    correctly, confirming the best historical molecule is at position 0.

    Only returns entries with a valid product_name (required for submission).
    """
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT score, smiles, product_name FROM boltz_cache "
                "WHERE protein=? AND product_name IS NOT NULL "
                "ORDER BY score DESC LIMIT ?",
                (protein, limit),
            ).fetchall()
        return [
            {'product_name': pn, 'product_smiles': sm, 'combined_score': sc}
            for sc, sm, pn in rows
        ]
    except Exception:
        return []


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

_SAVI_FILE_CACHE: dict = {}   # repo_url -> sorted list of CSV filenames (populated once)
_SAVI_SEEN_FILES: dict = {}   # repo_url -> set of files used this epoch (reset externally)


def stream_random_chunk_from_dataset(dataset_repo: str, chunk_size: int, rxn_bias: Optional[str] = None) -> Any:
    """
    Streams a random chunk from the specified Hugging Face dataset repo.

    File list is cached after the first call to avoid repeated HuggingFace API
    requests.  Within each epoch, files are sampled without replacement so that
    successive outer-loop cycles explore distinct regions of SAVI-2020 chemical
    space.  When all files have been seen the seen-set resets and a new cycle begins.

    Args:
        dataset_repo (str): Hugging Face dataset repository path (user/repo).
        chunk_size (int): Size of each chunk to stream.

    Returns:
        Any: A batched (chunked) dataset iterator.
    """
    if dataset_repo not in _SAVI_FILE_CACHE:
        all_files = list_repo_files(dataset_repo, repo_type='dataset')
        _SAVI_FILE_CACHE[dataset_repo] = sorted(
            f for f in all_files if f.endswith('.csv')
        )

    files = _SAVI_FILE_CACHE[dataset_repo]

    seen = _SAVI_SEEN_FILES.setdefault(dataset_repo, set())
    available = [f for f in files if f not in seen]
    if not available:
        seen.clear()
        available = files
    # §YY: 2× weight for files whose path contains the winning reaction class.
    # Biases the sampling toward the chemical family that produced the best
    # Boltz-validated molecule in prior epochs.  Falls back to uniform when
    # rxn_bias is None (first epoch) or no file matches the class string.
    if rxn_bias:
        _yy_weights = [2.0 if rxn_bias in f else 1.0 for f in available]
        random_file = random.choices(available, weights=_yy_weights, k=1)[0]
    else:
        random_file = random.choice(available)
    seen.add(random_file)

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
                chunk_size=state['chunk_size'],
                rxn_bias=state.get('best_boltz_rxn_class'),
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
                # SAVI stream pool: accumulates the top-10000 PSICHIC-scored
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
                        state['savi_stream_pool'] = _pool_combined.head(10000).reset_index(drop=True)

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
                    #   - Requires >=500 molecules in the stream pool (enough for NN).
                    #   - Fires only when > boltz_trigger * 1.5 blocks remain, so
                    #     SALSA hits are in global_candidate_pool before Boltz starts.
                    #   - One-shot per epoch (salsa_run_this_epoch flag).
                    # ---------------------------------------------------------------
                    salsa_pool = state.get('savi_stream_pool')
                    salsa_pool_size = 0 if salsa_pool is None else len(salsa_pool)
                    boltz_trigger = state.get('boltz_trigger_blocks', 100)
                    # Ensure SALSA fires at least 30 blocks before Boltz.
                    # The 1.5x formula breaks down on fast hardware (A100/H100)
                    # where the adaptive trigger drops boltz_trigger_blocks to ~39:
                    # 1.5 x 39 = 58 < 39 + 30 = 69.  Without this floor, SALSA
                    # could fire within the Boltz window on the same chunk iteration.
                    salsa_threshold = max(int(boltz_trigger * 1.5), boltz_trigger + 30)
                    if (
                        not state.get('salsa_run_this_epoch', False)
                        and salsa_pool_size >= 500
                        and blocks_until_epoch > salsa_threshold
                        and state.get('global_candidate_pool') is not None
                        and not state['global_candidate_pool'].empty
                    ):
                        state['salsa_run_this_epoch'] = True

                        # §ZZ: Re-rank global_candidate_pool by mini-surrogate before
                        # selecting SALSA seeds.  When ≥ 40 Boltz scores are cached for
                        # this protein (typically epoch 3+), the Ridge surrogate on 20
                        # RDKit descriptors re-orders candidates with a Boltz-calibrated
                        # signal so SALSA explores the most promising chemical region
                        # first.  Falls back silently to PSICHIC ranking on cache miss.
                        try:
                            _zz_seed_model = fit_surrogate(
                                state.get('boltz_cache_db', BOLTZ_CACHE_DB),
                                state['config'].weekly_target,
                            )
                            if _zz_seed_model is not None and not state['global_candidate_pool'].empty:
                                state['global_candidate_pool'] = rank_pool_by_surrogate(
                                    state['global_candidate_pool'], _zz_seed_model
                                )
                                bt.logging.info("[ZZ] SALSA seeds re-ranked by surrogate.")
                        except Exception as _zz_s_err:
                            bt.logging.debug(f"[ZZ] SALSA seed re-rank skipped: {_zz_s_err}")

                        # Multi-seed SALSA: run from up to top-3 candidates so we
                        # explore three distinct chemical neighbourhoods in one pass.
                        # Runtime: ~3 x 180 ms = ~540 ms CPU -- negligible vs Boltz.
                        _n_seeds = min(3, len(state['global_candidate_pool']))
                        _seeds = state['global_candidate_pool'].head(_n_seeds)['product_smiles'].tolist()
                        # §SS: extend with up to 3 ChEMBL known actives as additional seeds.
                        # These are validated binders fetched at startup; each is used as a
                        # SALSA starting point and the NN lookup maps perturbations back to
                        # valid SAVI-2020 molecules.  Falls back silently if none are available.
                        _chembl_ok = [
                            s for s in state.get('chembl_seeds', [])
                            if Chem.MolFromSmiles(s) is not None and s not in _seeds
                        ][:3]
                        if _chembl_ok:
                            _seeds = _seeds + _chembl_ok
                            bt.logging.info(
                                f"[SS] Adding {len(_chembl_ok)} ChEMBL active(s) as extra SALSA seed(s)."
                            )

                        # §UU: extend with up to 3 prior-epoch Boltz-validated seeds from disk cache.
                        # In epoch 2+ on the same weekly target, the disk cache holds molecules that
                        # have already been scored by the actual Boltz-2 oracle — far better seeds
                        # than PSICHIC-ranked candidates alone.  Silently ignored on the first epoch
                        # (empty cache) and when `binding_pocket` changes (different protein key).
                        _uu_db_path = state.get('boltz_cache_db', BOLTZ_CACHE_DB)
                        _uu_protein = state['config'].weekly_target
                        _uu_cached = _disk_cache_get_candidates(_uu_db_path, _uu_protein, limit=5)
                        _uu_seeds = []
                        for _uu_row in _uu_cached:
                            _uu_sm = _uu_row.get('product_smiles', '')
                            if (
                                _uu_sm
                                and _uu_sm not in _seeds
                                and Chem.MolFromSmiles(_uu_sm) is not None
                                and is_boltz_safe_smiles(_uu_sm)[0]
                            ):
                                _uu_seeds.append(_uu_sm)
                            if len(_uu_seeds) >= 3:
                                break
                        if _uu_seeds:
                            _seeds = _seeds + _uu_seeds
                            bt.logging.info(
                                f"[UU] Adding {len(_uu_seeds)} prior-epoch Boltz-validated seed(s) to SALSA."
                            )

                        _seed_parts = [f"{_n_seeds} PSICHIC"]
                        if _chembl_ok:
                            _seed_parts.append(f"{len(_chembl_ok)} ChEMBL")
                        if _uu_seeds:
                            _seed_parts.append(f"{len(_uu_seeds)} Boltz-cache")
                        _seed_desc = " + ".join(_seed_parts)
                        bt.logging.info(
                            f"SALSA: triggering with {salsa_pool_size}-molecule pool, "
                            f"{blocks_until_epoch} blocks remaining (threshold={salsa_threshold}), "
                            f"{len(_seeds)} seed(s) ({_seed_desc})..."
                        )
                        try:
                            _all_salsa = []
                            for _seed_smiles in _seeds:
                                _hits = await asyncio.to_thread(
                                    run_salsa_search,
                                    _seed_smiles,
                                    salsa_pool,
                                    3,   # rounds
                                    200, # n_perturb — allows ring walk + terminal removal to contribute
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
                    # Runtime: ~1-3 s CPU for 5 generations -- negligible vs Boltz.
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
                                # §BBB: store best GA hit for post-GA SALSA pass.
                                state['best_ga_smiles'] = ga_hits.iloc[0]['product_smiles']
                            else:
                                bt.logging.info("GradientGA: no hits found.")
                        except Exception as _ga_err:
                            bt.logging.error(f"GradientGA error: {_ga_err}")

                    # §BBB: Post-GA SALSA — one SALSA pass from the best GA hit, exploring
                    # its chemical neighbourhood before Boltz fires.  GA often discovers
                    # molecules in a different region of chemical space than PSICHIC/ChEMBL
                    # seeds; running SALSA from the GA winner maps its surroundings onto
                    # SAVI-2020 products that Boltz can then score alongside the PSICHIC
                    # candidates.  Fires once per epoch, immediately after the GA block.
                    if (
                        not state.get('bbb_run_this_epoch', False)
                        and state.get('best_ga_smiles')
                        and blocks_until_epoch > boltz_trigger_ga + 5
                        and state.get('savi_stream_pool') is not None
                        and not state['savi_stream_pool'].empty
                    ):
                        state['bbb_run_this_epoch'] = True
                        _bbb_pool = state['savi_stream_pool']
                        bt.logging.info(
                            f"§BBB: post-GA SALSA from GA winner "
                            f"({blocks_until_epoch} blocks remaining)..."
                        )
                        try:
                            _bbb_hits = await asyncio.to_thread(
                                run_salsa_search,
                                state['best_ga_smiles'],
                                _bbb_pool,
                                2,   # rounds — neighbourhood exploration
                                200, # n_perturb — full operator coverage
                                3,   # top_k
                            )
                            if not _bbb_hits.empty:
                                bt.logging.info(
                                    f"§BBB: {len(_bbb_hits)} post-GA SALSA hits — "
                                    f"merging into global_candidate_pool."
                                )
                                _bbb_combined = pd.concat(
                                    [state['global_candidate_pool'], _bbb_hits],
                                    ignore_index=True,
                                )
                                _bbb_combined.drop_duplicates(
                                    subset=['product_name'], inplace=True
                                )
                                _bbb_combined.sort_values(
                                    'combined_score', ascending=False, inplace=True
                                )
                                state['global_candidate_pool'] = (
                                    _bbb_combined.head(20).reset_index(drop=True)
                                )
                                bt.logging.info(
                                    f"§BBB: global_candidate_pool now has "
                                    f"{len(state['global_candidate_pool'])} entries."
                                )
                            else:
                                bt.logging.info("§BBB: no post-GA SALSA hits.")
                        except Exception as _bbb_err:
                            bt.logging.error(f"§BBB post-GA SALSA error: {_bbb_err}")

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
    allowed) until *max_k* slots are filled -- so if the pool is small or
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
            # Unparseable -- admit unconditionally (Boltz-safe filter upstream
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


def _reorder_for_diversity(state: Dict[str, Any]) -> None:
    """
    Reorder positions 1+ of state['candidate_product'] to maximise MACCS fingerprint
    diversity relative to the best Boltz molecule at position 0.

    Active when num_molecules_boltz > 1.  The validator computes compute_maccs_entropy()
    over the first N molecules and applies a diversity bonus to boltz_score.  Placing the
    most MACCS-dissimilar molecules in slots 1..N-1 maximises that bonus without
    disturbing the best Boltz binder at position 0.

    Zero-effect when num_molecules_boltz == 1 (current validator default).
    """
    from rdkit.Chem import MACCSkeys, DataStructs as _DS

    current = state.get('candidate_product') or ''
    names = [n for n in current.split(',') if n]
    if len(names) <= 1:
        return

    # Build name->SMILES lookup from all available molecule pools (ordered by quality)
    name_to_smiles: Dict[str, str] = {}
    for frame in (
        state.get('global_candidate_pool'),
        state.get('candidate_molecules'),
        state.get('savi_stream_pool'),
    ):
        if frame is None or frame.empty:
            continue
        if 'product_name' not in frame.columns or 'product_smiles' not in frame.columns:
            continue
        for _, r in frame.iterrows():
            pn, ps = str(r.get('product_name', '')), str(r.get('product_smiles', ''))
            if pn and ps and pn not in name_to_smiles:
                name_to_smiles[pn] = ps

    best_smiles = name_to_smiles.get(names[0], '')
    if not best_smiles:
        return
    best_mol = Chem.MolFromSmiles(best_smiles)
    if best_mol is None:
        return
    best_fp = MACCSkeys.GenMACCSKeys(best_mol)

    scored_rest: List[Tuple[float, str]] = []
    unknown: List[str] = []
    for name in names[1:]:
        smiles = name_to_smiles.get(name, '')
        if not smiles:
            unknown.append(name)
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            unknown.append(name)
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        sim = _DS.TanimotoSimilarity(best_fp, fp)
        scored_rest.append((1.0 - sim, name))  # higher distance = more diverse

    scored_rest.sort(reverse=True)
    state['candidate_product'] = ','.join(
        [names[0]] + [n for _, n in scored_rest] + unknown
    )
    bt.logging.debug(
        f"§C diversity reorder: {len(scored_rest)} molecules sorted by MACCS distance from anchor"
    )


async def run_boltz_prescoring(state: Dict[str, Any], max_candidates: int = 5) -> None:
    """
    Runs Boltz-2 affinity predictions on the top PSICHIC candidates and reorders
    state['candidate_product'] so the highest-scoring Boltz-2 molecule comes first.

    Implements an ANYTIME / incremental scoring strategy: candidates are scored
    one-by-one in PSICHIC-rank order and state['candidate_product'] is reordered
    immediately after each molecule is scored.  This means even if the epoch ends
    mid-run, the submission already reflects the best Boltz score seen so far --
    not just the raw PSICHIC ranking.

    Results are cached in state['boltz_score_cache'] keyed by
    (canonical_smiles, protein_code) so molecules already scored in this session
    skip inference entirely -- saving 45-150 s of GPU time per cache hit.

    The validator uses sample_selection="first" with num_molecules_boltz=1, so the
    molecule placed first in the submission is the one that determines our Boltz score.
    Boltz scoring formula: (affinity_probability_binary - affinity_pred_value) / heavy_atom_count

    Args:
        state: Shared miner state dict.
        max_candidates: Maximum number of PSICHIC top-molecules to score with Boltz-2.
    """
    protein = state['config'].weekly_target
    boltz_cache: Dict[Tuple[str, str], float] = state.setdefault('boltz_score_cache', {})
    db_path: str = state.get('boltz_cache_db', BOLTZ_CACHE_DB)

    # Prefer global candidate pool (spans entire epoch, sorted by ligand efficiency)
    # so Boltz always evaluates the best molecules seen so far, not just those from
    # the most recent best-batch.  Fall back to current-batch candidates if needed.
    candidates = state.get('global_candidate_pool')
    if candidates is None or candidates.empty:
        candidates = state.get('candidate_molecules')
    if candidates is None or candidates.empty:
        # §JJ — Cache-fallback synthetic pool: when PSICHIC has not yet produced
        # candidates (e.g., early-epoch miner restart), use disk-cached entries
        # as a synthetic pool.  All entries are instant cache hits — zero GPU time
        # — but the §CC warm-start guard and _reorder_submission still execute,
        # ensuring the best historical molecule stays at position 0 even when
        # both PSICHIC pools are empty at the time Boltz first triggers.
        _cached_rows = _disk_cache_get_candidates(db_path, protein, limit=max_candidates)
        if _cached_rows:
            candidates = pd.DataFrame(_cached_rows)
            candidates['heavy_atoms'] = candidates['product_smiles'].apply(
                lambda s: get_heavy_atom_count(s) or 1
            )
            bt.logging.info(
                f"Boltz-2 §JJ: PSICHIC pools empty — using {len(candidates)} "
                f"disk-cached entries as synthetic pool (all instant cache hits)."
            )
        else:
            bt.logging.warning("Boltz-2 pre-scoring: no candidate molecules available.")
            return

    # Pull a wider slice (3x budget) so the scaffold-diversity filter has
    # candidates to choose from even when SALSA has converged to one region.
    # global_candidate_pool is capped at 20 rows, so this is at most 20.
    candidates = candidates.head(max_candidates * 3).copy()
    safe_mask = candidates['product_smiles'].apply(lambda s: is_boltz_safe_smiles(s)[0])
    candidates = candidates[safe_mask].reset_index(drop=True)

    # §ZZ: Re-rank Boltz candidates by mini-surrogate when ≥ 40 scores are cached.
    # Ridge regression on 20 RDKit descriptors gives Boltz-calibrated ordering so
    # the scaffold diversity filter preferentially selects molecules whose descriptor
    # profile correlates with high Boltz scores for this specific protein.
    # Fitting takes ~30 ms (40 points × 20 features) — negligible vs inference.
    # Falls back silently to PSICHIC ordering when the cache has < 40 entries.
    if not candidates.empty:
        try:
            _zz_model = fit_surrogate(db_path, protein)
            if _zz_model is not None:
                candidates = rank_pool_by_surrogate(candidates, _zz_model)
                bt.logging.info(
                    f"[ZZ] Pre-Boltz candidates re-ranked by surrogate "
                    f"({len(candidates)} entries, target={protein})."
                )
        except Exception as _zz_err:
            bt.logging.debug(f"[ZZ] Candidate surrogate re-rank skipped: {_zz_err}")

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
    # §RR: confidence-adjusted ordering scores (same as all_scores except for very low-
    # confidence GPU inference results; never written to cache or used by §CC/§MM).
    _rr_eff_scores: Dict[str, float] = {}

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
            f"  -> submission updated after {len(scores)}/{len(candidates)} scored: "
            f"best={best_name} (boltz_score={best_score:.4f})"
        )

    bt.logging.info(
        f"Boltz-2 anytime pre-scoring: {len(candidates)} candidates for target {protein}..."
    )

    for i, row in candidates.iterrows():
        smiles = row['product_smiles']
        canon = get_canonical_smiles(smiles)
        key = (canon, protein)

        # --- Cache lookup (in-memory -> disk -> GPU inference) ---
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
                            f"[{i+1}/{len(candidates)}] epoch ends in <5 blocks -- "
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

                    # §LL: Log component breakdown to aid diagnostics and tuning.
                    # affinity_probability_binary (apb) and affinity_pred_value (apv)
                    # drive the boltz_score formula; confidence_score and ligand_iptm
                    # reflect structural reliability (low values = uncertain binding mode).
                    _comps = wrapper.per_molecule_components.get(uid, {}).get(smiles, {})
                    def _fv(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"
                    bt.logging.info(
                        f"  [Boltz components] score={score:.4f} | "
                        f"apb={_fv(_comps.get('affinity_probability_binary'))} "
                        f"apv={_fv(_comps.get('affinity_pred_value'))} "
                        f"conf={_fv(_comps.get('confidence_score'))} "
                        f"ligand_iptm={_fv(_comps.get('ligand_iptm'))}"
                    )

                    # §RR: Confidence-weighted ordering for GPU inference results only.
                    # Molecules where BOTH ligand_iptm < 0.25 AND confidence_score < 0.30
                    # have genuinely uncertain binding modes; their validator re-run score
                    # is more likely to differ substantially from the miner measurement.
                    # Apply a modest ordering penalty so high-confidence molecules are
                    # preferred when scores are close.  The true score is always cached
                    # unmodified (§CC and §MM must compare validator-aligned values).
                    _rr_score = score  # default: no penalty
                    if math.isfinite(score):
                        _li = _comps.get('ligand_iptm')
                        _cs = _comps.get('confidence_score')
                        if isinstance(_li, (int, float)) and isinstance(_cs, (int, float)):
                            if _li < 0.25 and _cs < 0.30:
                                # Scale factor: 0.50 when (li,cs)=(0,0) → ~0.96 at threshold.
                                _rr_factor = 0.50 + (_li + _cs) / 1.10
                                _rr_score = score * _rr_factor
                                bt.logging.info(
                                    f"  [§RR] Low-conf penalty "
                                    f"(ligand_iptm={_li:.3f}, conf={_cs:.3f}): "
                                    f"ordering {score:.4f} -> {_rr_score:.4f}"
                                )
                    _rr_eff_scores[smiles] = _rr_score

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
                            f"  adaptive timing: {elapsed:.1f}s/mol -> "
                            f"trigger={state['boltz_trigger_blocks']} blocks"
                        )
                except Exception as e:
                    bt.logging.error(f"  Boltz-2 inference failed: {e}")
                    traceback.print_exc()
                    score = -math.inf

        all_scores[smiles] = score
        # For cache hits _rr_eff_scores entry was not set in the GPU-inference block above.
        # Default to the true score (no confidence data available for cached results).
        if smiles not in _rr_eff_scores:
            _rr_eff_scores[smiles] = score

        # Reorder submission immediately -- anytime guarantee: if epoch ends
        # after this molecule, the best Boltz score seen so far is at position 0.
        # §RR: use confidence-adjusted scores for ordering; all_scores keeps true values
        # for §CC warm-start guard and §MM comparisons.
        _reorder_submission(_rr_eff_scores)

    # Final summary
    valid_scores = {s: v for s, v in all_scores.items() if v != -math.inf}
    bt.logging.info(
        f"Boltz-2 anytime pre-scoring complete: "
        f"{len(valid_scores)}/{len(candidates)} molecules scored."
    )

    # §FF: Boltz-guided SALSA -- second SALSA pass seeded from the best Boltz molecule.
    # The main loop above uses the PSICHIC-ranked candidate pool as seeds; PSICHIC and
    # Boltz-2 correlate imperfectly, so the validated Boltz winner may occupy a different
    # region of chemical space.  By running SALSA from the actual best-Boltz SMILES we
    # explore its chemical neighbourhood and may find SAVI-2020 molecules that score
    # even better -- without any additional PSICHIC overhead.
    # Only fires when the epoch has >=2 mol-lengths + 2 min of runway remaining.
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
                    f"§FF Boltz-guided SALSA: {_remaining_s_ff:.0f}s remaining -- "
                    f"seeding 2-round SALSA from best Boltz molecule..."
                )
                ff_salsa_hits = await asyncio.to_thread(
                    run_salsa_search,
                    _ff_best_smiles,
                    _savi_pool_ff,
                    2,   # rounds (fewer -- time is limited)
                    200, # n_perturb — full operator coverage (ring walk + terminal removal)
                    3,   # top_k
                )
                if not ff_salsa_hits.empty:
                    bt.logging.info(
                        f"§FF: {len(ff_salsa_hits)} Boltz-guided SALSA hits -- §NN two-phase screening..."
                    )
                    _ff_best_score = max(
                        (v for v in all_scores.values() if math.isfinite(v)), default=-math.inf
                    )

                    # §NN Phase 1: fast-screen all hits (cache hits reuse full score; misses use
                    # fast Boltz with reduced sampling steps so we can screen more hits cheaply).
                    # Fast scores are NOT stored in the persistent cache — only full scores are.
                    _ff_screen: Dict[str, float] = {}  # smiles -> score (cached full or fast)
                    _ff_rows: Dict[str, Any] = {}      # smiles -> row for later lookup
                    for _, _ff_row in ff_salsa_hits.iterrows():
                        _ff_smiles = _ff_row['product_smiles']
                        _ff_canon = get_canonical_smiles(_ff_smiles)
                        _ff_key = (_ff_canon, protein)
                        _ff_rows[_ff_smiles] = _ff_row
                        if _ff_key in boltz_cache:
                            _ff_screen[_ff_smiles] = boltz_cache[_ff_key]
                            bt.logging.debug(f"§FF §NN cache hit: {boltz_cache[_ff_key]:.4f}")
                        else:
                            try:
                                _curr_blk2 = await state['subtensor'].get_current_block()
                                _next_ep2 = ((_curr_blk2 // state['epoch_length']) + 1) * state['epoch_length']
                                if _next_ep2 - _curr_blk2 < 5:
                                    bt.logging.info("§FF §NN: epoch ends in <5 blocks -- stopping.")
                                    break
                            except Exception:
                                pass
                            _ff_uid_s = 0
                            _ff_vmbu_s = {_ff_uid_s: {"smiles": [_ff_smiles], "names": [_ff_row['product_name']]}}
                            _ff_sd_s = {_ff_uid_s: {}}
                            try:
                                await asyncio.to_thread(
                                    wrapper.score_molecules_target,
                                    _ff_vmbu_s, _ff_sd_s, subnet_config, '0x' + '0' * 64, True,
                                )
                                _ff_screen[_ff_smiles] = wrapper.per_molecule_metric.get(_ff_uid_s, {}).get(_ff_smiles, -math.inf)
                            except Exception as _ff_es:
                                bt.logging.error(f"§FF §NN fast-screen error: {_ff_es}")
                                _ff_screen[_ff_smiles] = -math.inf

                    # §NN Phase 2: full-score only the best fast-screened candidate.
                    _ff_winner = max(
                        (_s for _s, _v in _ff_screen.items() if math.isfinite(_v)),
                        key=lambda _s: _ff_screen[_s],
                        default=None,
                    )
                    if _ff_winner is not None:
                        _ff_w_row = _ff_rows[_ff_winner]
                        _ff_w_canon = get_canonical_smiles(_ff_winner)
                        _ff_w_key = (_ff_w_canon, protein)
                        if _ff_w_key in boltz_cache:
                            _ff_score = boltz_cache[_ff_w_key]
                        else:
                            try:
                                _curr_blk3 = await state['subtensor'].get_current_block()
                                _next_ep3 = ((_curr_blk3 // state['epoch_length']) + 1) * state['epoch_length']
                                if _next_ep3 - _curr_blk3 >= 5:
                                    _ff_uid_f = 0
                                    _ff_vmbu_f = {_ff_uid_f: {"smiles": [_ff_winner], "names": [_ff_w_row['product_name']]}}
                                    _ff_sd_f = {_ff_uid_f: {}}
                                    await asyncio.to_thread(
                                        wrapper.score_molecules_target,
                                        _ff_vmbu_f, _ff_sd_f, subnet_config, '0x' + '0' * 64,
                                    )
                                    _ff_score = wrapper.per_molecule_metric.get(_ff_uid_f, {}).get(_ff_winner, -math.inf)
                                    boltz_cache[_ff_w_key] = _ff_score
                                    _disk_cache_put(db_path, _ff_w_canon, protein, _ff_score,
                                                    product_name=_ff_w_row.get('product_name'))
                                    bt.logging.info(
                                        f"§FF §NN full-scored winner: {_ff_w_row.get('product_name', '?')} "
                                        f"boltz={_ff_score:.4f} (screened {len(_ff_screen)} hits)"
                                    )
                                else:
                                    _ff_score = -math.inf
                            except Exception as _ff_e:
                                bt.logging.error(f"§FF §NN full-score error: {_ff_e}")
                                _ff_score = -math.inf

                        if math.isfinite(_ff_score) and _ff_score > _ff_best_score:
                            _ff_prev_score = _ff_best_score
                            _ff_best_score = _ff_score
                            _ff_pname = _ff_w_row.get('product_name', '')
                            if _ff_pname:
                                _orig = state['candidate_product'].split(',')
                                if _ff_pname in _orig:
                                    state['candidate_product'] = ','.join(
                                        [_ff_pname] + [n for n in _orig if n != _ff_pname]
                                    )
                                else:
                                    state['candidate_product'] = ','.join([_ff_pname] + _orig)
                                bt.logging.info(
                                    f"§FF §NN: new best from Boltz-guided SALSA -- "
                                    f"{_ff_pname} (boltz={_ff_score:.4f} > prev={_ff_prev_score:.4f})"
                                )
        except Exception as _ff_err:
            bt.logging.warning(f"§FF Boltz-guided SALSA failed (non-fatal): {_ff_err}")

    # §MM: Multi-round iterative Boltz-SALSA hill-climbing.
    # After the initial Boltz pass (all_scores) and §FF (one Boltz-SALSA round),
    # continue hill-climbing from the current best Boltz molecule while time permits.
    # Each round: SALSA from current best → score top-3 SALSA hits with Boltz →
    # if improved, advance seed and repeat; otherwise stop.
    #
    # On A100 hardware (45 s/mol, 1200 s trigger window):
    #   Initial pass: ~270 s (6 mols)
    #   §FF:          ~135 s (3 mols)
    #   §MM budget:   ~795 s → up to 5 more rounds × 3 mols = 15 additional Boltz calls
    # On RTX 3090 (150 s/mol): typically no budget remains after §FF — §MM exits immediately.
    #
    # Build _mm_all_scored: union of initial-pass scores (all_scores) and §FF hits
    # already stored in boltz_cache, so the seed and baseline reflect the true epoch best.
    _mm_all_scored: Dict[str, float] = dict(all_scores)
    for _mm_ck, _mm_cv in boltz_cache.items():
        if isinstance(_mm_ck, tuple) and len(_mm_ck) == 2 and _mm_ck[1] == protein:
            _mm_sm = _mm_ck[0]
            if _mm_sm not in _mm_all_scored and math.isfinite(_mm_cv):
                _mm_all_scored[_mm_sm] = _mm_cv

    _mm_best_score: float = max(
        (_v for _v in _mm_all_scored.values() if math.isfinite(_v)), default=-math.inf
    )
    _mm_seed_smiles: Optional[str] = (
        max(
            (_s for _s, _v in _mm_all_scored.items() if math.isfinite(_v)),
            key=lambda s: _mm_all_scored[s],
            default=None,
        )
        if _mm_all_scored else None
    )
    _mm_savi_pool = state.get('savi_stream_pool')
    # 10 rounds: on A100 the time guard fires after ~7 rounds anyway; on RTX 3090
    # it fires after 0-1 rounds.  The higher cap lets fast hardware fully utilise
    # the epoch budget without artificially stopping early.
    _mm_max_rounds = 10
    _mm_stop = False

    if (
        _mm_seed_smiles is not None
        and math.isfinite(_mm_best_score)
        and _mm_savi_pool is not None
        and not _mm_savi_pool.empty
    ):
        bt.logging.info(
            f"§MM: starting multi-round Boltz-SALSA "
            f"(seed_score={_mm_best_score:.4f}, max_rounds={_mm_max_rounds})"
        )
        _mm_rounds_run = 0
        _mm_tried_seeds: set = set()  # prevent cycling when basin-hopping
        for _mm_round_idx in range(_mm_max_rounds):
            if _mm_stop:
                break

            # Mark current seed as tried so §QQ basin-hopping skips it next time.
            _mm_tried_seeds.add(_mm_seed_smiles)

            # Time check before each round — skip if < 2 mol-times + 2 min remain
            try:
                _mm_curr_blk = await state['subtensor'].get_current_block()
                _mm_next_ep = ((_mm_curr_blk // state['epoch_length']) + 1) * state['epoch_length']
                _mm_remaining_s = (_mm_next_ep - _mm_curr_blk) * 12
            except Exception:
                break

            _mm_t_mol = state.get('boltz_time_per_mol', 150.0)
            if _mm_remaining_s < _mm_t_mol * 2 + 120:
                bt.logging.info(
                    f"§MM round {_mm_round_idx + 1}: "
                    f"{_mm_remaining_s:.0f}s remaining < {_mm_t_mol * 2 + 120:.0f}s needed — stopping."
                )
                break

            # SALSA from current best Boltz seed
            try:
                _mm_salsa_hits = await asyncio.to_thread(
                    run_salsa_search,
                    _mm_seed_smiles,
                    _mm_savi_pool,
                    2,   # rounds — neighbourhood exploration
                    200, # n_perturb — full operator coverage (ring walk + terminal removal)
                    3,   # top_k — cap Boltz calls per round
                )
            except Exception as _mm_salsa_err:
                bt.logging.warning(f"§MM SALSA error: {_mm_salsa_err}")
                break

            if _mm_salsa_hits.empty:
                bt.logging.info(f"§MM round {_mm_round_idx + 1}: no SALSA hits — stopping.")
                break

            _mm_improved = False
            _mm_round_best_score = _mm_best_score
            _mm_round_best_smiles = _mm_seed_smiles

            # §NN Phase 1: fast-screen each round's SALSA hits.
            # Cache hits reuse the stored full score; misses run fast Boltz (no cache store).
            _mm_screen: Dict[str, float] = {}  # smiles -> score
            _mm_row_map: Dict[str, Any] = {}   # smiles -> row
            for _, _mm_row in _mm_salsa_hits.iterrows():
                if _mm_stop:
                    break
                _mm_smiles = _mm_row['product_smiles']
                _mm_canon = get_canonical_smiles(_mm_smiles)
                _mm_key = (_mm_canon, protein)
                _mm_row_map[_mm_smiles] = _mm_row
                if _mm_key in boltz_cache:
                    _mm_screen[_mm_smiles] = boltz_cache[_mm_key]
                    bt.logging.debug(f"§MM §NN cache hit: {boltz_cache[_mm_key]:.4f}")
                else:
                    try:
                        _mm_blk2 = await state['subtensor'].get_current_block()
                        _mm_ep2 = ((_mm_blk2 // state['epoch_length']) + 1) * state['epoch_length']
                        if _mm_ep2 - _mm_blk2 < 5:
                            bt.logging.info("§MM §NN: epoch ends in <5 blocks — stopping.")
                            _mm_stop = True
                            break
                    except Exception:
                        pass
                    _mm_uid_s = 0
                    _mm_vmbu_s = {_mm_uid_s: {"smiles": [_mm_smiles], "names": [_mm_row['product_name']]}}
                    _mm_sd_s: Dict[str, Any] = {_mm_uid_s: {}}
                    try:
                        await asyncio.to_thread(
                            wrapper.score_molecules_target,
                            _mm_vmbu_s, _mm_sd_s, subnet_config, '0x' + '0' * 64, True,
                        )
                        _mm_screen[_mm_smiles] = wrapper.per_molecule_metric.get(_mm_uid_s, {}).get(_mm_smiles, -math.inf)
                    except Exception as _mm_es:
                        bt.logging.error(f"§MM §NN fast-screen error: {_mm_es}")
                        _mm_screen[_mm_smiles] = -math.inf

            if _mm_stop:
                break

            # §NN Phase 2: full-score only the round's best fast-screened candidate.
            _mm_round_winner = max(
                (_s for _s, _v in _mm_screen.items() if math.isfinite(_v)),
                key=lambda _s: _mm_screen[_s],
                default=None,
            )
            if _mm_round_winner is not None:
                _mm_w_row = _mm_row_map[_mm_round_winner]
                _mm_w_canon = get_canonical_smiles(_mm_round_winner)
                _mm_w_key = (_mm_w_canon, protein)
                if _mm_w_key in boltz_cache:
                    _mm_score = boltz_cache[_mm_w_key]
                else:
                    try:
                        _mm_blk3 = await state['subtensor'].get_current_block()
                        _mm_ep3 = ((_mm_blk3 // state['epoch_length']) + 1) * state['epoch_length']
                        if _mm_ep3 - _mm_blk3 < 5:
                            bt.logging.info("§MM §NN: epoch ends in <5 blocks — stopping.")
                            _mm_stop = True
                            _mm_score = -math.inf
                        else:
                            _mm_uid_f = 0
                            _mm_vmbu_f = {_mm_uid_f: {"smiles": [_mm_round_winner], "names": [_mm_w_row['product_name']]}}
                            _mm_sd_f: Dict[str, Any] = {_mm_uid_f: {}}
                            await asyncio.to_thread(
                                wrapper.score_molecules_target,
                                _mm_vmbu_f, _mm_sd_f, subnet_config, '0x' + '0' * 64,
                            )
                            _mm_score = wrapper.per_molecule_metric.get(_mm_uid_f, {}).get(_mm_round_winner, -math.inf)
                            boltz_cache[_mm_w_key] = _mm_score
                            _disk_cache_put(db_path, _mm_w_canon, protein, _mm_score,
                                            product_name=_mm_w_row.get('product_name'))
                            if wrapper.last_inference_duration > 0:
                                state['boltz_time_per_mol'] = wrapper.last_inference_duration
                            bt.logging.info(
                                f"§MM §NN [{_mm_round_idx + 1}/{_mm_max_rounds}] full-scored winner: "
                                f"{_mm_w_row.get('product_name', '?')} boltz={_mm_score:.4f} "
                                f"(screened {len(_mm_screen)} hits)"
                            )
                    except Exception as _mm_e:
                        bt.logging.error(f"§MM §NN full-score error: {_mm_e}")
                        _mm_score = -math.inf

                if math.isfinite(_mm_score) and _mm_score > _mm_round_best_score:
                    _mm_round_best_score = _mm_score
                    _mm_round_best_smiles = _mm_round_winner
                    _mm_pname = _mm_w_row.get('product_name', '')
                    if _mm_pname:
                        _mm_orig = state['candidate_product'].split(',')
                        state['candidate_product'] = ','.join(
                            [_mm_pname] + [n for n in _mm_orig if n != _mm_pname]
                        )
                        bt.logging.info(
                            f"§MM §NN: new best: {_mm_pname} "
                            f"(boltz={_mm_score:.4f} > prev={_mm_best_score:.4f})"
                        )
                    _mm_improved = True

            # Expose §MM scores to §CC so the warm-start guard sees the full epoch best
            for _mm_ck2, _mm_cv2 in boltz_cache.items():
                if (
                    isinstance(_mm_ck2, tuple)
                    and _mm_ck2[1] == protein
                    and _mm_ck2[0] not in all_scores
                    and math.isfinite(_mm_cv2)
                ):
                    all_scores[_mm_ck2[0]] = _mm_cv2

            _mm_rounds_run = _mm_round_idx + 1

            if not _mm_improved:
                # §QQ/§VV Basin-hopping: instead of stopping on the first no-improvement
                # round, try the next-best scored molecule not yet used as a seed.
                # §VV adds scaffold awareness: prefer candidates with a Murcko scaffold
                # not yet explored by any §MM seed, so hops cross chemical space rather
                # than revisiting the same structural region.
                # Uses all_scores (updated above with §MM results) so molecules
                # discovered during §MM itself can serve as alternative seeds.
                from rdkit.Chem import MurckoScaffold as _Murcko

                def _get_scaffold_smi(smi: str) -> str:
                    m = Chem.MolFromSmiles(smi)
                    if not m:
                        return smi
                    try:
                        sc = _Murcko.GetScaffoldForMol(m)
                        return Chem.MolToSmiles(sc) if sc else smi
                    except Exception:
                        return smi

                _mm_tried_scaffolds: set = {_get_scaffold_smi(s) for s in _mm_tried_seeds}

                # First pass: prefer a candidate with an unseen Murcko scaffold.
                _mm_next_seed = max(
                    (
                        _s for _s, _v in all_scores.items()
                        if _s not in _mm_tried_seeds
                        and math.isfinite(_v)
                        and _get_scaffold_smi(_s) not in _mm_tried_scaffolds
                    ),
                    key=lambda _s: all_scores[_s],
                    default=None,
                )
                _mm_hop_novel = _mm_next_seed is not None

                # Fallback: any untried seed (even from an already-explored scaffold).
                if _mm_next_seed is None:
                    _mm_next_seed = max(
                        (_s for _s, _v in all_scores.items()
                         if _s not in _mm_tried_seeds and math.isfinite(_v)),
                        key=lambda _s: all_scores[_s],
                        default=None,
                    )

                if _mm_next_seed is None:
                    bt.logging.info(
                        f"§MM round {_mm_round_idx + 1}: no improvement; "
                        f"all {len(_mm_tried_seeds)} seed(s) exhausted — stopping."
                    )
                    break
                _hop_tag = "novel scaffold" if _mm_hop_novel else "same scaffold"
                bt.logging.info(
                    f"§MM round {_mm_round_idx + 1}: no improvement from current seed; "
                    f"§QQ/§VV basin-hop ({_hop_tag}) "
                    f"(score={all_scores[_mm_next_seed]:.4f})."
                )
                _mm_seed_smiles = _mm_next_seed
                # _mm_best_score unchanged — basin-hop doesn't claim an improvement
            else:
                # Advance seed to this round's best for the next iteration
                _mm_best_score = _mm_round_best_score
                _mm_seed_smiles = _mm_round_best_smiles

        bt.logging.info(
            f"§MM complete: {_mm_rounds_run} round(s) run, "
            f"final_best={_mm_best_score:.4f}"
        )

    # §XX — Tautomer Enumeration for Borderline Candidates.
    # After §MM converges, enumerate RDKit canonical tautomers of the epoch's
    # best-scoring molecule.  Tautomers share the same molecular formula but differ
    # in bond order and proton placement, producing distinct Morgan fingerprints that
    # map to *different* SAVI-2020 neighbours than the bioisosteric probes used by
    # SALSA.  This explores the H-bond donor/acceptor neighbourhood of the best
    # binder without any PSICHIC cost.  Only fires when ≥ 1 mol-time + 60 s of
    # runway remain after §MM.
    _xx_savi_pool = state.get('savi_stream_pool')
    _xx_best_smiles = (
        max(all_scores, key=lambda s: all_scores.get(s, -math.inf), default=None)
        if all_scores else None
    )
    _xx_best_epoch = max(
        (v for v in all_scores.values() if math.isfinite(v)), default=-math.inf
    )
    if (
        _xx_best_smiles is not None
        and math.isfinite(_xx_best_epoch)
        and _xx_savi_pool is not None
        and not _xx_savi_pool.empty
    ):
        try:
            _xx_blk0 = await state['subtensor'].get_current_block()
            _xx_ep0 = ((_xx_blk0 // state['epoch_length']) + 1) * state['epoch_length']
            _xx_t_mol = state.get('boltz_time_per_mol', 150.0)
            if (_xx_ep0 - _xx_blk0) * 12 > _xx_t_mol + 60:
                from rdkit.Chem.MolStandardize import rdMolStandardize as _rdMSt
                from utils.salsa import nearest_pool_molecules, precompute_pool_fps

                _xx_mol = Chem.MolFromSmiles(_xx_best_smiles)
                if _xx_mol is not None:
                    _xx_enumerator = _rdMSt.TautomerEnumerator()
                    _xx_tautomers = _xx_enumerator.Enumerate(_xx_mol)
                    _xx_seed_canon = Chem.MolToSmiles(_xx_mol)

                    _xx_novel: list = []
                    for _xx_t in _xx_tautomers:
                        if _xx_t is None:
                            continue
                        _xx_t_smi = Chem.MolToSmiles(_xx_t)
                        if _xx_t_smi == _xx_seed_canon:
                            continue
                        _ok, _ = is_boltz_safe_smiles(_xx_t_smi)
                        if not _ok:
                            continue
                        _xx_ha = get_heavy_atom_count(_xx_t_smi)
                        if _xx_ha is None or _xx_ha < 10 or _xx_ha > 35:
                            continue
                        _xx_novel.append(_xx_t_smi)

                    if _xx_novel:
                        bt.logging.info(
                            f"§XX: {len(_xx_novel)} novel tautomers of epoch best — "
                            f"mapping to SAVI-2020 neighbours"
                        )
                        _xx_valid_pool, _xx_pool_fps = precompute_pool_fps(_xx_savi_pool)
                        _xx_seen_neighbours: set = set()

                        for _xx_t_smi in _xx_novel[:6]:  # cap Boltz calls
                            try:
                                _xx_blk1 = await state['subtensor'].get_current_block()
                                _xx_ep1 = ((_xx_blk1 // state['epoch_length']) + 1) * state['epoch_length']
                                if (_xx_ep1 - _xx_blk1) * 12 < _xx_t_mol + 30:
                                    break
                            except Exception:
                                pass

                            _xx_near = nearest_pool_molecules(
                                _xx_t_smi, _xx_valid_pool, top_k=1, pool_fps=_xx_pool_fps
                            )
                            if _xx_near.empty:
                                continue
                            _xx_n_row = _xx_near.iloc[0]
                            _xx_n_smi = _xx_n_row['product_smiles']
                            _xx_n_pname = _xx_n_row.get('product_name', '')
                            if _xx_n_pname in _xx_seen_neighbours:
                                continue
                            _xx_seen_neighbours.add(_xx_n_pname)

                            _xx_n_canon = get_canonical_smiles(_xx_n_smi)
                            _xx_n_key = (_xx_n_canon, protein)

                            if _xx_n_key in boltz_cache:
                                _xx_n_score = boltz_cache[_xx_n_key]
                            else:
                                _xx_disk_s = _disk_cache_get(db_path, _xx_n_canon, protein)
                                if _xx_disk_s is not None:
                                    boltz_cache[_xx_n_key] = _xx_disk_s
                                    _xx_n_score = _xx_disk_s
                                else:
                                    _xx_uid = 0
                                    _xx_vmbu = {
                                        _xx_uid: {"smiles": [_xx_n_smi], "names": [_xx_n_pname]}
                                    }
                                    _xx_sd: Dict[str, Any] = {_xx_uid: {}}
                                    try:
                                        await asyncio.to_thread(
                                            wrapper.score_molecules_target,
                                            _xx_vmbu, _xx_sd, subnet_config,
                                            '0x' + '0' * 64,
                                        )
                                        _xx_n_score = wrapper.per_molecule_metric.get(
                                            _xx_uid, {}
                                        ).get(_xx_n_smi, -math.inf)
                                        boltz_cache[_xx_n_key] = _xx_n_score
                                        _disk_cache_put(
                                            db_path, _xx_n_canon, protein, _xx_n_score,
                                            product_name=_xx_n_pname or None,
                                        )
                                        if wrapper.last_inference_duration > 0:
                                            state['boltz_time_per_mol'] = wrapper.last_inference_duration
                                        bt.logging.info(
                                            f"§XX: tautomer SAVI neighbour "
                                            f"{_xx_n_pname!r} scored boltz={_xx_n_score:.4f}"
                                        )
                                    except Exception as _xx_be:
                                        bt.logging.error(f"§XX Boltz error: {_xx_be}")
                                        _xx_n_score = -math.inf

                            all_scores[_xx_n_canon] = _xx_n_score
                            if math.isfinite(_xx_n_score) and _xx_n_score > _xx_best_epoch:
                                _xx_prev_epoch = _xx_best_epoch
                                _xx_best_epoch = _xx_n_score
                                if _xx_n_pname:
                                    _xx_orig = state['candidate_product'].split(',')
                                    state['candidate_product'] = ','.join(
                                        [_xx_n_pname] + [n for n in _xx_orig if n != _xx_n_pname]
                                    )
                                    bt.logging.info(
                                        f"§XX: new epoch best from tautomer search — "
                                        f"{_xx_n_pname} "
                                        f"(boltz={_xx_n_score:.4f} > prev={_xx_prev_epoch:.4f})"
                                    )
        except Exception as _xx_err:
            bt.logging.warning(f"§XX tautomer search failed (non-fatal): {_xx_err}")

    # §WW — Multi-seed stability estimation for top-2 candidates.
    # Boltz-2 is a stochastic diffusion model; the same molecule can score
    # differently across random seeds.  The validator always uses seed=68, so
    # single-seed estimates are the ground truth for absolute scores.  However,
    # for the submission ORDER decision (which mol goes at position 0), knowing
    # which top-2 candidate is more STABLE across seeds is more reliable than a
    # single noisy estimate.  After §XX, if ≥ 4 mol-times remain, run seeds 42
    # and 123 on the top-2 candidates; put the one with the best MEAN at pos 0.
    # Alternate-seed scores are NOT written to disk cache — the validator uses
    # seed=68 and §CC must compare against seed-68 baselines.
    _ww_valid_all = {s: v for s, v in all_scores.items() if math.isfinite(v)}
    if len(_ww_valid_all) >= 2:
        try:
            _ww_blk = await state['subtensor'].get_current_block()
            _ww_ep = ((_ww_blk // state['epoch_length']) + 1) * state['epoch_length']
            _ww_remaining = (_ww_ep - _ww_blk) * 12
            _ww_t_mol = state.get('boltz_time_per_mol', 150.0)
            if _ww_remaining >= _ww_t_mol * 4 + 60:
                bt.logging.info(
                    f"§WW: {_ww_remaining:.0f}s remaining — multi-seed stability check on top-2..."
                )
                # Build SMILES→product_name lookup from all scored molecule pools
                _ww_name_lookup: Dict[str, str] = {}
                for _ww_frame in (candidates, state.get('global_candidate_pool'), state.get('savi_stream_pool')):
                    if _ww_frame is None or _ww_frame.empty:
                        continue
                    if 'product_smiles' not in _ww_frame.columns or 'product_name' not in _ww_frame.columns:
                        continue
                    for _, _ww_r in _ww_frame.iterrows():
                        _ww_ps = str(_ww_r.get('product_smiles', ''))
                        _ww_pn = str(_ww_r.get('product_name', ''))
                        if _ww_ps and _ww_pn:
                            _ww_name_lookup.setdefault(_ww_ps, _ww_pn)
                            _ww_can = get_canonical_smiles(_ww_ps)
                            if _ww_can:
                                _ww_name_lookup.setdefault(_ww_can, _ww_pn)

                _ww_extra_seeds = [42, 123]
                _ww_top2 = sorted(_ww_valid_all.items(), key=lambda kv: kv[1], reverse=True)[:2]
                _ww_mean: Dict[str, float] = {}
                _ww_stop_early = False

                for _ww_sm, _ww_seed68_score in _ww_top2:
                    if _ww_stop_early:
                        break
                    _ww_pname = (
                        _ww_name_lookup.get(_ww_sm)
                        or _ww_name_lookup.get(get_canonical_smiles(_ww_sm), '')
                    )
                    if not _ww_pname:
                        bt.logging.debug(f"§WW: no product_name for SMILES — using seed-68 score only.")
                        _ww_mean[_ww_sm] = _ww_seed68_score
                        continue

                    _ww_mol_scores = [_ww_seed68_score]
                    for _ww_seed in _ww_extra_seeds:
                        try:
                            _ww_blk2 = await state['subtensor'].get_current_block()
                            _ww_ep2 = ((_ww_blk2 // state['epoch_length']) + 1) * state['epoch_length']
                            if (_ww_ep2 - _ww_blk2) * 12 < _ww_t_mol + 30:
                                bt.logging.info("§WW: time guard fired — stopping early.")
                                _ww_stop_early = True
                                break
                        except Exception:
                            pass
                        _ww_uid = 0
                        _ww_vmbu = {_ww_uid: {"smiles": [_ww_sm], "names": [_ww_pname]}}
                        _ww_sd2: Dict[str, Any] = {_ww_uid: {}}
                        try:
                            await asyncio.to_thread(
                                wrapper.score_molecules_target,
                                _ww_vmbu, _ww_sd2, subnet_config, '0x' + '0' * 64, False, _ww_seed,
                            )
                            _ww_s = wrapper.per_molecule_metric.get(_ww_uid, {}).get(_ww_sm, -math.inf)
                            if math.isfinite(_ww_s):
                                _ww_mol_scores.append(_ww_s)
                            bt.logging.info(
                                f"§WW: {_ww_pname} seed={_ww_seed} score={_ww_s:.4f}"
                            )
                        except Exception as _ww_se:
                            bt.logging.error(f"§WW seed={_ww_seed} inference error: {_ww_se}")

                    _ww_mean[_ww_sm] = sum(_ww_mol_scores) / len(_ww_mol_scores)
                    bt.logging.info(
                        f"§WW: {_ww_pname} mean={_ww_mean[_ww_sm]:.4f} "
                        f"({len(_ww_mol_scores)} seed(s): {[f'{s:.4f}' for s in _ww_mol_scores]})"
                    )

                # Reorder pos-0 based on mean scores if top-2 both have estimates
                if len(_ww_mean) >= 2:
                    _ww_best_sm = max(_ww_mean, key=_ww_mean.get)
                    _ww_best_pname = (
                        _ww_name_lookup.get(_ww_best_sm)
                        or _ww_name_lookup.get(get_canonical_smiles(_ww_best_sm), '')
                    )
                    if _ww_best_pname:
                        _ww_orig = (state.get('candidate_product') or '').split(',')
                        if _ww_orig and _ww_orig[0] != _ww_best_pname:
                            state['candidate_product'] = ','.join(
                                [_ww_best_pname] + [n for n in _ww_orig if n != _ww_best_pname]
                            )
                            _ww_prev = _ww_orig[0] if _ww_orig else '?'
                            _ww_prev_mean = next(
                                (v for s, v in _ww_mean.items() if s != _ww_best_sm), -math.inf
                            )
                            bt.logging.info(
                                f"§WW: pos-0 swapped → {_ww_best_pname} "
                                f"(mean={_ww_mean[_ww_best_sm]:.4f}) over "
                                f"{_ww_prev} (mean={_ww_prev_mean:.4f})"
                            )
                        else:
                            bt.logging.info(
                                f"§WW: pos-0 confirmed ({_ww_best_pname}); "
                                f"mean-score agrees with seed-68 ordering."
                            )
        except Exception as _ww_err:
            bt.logging.warning(f"§WW multi-seed check failed (non-fatal): {_ww_err}")

    # Merge §FF / §MM scores into all_scores before the §CC guard.
    # The §MM loop exposes boltz_cache → all_scores at the end of each complete
    # round, but if §MM exits before round 0 (time check fails immediately) or
    # the §MM block is never entered, §FF scores stay in boltz_cache but never
    # reach all_scores.  Without this merge, best_new below would exclude the
    # §FF winner, making the §CC guard incorrectly promote a disk-cached
    # molecule over a better §FF molecule found this epoch.
    for _pre_cc_k, _pre_cc_v in boltz_cache.items():
        if (
            isinstance(_pre_cc_k, tuple)
            and len(_pre_cc_k) == 2
            and _pre_cc_k[1] == protein
            and _pre_cc_k[0] not in all_scores
            and math.isfinite(_pre_cc_v)
        ):
            all_scores[_pre_cc_k[0]] = _pre_cc_v

    # Warm-start guard (§CC): after reordering by this epoch's Boltz scores,
    # check whether the best molecule in the persistent disk cache beats the
    # best new score.  This covers the scenario where the warm-start molecule
    # (scored in a prior session) was evicted from global_candidate_pool when
    # PSICHIC reset the pool at epoch start, so it never appeared in
    # `candidates` above -- but its cached Boltz score is still valid and may
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

    # §C: Multi-molecule diversity reordering.
    # When num_molecules_boltz > 1, the validator awards a MACCS entropy bonus.
    # Reorder slots 1..N-1 by decreasing MACCS distance from the best Boltz molecule
    # so the submitted set maximises structural diversity -- and the bonus -- without
    # changing the best binder at position 0.
    # Currently a no-op (num_molecules_boltz == 1), but activates automatically
    # when the validator increases that parameter.
    _num_boltz = getattr(state.get('config'), 'num_molecules_boltz', 1)
    if _num_boltz > 1:
        try:
            _reorder_for_diversity(state)
        except Exception as _div_err:
            bt.logging.warning(f"§C diversity reorder failed (non-fatal): {_div_err}")

    # §KK. Post-Boltz early submission.
    # The validator breaks ties between miners with equal Boltz scores using
    # block_submitted (smallest = earliest wins), then push_time.  Waiting for the
    # 20-block window means we submit at most 20 blocks before epoch end -- but our
    # best Boltz molecule is already finalised now.  By submitting immediately after
    # Boltz validation we secure the earliest possible block_submitted for our
    # validated candidate, which can be 40–80 blocks (8–16 minutes) earlier on
    # fast / slow hardware respectively.
    #
    # MetadataError (chain rate-limit: too soon to commit again) is caught silently
    # inside submit_response.  In that case we fall back to the normal 20-block
    # submission gate with no regression.  The `candidate_product !=
    # last_submitted_product` guard prevents re-uploading the same molecule.
    try:
        if (
            state.get('candidate_product')
            and state.get('candidate_product') != state.get('last_submitted_product')
            and state.get('subtensor') is not None
        ):
            bt.logging.info("[KK] Post-Boltz early submission attempt.")
            await submit_response(state)
    except Exception as _kk_err:
        bt.logging.warning(f"[KK] Early submission failed (non-fatal): {_kk_err}")

    # §YY: Track the winning molecule's reaction class for biasing next-epoch
    # SAVI-2020 streaming.  The product_name prefix (e.g. "rxn:5") identifies
    # the SAVI-2020 reaction template that produced the best Boltz-validated
    # molecule.  Storing it in state lets stream_random_chunk_from_dataset
    # apply a 2× weight to CSV files from that reaction class, increasing the
    # probability of sampling structurally similar candidates on subsequent
    # epochs without excluding other classes.  Persists across epoch boundaries
    # (not reset in the epoch-reset block) so bias accumulates over the session.
    try:
        _yy_pname = (state.get('candidate_product') or '').split(',')[0].strip()
        if _yy_pname and '/' in _yy_pname:
            _yy_rxn = _yy_pname.split('/')[0]
            if _yy_rxn.startswith('rxn'):
                state['best_boltz_rxn_class'] = _yy_rxn
                bt.logging.info(
                    f"[YY] Winning reaction class: {_yy_rxn!r} "
                    f"— SAVI streaming biased toward this class next epoch."
                )
    except Exception as _yy_err:
        bt.logging.debug(f"[YY] rxn_class extraction failed (non-fatal): {_yy_err}")


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
        'savi_stream_pool': None,        # all PSICHIC-scored molecules this epoch (capped at 10000)
        'salsa_run_this_epoch': False,   # prevent duplicate SALSA runs per epoch
        'ga_run_this_epoch': False,      # prevent duplicate GradientGA runs per epoch
        'bbb_run_this_epoch': False,     # §BBB: prevent duplicate post-GA SALSA runs per epoch
        'best_ga_smiles': None,          # §BBB: best SMILES found by GradientGA this epoch
        'chembl_seeds': [],              # §SS: ChEMBL known actives, fetched at startup
        'best_boltz_rxn_class': None,    # §YY: winning rxn class for SAVI streaming bias (persists across epochs)
        'best_score': float('-inf'),
        'boltz_prescored': False,
        'last_submitted_product': None,
        'last_submission_time': None,
        'shutdown_event': asyncio.Event(),

        # Boltz score cache: {(canonical_smiles, protein_code): float}
        # In-memory layer -- persists across epochs within a session.
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
    # Boltz-2 predictions are significantly weaker without an MSA -- this call
    # is a no-op when the file already exists and fetches it via ColabFold
    # API (~1-5 min) only when the target has rotated to a new protein.
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
            bt.logging.warning(f"[MSA] Could not retrieve sequence for {config.weekly_target} -- skipping MSA fetch.")
    except Exception as _msa_exc:
        bt.logging.warning(f"[MSA] MSA check failed (non-fatal): {_msa_exc}")

    # §SS: Pre-fetch ChEMBL known actives for the weekly target in the background.
    # When SALSA fires (~15-30 min into the epoch), these actives are used as
    # additional seeds alongside the PSICHIC-ranked candidates.  ChEMBL actives
    # are validated binders (pChEMBL >= 7.0 → IC50 ≤ 100 nM), so they guide
    # SALSA's nearest-neighbour search toward chemically relevant SAVI-2020
    # molecules without requiring any PSICHIC calls on the actives themselves.
    async def _chembl_fetch_bg(target: str) -> None:
        try:
            _seeds_c = await asyncio.to_thread(get_chembl_seeds, target)
            state['chembl_seeds'] = _seeds_c
            bt.logging.info(f"[SS] ChEMBL: loaded {len(_seeds_c)} actives for {target}")
        except Exception as _ce:
            bt.logging.warning(f"[SS] ChEMBL fetch failed (non-fatal): {_ce}")

    asyncio.create_task(_chembl_fetch_bg(config.weekly_target))

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
                state['bbb_run_this_epoch'] = False
                state['best_ga_smiles'] = None
                state['best_score'] = float('-inf')
                state['boltz_prescored'] = False
                state['last_submitted_product'] = None
                state['shutdown_event'] = asyncio.Event()

                # Reset SAVI-2020 file-sampling seen-set so the new epoch
                # starts a fresh without-replacement cycle over all CSV files.
                _SAVI_SEEN_FILES.pop(state['hugging_face_dataset_repo'], None)

                # §SS: refresh ChEMBL seeds for the new epoch target.
                # In practice the weekly target rarely changes mid-session, so
                # this is mostly a no-op; the background task is cheap anyway.
                state['chembl_seeds'] = []
                asyncio.create_task(_chembl_fetch_bg(config.weekly_target))

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
