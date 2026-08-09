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
import difflib
import hashlib
import json
import sqlite3

import numpy as np

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
    upload_boltz_cache_export,
    download_boltz_cache_export,
    get_challenge_params_from_blockhash,
    get_heavy_atom_count,
    compute_maccs_entropy,
)
from utils.molecules import is_boltz_safe_smiles, get_canonical_smiles, molecule_unique_for_protein_hf
from utils.msa import ensure_msa
from utils.salsa import run_salsa_search
from utils.genetic import run_gradient_ga
from utils.chembl import get_chembl_seeds
from utils.surrogate import fit_surrogate, rank_pool_by_surrogate, ucb_rank_pool, fit_dual_surrogate, dual_surrogate_rank_pool, dual_surrogate_ucb_rank_pool, augment_pool_with_surrogate_blend, fit_dual_surrogate_with_embeddings, dual_surrogate_ucb_rank_pool_emb
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
        # Add columns when upgrading from older schemas (swallow if already present).
        for _col_ddl in (
            "ALTER TABLE boltz_cache ADD COLUMN product_name TEXT",
            # §YYYYY: store raw Boltz components alongside the combined score so the
            # §ZZ/§RRRR surrogate can train on individual APB / APV distributions
            # and future analysis can study per-component structure-activity trends.
            "ALTER TABLE boltz_cache ADD COLUMN affinity_prob_binary REAL",
            "ALTER TABLE boltz_cache ADD COLUMN affinity_pred_val REAL",
            # §DDDDDD: store ligand_iptm (inter-chain pTM for the ligand chain) so
            # the surrogate can down-weight low-confidence training examples.
            "ALTER TABLE boltz_cache ADD COLUMN ligand_iptm REAL",
            # §WWWWWWW: store std of per-sample LE across the 3 affinity diffusion
            # samples (primary + _1 + _2).  High std → noisy Boltz prediction →
            # down-weight in surrogate training alongside ligand_iptm.
            "ALTER TABLE boltz_cache ADD COLUMN boltz_le_std REAL",
            # §XXXXXXXX: store std of LE scores across the 3 §WW random seeds (68/42/123).
            # Captures inter-seed diffusion variance — orthogonal to boltz_le_std (which
            # measures intra-run sample variance).  High ww_std → noisy cross-seed
            # prediction → additional down-weight in surrogate training.
            "ALTER TABLE boltz_cache ADD COLUMN boltz_ww_std REAL",
            # §FFFFFFFFFF: store overall complex confidence (Boltz-2 combined pLDDT-like
            # score over the full complex).  Complements ligand_iptm (global alignment)
            # with a holistic quality signal.  LOW confidence_score → additional
            # down-weight in surrogate training alongside ligand_iptm.
            "ALTER TABLE boltz_cache ADD COLUMN confidence_score REAL",
            # §HHHHHHHHHH: store mean-pooled Boltz-2 evoformer trunk embedding for the
            # ligand chain (384D float32 vector serialised as BLOB).  Used to train a
            # protein-conditioned PCA-augmented RF surrogate that captures binding
            # complementarity beyond Morgan fingerprint topology.
            "ALTER TABLE boltz_cache ADD COLUMN boltz_embedding BLOB",
            # §IIIIIIIIII: store PSICHIC ligand-efficiency score (combined_score =
            # (target_affinity - weight*antitarget_affinity) / heavy_atoms) recorded
            # at Boltz call time.  Used as an extra surrogate training feature so the
            # RF model learns the PSICHIC→Boltz correction rather than relying solely
            # on Morgan FP / physicochemical descriptors.  NULL for legacy rows and
            # §MM SALSA-discovered molecules not in the PSICHIC candidate pool.
            "ALTER TABLE boltz_cache ADD COLUMN psichic_le REAL",
        ):
            try:
                conn.execute(_col_ddl)
            except Exception:
                pass
        # §BBBBB: key-value store for miner timing state that survives restarts.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS miner_state (
                key   TEXT PRIMARY KEY,
                value REAL NOT NULL,
                ts    INTEGER DEFAULT (strftime('%s','now'))
            )
        """)
        # §CCCCCC: extend miner_state with a text column for string state values.
        try:
            conn.execute("ALTER TABLE miner_state ADD COLUMN value_text TEXT")
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


def _disk_cache_put(
    db_path: str,
    smiles: str,
    protein: str,
    score: float,
    product_name: Optional[str] = None,
    apb: Optional[float] = None,
    apv: Optional[float] = None,
    ligand_iptm: Optional[float] = None,
    boltz_le_std: Optional[float] = None,
    confidence_score: Optional[float] = None,
    boltz_embedding: Optional[bytes] = None,
    psichic_le: Optional[float] = None,
) -> None:
    """Upsert a Boltz score into the persistent cache (silently ignores errors).

    §YYYYY: apb (affinity_probability_binary) and apv (affinity_pred_value) are
    stored separately so the §ZZ surrogate can later train on individual components
    and per-component analysis is available without re-running Boltz.
    §DDDDDD: ligand_iptm is also stored so the surrogate can apply confidence-based
    sample weighting — low ligand_iptm (<0.25) indicates an uncertain binding pose
    whose APB/APV values are noisy and should contribute less to surrogate training.
    §WWWWWWW: boltz_le_std is the std of LE across the 3 affinity diffusion samples;
    high variance → additional down-weight on top of ligand_iptm in the surrogate.
    §FFFFFFFFFF: confidence_score is the overall complex confidence (Boltz-2 combined
    pLDDT-like metric).  Low confidence_score → additional surrogate down-weight
    complementary to ligand_iptm (which measures global structural alignment).
    §IIIIIIIIII: psichic_le is the PSICHIC ligand-efficiency score at Boltz call time.
    Stored as surrogate training feature so the RF model learns PSICHIC→Boltz correction.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO boltz_cache "
                "(smiles, protein, score, product_name, affinity_prob_binary, "
                "affinity_pred_val, ligand_iptm, boltz_le_std, confidence_score, "
                "boltz_embedding, psichic_le) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (smiles, protein, score, product_name, apb, apv, ligand_iptm,
                 boltz_le_std, confidence_score, boltz_embedding, psichic_le),
            )
    except Exception:
        pass


def _emb_to_bytes(comps: dict) -> Optional[bytes]:
    """§HHHHHHHHHH: Extract embedding bytes from per_molecule_components dict for cache storage."""
    emb = comps.get('boltz_embedding')
    try:
        return emb.tobytes() if emb is not None else None
    except Exception:
        return None


def _compute_le_std(comps: dict) -> Optional[float]:
    """§WWWWWWW: Compute std of per-sample LE from boltz per_molecule_components dict.

    Returns the sample std of (apb - apv) / ha across the 3 affinity diffusion
    samples (primary + _1 + _2), or None when fewer than 2 samples are available.
    Called at each _disk_cache_put site so high-variance Boltz runs get an extra
    down-weight in surrogate training on top of the §DDDDDD ligand_iptm penalty.
    """
    try:
        ha = comps.get('heavy_atom_count') or 1
        if not ha or ha <= 0:
            return None
        pairs = [
            (comps.get('affinity_probability_binary'), comps.get('affinity_pred_value')),
            (comps.get('affinity_probability_binary1'), comps.get('affinity_pred_value1')),
            (comps.get('affinity_probability_binary2'), comps.get('affinity_pred_value2')),
        ]
        mem_scores = [
            (apb - apv) / ha
            for apb, apv in pairs
            if isinstance(apb, (int, float)) and isinstance(apv, (int, float))
            and math.isfinite(apb) and math.isfinite(apv)
        ]
        return float(np.std(mem_scores, ddof=0)) if len(mem_scores) >= 2 else None
    except Exception:
        return None


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


def _disk_cache_list_proteins(db_path: str, exclude: str = "") -> list:
    """Return distinct protein accessions stored in the cache, excluding *exclude*."""
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT protein FROM boltz_cache WHERE protein != ?",
                (exclude,),
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def _load_miner_state(db_path: str, key: str) -> Optional[float]:
    """§BBBBB: Return a persisted miner state value by key, or None on miss/error."""
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT value FROM miner_state WHERE key=?", (key,)
            ).fetchone()
        return float(row[0]) if row else None
    except Exception:
        return None


def _save_miner_state(db_path: str, key: str, value: float) -> None:
    """§BBBBB: Persist a miner state value (silently ignores errors)."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO miner_state (key, value) VALUES (?,?)",
                (key, value),
            )
    except Exception:
        pass


def _load_miner_state_text(db_path: str, key: str) -> Optional[str]:
    """§CCCCCC: Return a persisted text miner state value, or None on miss/error."""
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT value_text FROM miner_state WHERE key=?", (key,)
            ).fetchone()
        return str(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def _save_miner_state_text(db_path: str, key: str, text_value: str) -> None:
    """§CCCCCC: Persist a text miner state value (silently ignores errors)."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO miner_state (key, value, value_text) VALUES (?,?,?)",
                (key, 0.0, text_value),
            )
    except Exception:
        pass


def _save_rxn_class_scores(db_path: str, rxn_class: str, score: float) -> None:
    """
    §EEEEEE: Append *score* to the per-reaction-class Boltz score history.

    History is stored in the `miner_state` table under key
    'rxn_class_scores_json' as a JSON object mapping rxn_class → [score, ...].
    Each list is capped at 50 entries (most-recent-first) to prevent unbounded
    growth.  Silently ignores all errors so a DB write failure never interrupts
    the main loop.
    """
    try:
        raw = _load_miner_state_text(db_path, 'rxn_class_scores_json') or '{}'
        data: Dict[str, list] = json.loads(raw)
        scores = data.get(rxn_class, [])
        scores.append(round(score, 6))
        data[rxn_class] = scores[-50:]
        _save_miner_state_text(db_path, 'rxn_class_scores_json', json.dumps(data))
    except Exception:
        pass


def _load_rxn_class_weights(db_path: str) -> Dict[str, float]:
    """
    §EEEEEE: Compute per-class SAVI sampling weights from persisted score history.

    Reads the JSON history written by _save_rxn_class_scores, computes the mean
    Boltz score per class, then assigns rank-based sampling weights:

        top-1 class → 4×   (highest mean score)
        top-2 class → 2×
        top-3 class → 1.5×
        all others  → 1×   (unchanged from baseline uniform)

    Returns an empty dict when no history exists or on any error — callers fall
    back to the §YY single-class 2× bias or uniform sampling in that case.
    """
    try:
        raw = _load_miner_state_text(db_path, 'rxn_class_scores_json')
        if not raw:
            return {}
        data: Dict[str, list] = json.loads(raw)
        means = {k: sum(v) / len(v) for k, v in data.items() if v}
        if not means:
            return {}
        ranked = sorted(means, key=means.get, reverse=True)
        weights: Dict[str, float] = {cls: 1.0 for cls in means}
        for i, cls in enumerate(ranked[:3]):
            weights[cls] = [4.0, 2.0, 1.5][i]
        return weights
    except Exception:
        return {}


def _compute_ha_bucket_le(db_path: str, protein: str) -> tuple:
    """
    §OOOOOO: Return (avg_le_frag, avg_le_drug, n_frag, n_drug) from the Boltz
    disk cache for *protein*.  'frag' = ≤18 heavy atoms; 'drug' = >18.

    Used to adapt the §TTTT fragment-slot quota toward evidence-based sizing:
    if historical small-molecule Boltz scores consistently exceed drug-like scores,
    more pool slots should be reserved for the ≤18-HA bucket.

    Returns (None, None, 0, 0) when the cache is empty, unavailable, or the
    SMILES cannot be parsed — all safe fallback cases for the caller.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT smiles, score FROM boltz_cache WHERE protein=? AND score > -1e9",
                (protein,),
            ).fetchall()
        if not rows:
            return (None, None, 0, 0)
        frag_scores: List[float] = []
        drug_scores: List[float] = []
        for smiles, score in rows:
            ha = get_heavy_atom_count(smiles)
            if ha is None:
                continue
            if ha <= 18:
                frag_scores.append(score)
            else:
                drug_scores.append(score)
        avg_frag = sum(frag_scores) / len(frag_scores) if frag_scores else None
        avg_drug = sum(drug_scores) / len(drug_scores) if drug_scores else None
        return (avg_frag, avg_drug, len(frag_scores), len(drug_scores))
    except Exception:
        return (None, None, 0, 0)


def _cross_target_seeds_from_cache(
    db_path: str,
    current_protein: str,
    identity_threshold: float = 0.40,
    limit: int = 3,
) -> list:
    """
    Return SMILES for Boltz-validated molecules from prior-target proteins that
    are sequence homologs of *current_protein* (identity >= threshold).

    Must be called BEFORE _cleanup_boltz_cache so prior-protein rows still exist.
    Uses difflib.SequenceMatcher for fast approximate identity — no extra deps.
    """
    prior_proteins = _disk_cache_list_proteins(db_path, exclude=current_protein)
    if not prior_proteins:
        return []
    current_seq = get_sequence_from_protein_code(current_protein)
    if not current_seq:
        return []
    results: list = []
    for prior in prior_proteins:
        prior_seq = get_sequence_from_protein_code(prior)
        if not prior_seq:
            continue
        ratio = difflib.SequenceMatcher(None, current_seq, prior_seq).ratio()
        if ratio >= identity_threshold:
            hits = _disk_cache_get_candidates(db_path, prior, limit=limit)
            for h in hits:
                sm = h.get('product_smiles', '')
                if sm and sm not in results:
                    results.append(sm)
            bt.logging.info(
                f"§WWWWW: homolog {prior} ({ratio:.1%} seq-identity) → "
                f"{len(hits)} cross-target seed(s)"
            )
    return results


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


def stream_random_chunk_from_dataset(dataset_repo: str, chunk_size: int, rxn_bias: Optional[str] = None, rxn_weights: Optional[Dict[str, float]] = None) -> Any:
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
    # §EEEEEE: when rxn_weights (per-class score history) is available, apply
    # rank-based weights (4×/2×/1.5×/1×) for the top-3 classes instead of the
    # binary 2×/1× used by §YY — captures multi-modal binding landscapes where
    # more than one reaction class consistently produces good Boltz binders.
    # Falls back to §YY (single-class 2×) then uniform when not populated.
    if rxn_weights:
        _eeeeee_w = []
        for f in available:
            best_w = 1.0
            for cls, w in rxn_weights.items():
                if cls in f and w > best_w:
                    best_w = w
            _eeeeee_w.append(best_w)
        random_file = random.choices(available, weights=_eeeeee_w, k=1)[0]
    elif rxn_bias:
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
                rxn_weights=state.get('rxn_class_weights') or None,
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

                # §YYYYYY: Re-rank with startup dual surrogate when available.
                # augment_pool_with_surrogate_blend only fires for RF models (≥100 pts);
                # on cold starts or Ridge-only quality it returns df unchanged.
                _yyyyyy_sds = state.get('startup_dual_surrogate')
                if _yyyyyy_sds is not None:
                    _yyyyyy_aug = augment_pool_with_surrogate_blend(df, _yyyyyy_sds)
                    if 'surrogate_salsa_score' in _yyyyyy_aug.columns:
                        df = _yyyyyy_aug.sort_values(
                            'surrogate_salsa_score', ascending=False
                        ).drop(columns=['surrogate_salsa_score'])
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
                        # §TTTT: Fragment-slot quota — reserve up to 1000 of the 10,000
                        # pool slots for molecules with ≤18 heavy atoms.  The validator
                        # scoring formula divides by heavy_atom_count, so a fragment with
                        # moderate absolute affinity beats a drug-like molecule with higher
                        # absolute affinity.  Without a quota, fragments are crowded out
                        # by the more abundant 20–35 HA SAVI-2020 products even though
                        # the ligand-efficiency PSICHIC score already normalises by HA.
                        # Keeping fragments in the pool lets SALSA nearest-neighbour
                        # lookup map perturbation probes to small SAVI-2020 products and
                        # explore fragment-like chemical space.  Drug-like molecules (>18
                        # HA) fill the remaining 9,000 slots sorted by combined_score.
                        # NaN heavy_atoms (should not occur but defensive) are treated as
                        # drug-like (25 HA assumed) to avoid accidental fragment mis-count.
                        # §OOOOOO: quota adapts at startup from Boltz cache evidence.
                        _tttt_quota = state.get('tttt_fragment_quota', 1000)
                        _tttt_ha = _pool_combined['heavy_atoms'].fillna(25)
                        _tttt_frags = _pool_combined[_tttt_ha <= 18].head(_tttt_quota)
                        _tttt_rest  = _pool_combined[_tttt_ha  > 18].head(10000 - _tttt_quota)
                        _pool_capped = pd.concat([_tttt_frags, _tttt_rest], ignore_index=True)
                        _pool_capped.drop_duplicates(subset=['product_name'], inplace=True)
                        _pool_capped.sort_values('combined_score', ascending=False, inplace=True)
                        state['savi_stream_pool'] = _pool_capped.head(10000).reset_index(drop=True)

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
                            _db_path_s = state.get('boltz_cache_db', BOLTZ_CACHE_DB)
                            _prot_s = state['config'].weekly_target
                            _dual_s = fit_dual_surrogate(_db_path_s, _prot_s)
                            if _dual_s is not None and not state['global_candidate_pool'].empty:
                                state['global_candidate_pool'] = dual_surrogate_ucb_rank_pool(
                                    state['global_candidate_pool'], _dual_s
                                )
                                bt.logging.info("[§AAAAAA] SALSA seeds re-ranked by dual APB+APV UCB surrogate.")
                            else:
                                _zz_seed_model = fit_surrogate(_db_path_s, _prot_s)
                                if _zz_seed_model is not None and not state['global_candidate_pool'].empty:
                                    state['global_candidate_pool'] = ucb_rank_pool(
                                        state['global_candidate_pool'], _zz_seed_model
                                    )
                                    bt.logging.info("[§RRRR/ZZ] SALSA seeds re-ranked by UCB surrogate.")
                        except Exception as _zz_s_err:
                            bt.logging.debug(f"[ZZ/YYYYY] SALSA seed re-rank skipped: {_zz_s_err}")

                        # §OOOO: multi-seed SALSA with scaffold-diverse input seeds.
                        # Take the top-5 (by PSICHIC / surrogate-reranked order) then
                        # apply _scaffold_diverse_candidates to select the 3 most
                        # structurally distinct starting points.  When SALSA or prior
                        # streaming has converged to one scaffold region, this ensures
                        # the 3 SALSA passes each explore a genuinely different chemical
                        # neighbourhood — complementing §NNNN (output diversity) and
                        # §VV/§QQ (basin-hop diversity) with upstream seed diversity.
                        # Runtime overhead: one extra MurckoScaffold call per candidate
                        # (~0.5 ms total) -- negligible vs Boltz.
                        _seed_cand_n = min(5, len(state['global_candidate_pool']))
                        _seed_cand = state['global_candidate_pool'].head(_seed_cand_n)
                        _seed_cand = _scaffold_diverse_candidates(_seed_cand, max_k=3)
                        _n_seeds = len(_seed_cand)
                        _seeds = _seed_cand['product_smiles'].tolist()
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
                        #
                        # §SSSSSS: diversity-aware seed selection.  Pull top-20 cache entries and
                        # apply max-min Tanimoto selection, always keeping rank-1 for exploitation.
                        # When §MM hill-climbing has converged to one scaffold family, the top-3 by
                        # score are nearly identical; max-min diversity ensures each of the 3 SALSA
                        # passes explores a genuinely different chemical neighbourhood.
                        _uu_db_path = state.get('boltz_cache_db', BOLTZ_CACHE_DB)
                        _uu_protein = state['config'].weekly_target
                        _uu_cached = _disk_cache_get_candidates(_uu_db_path, _uu_protein, limit=20)
                        _uu_valid = []
                        for _uu_row in _uu_cached:
                            _uu_sm = _uu_row.get('product_smiles', '')
                            if (
                                _uu_sm
                                and _uu_sm not in _seeds
                                and Chem.MolFromSmiles(_uu_sm) is not None
                                and is_boltz_safe_smiles(_uu_sm)[0]
                            ):
                                _uu_valid.append(_uu_sm)
                        _uu_seeds: list = []
                        if len(_uu_valid) <= 3:
                            _uu_seeds = _uu_valid
                        else:
                            # §SSSSSS: max-min diversity selection from top-20 cache candidates.
                            try:
                                from rdkit.Chem import AllChem
                                from rdkit import DataStructs
                                _uu_mols = [Chem.MolFromSmiles(s) for s in _uu_valid]
                                _uu_fps = [
                                    AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
                                    for m in _uu_mols
                                ]
                                _uu_sel = [0]  # always keep rank-1 (best score)
                                while len(_uu_sel) < 3:
                                    best_i, best_dist = -1, -1.0
                                    for i in range(len(_uu_valid)):
                                        if i in _uu_sel:
                                            continue
                                        min_sim = min(
                                            DataStructs.TanimotoSimilarity(_uu_fps[i], _uu_fps[j])
                                            for j in _uu_sel
                                        )
                                        dist = 1.0 - min_sim
                                        if dist > best_dist:
                                            best_dist, best_i = dist, i
                                    if best_i >= 0:
                                        _uu_sel.append(best_i)
                                    else:
                                        break
                                _uu_seeds = [_uu_valid[i] for i in _uu_sel]
                            except Exception:
                                _uu_seeds = _uu_valid[:3]  # fallback: top-3 by score
                        if _uu_seeds:
                            _seeds = _seeds + _uu_seeds
                            bt.logging.info(
                                f"[UU+§SSSSSS] {len(_uu_seeds)} cache seed(s) added to SALSA "
                                f"(diversity-selected from top-{len(_uu_valid)} cached)."
                            )

                        # §WWWWW: extend with seeds from homologous prior-target proteins.
                        # Populated before _cleanup_boltz_cache at startup; only useful on
                        # the first epoch after a weekly-target rotation to a family member.
                        _wwwww_ok = [
                            s for s in state.get('cross_target_seeds', [])
                            if s not in _seeds
                            and Chem.MolFromSmiles(s) is not None
                            and is_boltz_safe_smiles(s)[0]
                        ][:3]
                        if _wwwww_ok:
                            _seeds = _seeds + _wwwww_ok
                            bt.logging.info(
                                f"[WWWWW] Adding {len(_wwwww_ok)} cross-target homolog seed(s) to SALSA."
                            )

                        _seed_parts = [f"{_n_seeds} PSICHIC"]
                        if _chembl_ok:
                            _seed_parts.append(f"{len(_chembl_ok)} ChEMBL")
                        if _uu_seeds:
                            _seed_parts.append(f"{len(_uu_seeds)} Boltz-cache")
                        if _wwwww_ok:
                            _seed_parts.append(f"{len(_wwwww_ok)} cross-target")
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
                                    'combined_score',
                                    'product_smiles',
                                    'product_name',
                                    _salsa_operator_weights(_seed_smiles),  # §ZZZZZ
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
                        # §UUUUUU: Surrogate-guided GA fitness — when the dual RF surrogate
                        # has ≥100 cache points, augment the GA pool and seed_df with a
                        # surrogate-blended score column so GA's tournament selection and
                        # population sorting optimise toward the Boltz objective rather than
                        # the PSICHIC signal.  Falls back silently to 'combined_score' when
                        # the surrogate is unavailable (< 100 pts / Ridge tier / first epoch).
                        _uu_ga_score_col = 'combined_score'
                        _uu_ga_pool = ga_pool
                        _uu_ga_seed = state['global_candidate_pool']
                        try:
                            _uu_dual = fit_dual_surrogate(
                                state.get('boltz_cache_db', BOLTZ_CACHE_DB),
                                state['config'].weekly_target
                            )
                            if _uu_dual is not None:
                                _uu_ga_pool = augment_pool_with_surrogate_blend(_uu_ga_pool, _uu_dual)
                                _uu_ga_seed = augment_pool_with_surrogate_blend(_uu_ga_seed, _uu_dual)
                                if 'surrogate_salsa_score' in _uu_ga_pool.columns:
                                    _uu_ga_score_col = 'surrogate_salsa_score'
                                    bt.logging.info(
                                        "[§UUUUUU] GA pool augmented with dual surrogate blend "
                                        "— evolving toward Boltz-calibrated fitness."
                                    )
                        except Exception as _uu_ga_err:
                            bt.logging.debug(f"[§UUUUUU] GA surrogate blend skipped: {_uu_ga_err}")
                        try:
                            ga_hits = await asyncio.to_thread(
                                run_gradient_ga,
                                _uu_ga_seed,
                                _uu_ga_pool,
                                5,   # n_generations
                                50,  # pop_size
                                5,   # top_k
                                _uu_ga_score_col,  # §UUUUUU: surrogate-blended when RF available
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
                                'combined_score',
                                'product_smiles',
                                'product_name',
                                _salsa_operator_weights(state['best_ga_smiles']),  # §ZZZZZ
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
                    # §PPPPPP: Export Boltz cache to GitHub so it survives container
                    # restarts (boltz_score_cache.db is gitignored and ephemeral).
                    try:
                        _pp_db = state.get('boltz_cache_db', BOLTZ_CACHE_DB)
                        _pp_protein = state['config'].weekly_target
                        if upload_boltz_cache_export(_pp_db, _pp_protein):
                            bt.logging.info("[§PPPPPP] Boltz cache exported to GitHub.")
                        else:
                            bt.logging.debug("[§PPPPPP] Cache export skipped or failed.")
                    except Exception as _pp_err:
                        bt.logging.warning(f"[§PPPPPP] Cache export non-fatal: {_pp_err}")
                else:
                    bt.logging.error(f"Failed to upload file to GitHub for {commit_content}")
            except Exception as e:
                bt.logging.error(f"Failed to upload file for {commit_content}: {e}")


# ----------------------------------------------------------------------------
# 6. BOLTZ-2 PRE-SCORING
# ----------------------------------------------------------------------------

def _salsa_operator_weights(seed_smiles: str) -> dict | None:
    """
    §ZZZZZ: Return SALSA operator weights biased by the seed molecule's HA count.

    The Boltz scoring formula divides by heavy_atom_count, so shrinking the
    seed molecule directly improves the score floor.  When the seed is large
    (>25 HA) we favour terminal_remove; when it is small (<15 HA) we favour
    fg_add to grow toward the sweet-spot ligand-efficiency range.  The middle
    range gets equal weights (None → generate_perturbations default).
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(seed_smiles)
        if mol is None:
            return None
        ha = mol.GetNumHeavyAtoms()
    except Exception:
        return None

    if ha > 25:
        # Seed is large — bias toward shrinking (terminal_remove 50% budget)
        return {'bioisostere': 1.0, 'fg_add': 0.5, 'terminal_remove': 2.5, 'ring_walk': 0.5}
    if ha < 15:
        # Seed is small — bias toward growing (fg_add 44% budget)
        return {'bioisostere': 1.0, 'fg_add': 2.0, 'terminal_remove': 0.5, 'ring_walk': 1.0}
    return None  # 15 ≤ ha ≤ 25: equal weights


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

    # §IIIIIIIIII: Build PSICHIC LE map before candidates are resolved.
    # global_candidate_pool / candidate_molecules have combined_score = PSICHIC LE.
    # The disk-cache fallback sets combined_score = Boltz score — those get None.
    _psichic_le_map: Dict[str, float] = {}
    for _pl_src in (state.get('global_candidate_pool'), state.get('candidate_molecules')):
        if _pl_src is not None and not _pl_src.empty and 'combined_score' in _pl_src.columns:
            for _, _pl_row in _pl_src.iterrows():
                _pl_sm = str(_pl_row.get('product_smiles', ''))
                _pl_val = _pl_row.get('combined_score')
                if _pl_sm and isinstance(_pl_val, (int, float)) and math.isfinite(float(_pl_val)):
                    _psichic_le_map[get_canonical_smiles(_pl_sm)] = float(_pl_val)
            break  # prefer global_candidate_pool when available

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

    # §VVVVVV: Filter candidates already present in the Submission-Archive before
    # spending GPU time scoring them.  The validator's InChIKey uniqueness check
    # rejects the entire submission if the top molecule's InChIKey is in the archive
    # for this target protein.  Pre-filtering here ensures every Boltz-2 call is
    # spent on a candidate that can actually be accepted.
    # molecule_unique_for_protein_hf() uses a 60 s TTL HuggingFace cache, so the
    # first call per minute hits the network and subsequent calls are free.
    # Fails open (skips filtering) on any network or parse error.
    try:
        _vv_mask = candidates['product_smiles'].apply(
            lambda s: molecule_unique_for_protein_hf(protein, s)
        )
        _vv_n_filtered = int((~_vv_mask).sum())
        candidates = candidates[_vv_mask].reset_index(drop=True)
        if _vv_n_filtered > 0:
            bt.logging.info(
                f"[§VVVVVV] Removed {_vv_n_filtered} already-submitted candidate(s) "
                f"(InChIKey in Submission-Archive for {protein}) before Boltz scoring."
            )
    except Exception as _vv_err:
        bt.logging.debug(f"[§VVVVVV] Archive pre-filter skipped (non-fatal): {_vv_err}")

    # §ZZ / §YYYYY / §HHHHHHHHHH: Re-rank Boltz candidates by surrogate when ≥ 40 scores cached.
    # §HHHHHHHHHH: try the embedding-augmented dual surrogate first (RF tier + ≥20 embedding rows).
    # Embedding features encode protein-conditioned ligand binding complementarity from the
    # Boltz-2 evoformer trunk, improving surrogate NDCG beyond Morgan FP + physchem alone.
    # Falls back: §YYYYY dual UCB → §RRRR UCB (combined) → PSICHIC ordering.
    if not candidates.empty:
        try:
            _emb_dual = fit_dual_surrogate_with_embeddings(db_path, protein)
            if _emb_dual is not None:
                candidates = dual_surrogate_ucb_rank_pool_emb(candidates, _emb_dual)
                _emb_tag = "[emb+RF]" if _emb_dual[2] is not None else "[RF]"
                bt.logging.info(
                    f"[§HHHHHHHHHH/§AAAAAA] Pre-Boltz candidates re-ranked by embedding-augmented "
                    f"dual UCB surrogate {_emb_tag} ({len(candidates)} entries, target={protein})."
                )
            else:
                _dual = fit_dual_surrogate(db_path, protein)
                if _dual is not None:
                    candidates = dual_surrogate_ucb_rank_pool(candidates, _dual)
                    bt.logging.info(
                        f"[§AAAAAA] Pre-Boltz candidates re-ranked by dual APB+APV UCB surrogate "
                        f"({len(candidates)} entries, target={protein})."
                    )
                else:
                    _zz_model = fit_surrogate(db_path, protein)
                    if _zz_model is not None:
                        # §RRRR: UCB ranking when surrogate is RF (≥100 pts).
                        candidates = ucb_rank_pool(candidates, _zz_model)
                        bt.logging.info(
                            f"[§RRRR/ZZ] Pre-Boltz candidates re-ranked by UCB surrogate "
                            f"({len(candidates)} entries, target={protein})."
                        )
        except Exception as _zz_err:
            bt.logging.debug(f"[ZZ/YYYYY/HHHHHHHHHH] Candidate surrogate re-rank skipped: {_zz_err}")

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
    # §GGGGGG: Epoch-scoped fast-screen cache — stores fast-mode (50-step) Boltz scores
    # keyed by canonical SMILES for the current prescoring pass.  Unlike boltz_cache
    # (full-quality, persisted to disk), this is ephemeral and intentionally low-fidelity.
    # Purpose: when §FF or a later §MM round generates the same SALSA hit as an earlier
    # round, skip the GPU fast-screen call and reuse the previously computed fast score.
    # Zero regression: fast scores are only used for within-round ranking decisions, not
    # cached to disk, and full-quality scores always supersede them when available.
    _epoch_fast_cache: Dict[str, float] = {}

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
                    # §SSSS: Ensemble scoring — average ligand-efficiency across the 3
                    # Boltz affinity ensemble members (primary + _1 + _2).  Averaging
                    # valid pairs reduces noise in close-call ordering decisions without
                    # changing what is cached: all_scores / disk cache always store the
                    # primary score so §CC and §MM comparisons stay consistent.
                    _ssss_eff = _rr_score  # default: confidence-adjusted primary
                    try:
                        _ha = _comps.get('heavy_atom_count') or 1
                        if _ha and _ha > 0:
                            _ssss_pairs = [
                                (_comps.get('affinity_probability_binary'),
                                 _comps.get('affinity_pred_value')),
                                (_comps.get('affinity_probability_binary1'),
                                 _comps.get('affinity_pred_value1')),
                                (_comps.get('affinity_probability_binary2'),
                                 _comps.get('affinity_pred_value2')),
                            ]
                            _mem_scores = [
                                (apb - apv) / _ha
                                for apb, apv in _ssss_pairs
                                if isinstance(apb, (int, float))
                                and isinstance(apv, (int, float))
                                and math.isfinite(apb) and math.isfinite(apv)
                            ]
                            if len(_mem_scores) >= 2:
                                _ssss_raw = sum(_mem_scores) / len(_mem_scores)
                                # Preserve §RR confidence-penalty ratio if it was applied.
                                if math.isfinite(score) and score != 0 and _rr_score != score:
                                    _ssss_eff = _ssss_raw * (_rr_score / score)
                                else:
                                    _ssss_eff = _ssss_raw
                                bt.logging.debug(
                                    f"  [§SSSS] ensemble={_ssss_eff:.4f} "
                                    f"(n={len(_mem_scores)}, primary={score:.4f})"
                                )
                    except Exception:
                        pass  # fall back to _rr_score
                    _rr_eff_scores[smiles] = _ssss_eff

                    # Persist to both cache layers (product_name enables warm-start §AA)
                    boltz_cache[key] = score
                    _pname = row.get('product_name')
                    if not isinstance(_pname, str):
                        _pname = None
                    # §YYYYY: extract primary APB/APV for component caching.
                    # §DDDDDD: also extract ligand_iptm for confidence-weighted surrogate.
                    # §WWWWWWW: compute per-sample LE std for additional variance weighting.
                    # §FFFFFFFFFF: also extract confidence_score (overall complex confidence).
                    _yyyyy_apb = _comps.get('affinity_probability_binary')
                    _yyyyy_apv = _comps.get('affinity_pred_value')
                    _dddddd_li = _comps.get('ligand_iptm')
                    _wwwwwww_le_std = _compute_le_std(_comps)
                    _fffff_cs = _comps.get('confidence_score')
                    _disk_cache_put(
                        db_path, canon, protein, score, product_name=_pname,
                        apb=_yyyyy_apb if isinstance(_yyyyy_apb, (int, float)) else None,
                        apv=_yyyyy_apv if isinstance(_yyyyy_apv, (int, float)) else None,
                        ligand_iptm=_dddddd_li if isinstance(_dddddd_li, (int, float)) else None,
                        boltz_le_std=_wwwwwww_le_std,
                        confidence_score=_fffff_cs if isinstance(_fffff_cs, (int, float)) else None,
                        boltz_embedding=_emb_to_bytes(_comps),
                        psichic_le=_psichic_le_map.get(canon),  # §IIIIIIIIII
                    )

                    # Adaptive trigger: one molecule gives the most accurate per-mol timing
                    elapsed = wrapper.last_inference_duration
                    if elapsed > 0:
                        state['boltz_time_per_mol'] = elapsed  # persist for dynamic budget calc
                        adaptive_trigger = int(elapsed * max_candidates / 12) + 20
                        state['boltz_trigger_blocks'] = max(adaptive_trigger, 30)
                        # §BBBBB: persist timing so next restart uses calibrated trigger.
                        _save_miner_state(db_path, 'boltz_time_per_mol', elapsed)
                        _save_miner_state(db_path, 'boltz_trigger_blocks',
                                          float(state['boltz_trigger_blocks']))
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

    # §HHHHHH: Pre-compute surrogate-blended SAVI pool for §FF/§MM SALSA hill-climbing.
    # When the dual RF surrogate (§YYYYY/§AAAAAA) has >=100 cache points, augment the
    # SAVI stream pool with a 'surrogate_salsa_score' column that blends PSICHIC's
    # combined_score (40%) with the surrogate's predicted (APB - APV) / HA (60%), both
    # min-max normalised.  Passing this column as SALSA's score_col makes hill-climbing
    # converge toward molecules the miner has learned score well under Boltz-2 for THIS
    # target's chemistry -- complementing the general-purpose PSICHIC signal.
    # Falls back to 'combined_score' (pure PSICHIC) when the surrogate is unavailable,
    # uses Ridge (<100 pts), or augmentation fails -- zero regression in that case.
    _hhhhhh_pool: Optional[pd.DataFrame] = None
    _hhhhhh_score_col: str = 'combined_score'
    _hhhhhh_base = state.get('savi_stream_pool')
    if _dual is not None and _hhhhhh_base is not None and not _hhhhhh_base.empty:
        try:
            _aug = augment_pool_with_surrogate_blend(_hhhhhh_base, _dual)
            if 'surrogate_salsa_score' in _aug.columns:
                _hhhhhh_pool = _aug
                _hhhhhh_score_col = 'surrogate_salsa_score'
                bt.logging.info(
                    f"[§HHHHHH] SAVI pool ({len(_aug)} mols) augmented with surrogate-blend "
                    f"score -- §FF/§MM SALSA will hill-climb on blended Boltz signal."
                )
        except Exception as _hhhhhh_err:
            bt.logging.debug(f"[§HHHHHH] Pool augmentation skipped: {_hhhhhh_err}")

    # §FF: Boltz-guided SALSA -- second SALSA pass seeded from the best Boltz molecule.
    # The main loop above uses the PSICHIC-ranked candidate pool as seeds; PSICHIC and
    # Boltz-2 correlate imperfectly, so the validated Boltz winner may occupy a different
    # region of chemical space.  By running SALSA from the actual best-Boltz SMILES we
    # explore its chemical neighbourhood and may find SAVI-2020 molecules that score
    # even better -- without any additional PSICHIC overhead.
    # Only fires when the epoch has >=2 mol-lengths + 2 min of runway remaining.
    _ff_best_smiles = max(all_scores, key=lambda s: all_scores.get(s, -math.inf), default=None)
    _savi_pool_ff = _hhhhhh_pool if _hhhhhh_pool is not None else state.get('savi_stream_pool')
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
                    5,   # top_k — §NNNN: wider net for scaffold-diversity selection below
                    _hhhhhh_score_col,  # §HHHHHH: surrogate-blended or 'combined_score' fallback
                    'product_smiles',
                    'product_name',
                    _salsa_operator_weights(_ff_best_smiles),  # §ZZZZZ
                )
                # §NNNN: scaffold-diverse selection — ensures §NN fast-screen tests
                # different chemical hypotheses instead of scaffold-homogeneous top-3.
                if not ff_salsa_hits.empty and len(ff_salsa_hits) > 3:
                    ff_salsa_hits = _scaffold_diverse_candidates(ff_salsa_hits, max_k=3)
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
                    # §FFFFFF: cache misses are batched into ONE score_molecules_target call so
                    # the expensive Boltz2 checkpoint load happens once, not N times per round.
                    _ff_screen: Dict[str, float] = {}  # smiles -> score (cached full or fast)
                    _ff_rows: Dict[str, Any] = {}      # smiles -> row for later lookup
                    _ff_misses: List[Tuple[str, Any]] = []  # (smiles, row) for cache-miss molecules
                    for _, _ff_row in ff_salsa_hits.iterrows():
                        _ff_smiles = _ff_row['product_smiles']
                        _ff_canon = get_canonical_smiles(_ff_smiles)
                        _ff_key = (_ff_canon, protein)
                        _ff_rows[_ff_smiles] = _ff_row
                        if _ff_key in boltz_cache:
                            _ff_screen[_ff_smiles] = boltz_cache[_ff_key]
                            bt.logging.debug(f"§FF §NN cache hit: {boltz_cache[_ff_key]:.4f}")
                        elif _ff_canon in _epoch_fast_cache:
                            # §GGGGGG: molecule was fast-screened in an earlier §FF or §MM
                            # round this epoch — reuse the fast score, saving one batch slot.
                            _ff_screen[_ff_smiles] = _epoch_fast_cache[_ff_canon]
                            bt.logging.debug(
                                f"§FF §NN §GGGGGG fast-cache hit: {_epoch_fast_cache[_ff_canon]:.4f}"
                            )
                        else:
                            _ff_misses.append((_ff_smiles, _ff_row))
                    # §FFFFFF: batch all cache-miss fast-screens into ONE Boltz call.
                    if _ff_misses:
                        try:
                            _curr_blk2 = await state['subtensor'].get_current_block()
                            _next_ep2 = ((_curr_blk2 // state['epoch_length']) + 1) * state['epoch_length']
                            if _next_ep2 - _curr_blk2 < 5:
                                bt.logging.info("§FF §NN: epoch ends in <5 blocks -- stopping.")
                            else:
                                _ff_batch_vmbu = {
                                    _uid: {"smiles": [_sm], "names": [_row.get('product_name', '')]}
                                    for _uid, (_sm, _row) in enumerate(_ff_misses)
                                }
                                _ff_batch_sd: Dict[str, Any] = {_uid: {} for _uid in _ff_batch_vmbu}
                                await asyncio.to_thread(
                                    wrapper.score_molecules_target,
                                    _ff_batch_vmbu, _ff_batch_sd, subnet_config, '0x' + '0' * 64, True,
                                )
                                for _uid, (_sm, _row) in enumerate(_ff_misses):
                                    _ff_screen[_sm] = wrapper.per_molecule_metric.get(_uid, {}).get(_sm, -math.inf)
                                bt.logging.info(
                                    f"§FF §NN §FFFFFF: batch fast-screened {len(_ff_misses)} "
                                    f"cache-miss molecules in 1 Boltz call"
                                )
                                # §GGGGGG: store fast-screen scores so §MM rounds can
                                # reuse them without re-running 50-step Boltz inference.
                                for _uid, (_sm, _row) in enumerate(_ff_misses):
                                    _fc = get_canonical_smiles(_sm)
                                    _epoch_fast_cache[_fc] = _ff_screen.get(_sm, -math.inf)
                        except Exception as _ff_es:
                            bt.logging.error(f"§FF §NN batch fast-screen error: {_ff_es}")
                            for _sm, _ in _ff_misses:
                                _ff_screen.setdefault(_sm, -math.inf)

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
                                    _ff_comps = wrapper.per_molecule_components.get(_ff_uid_f, {}).get(_ff_winner, {})
                                    _ff_apb = _ff_comps.get('affinity_probability_binary')
                                    _ff_apv = _ff_comps.get('affinity_pred_value')
                                    _ff_li = _ff_comps.get('ligand_iptm')
                                    _ff_cs = _ff_comps.get('confidence_score')
                                    # §IIIIIIIIII: prefer map (PSICHIC candidates), fall back to row (SAVI pool).
                                    _ff_ple = _psichic_le_map.get(_ff_w_canon)
                                    if _ff_ple is None:
                                        _ff_raw = _ff_w_row.get('combined_score')
                                        if isinstance(_ff_raw, (int, float)) and math.isfinite(float(_ff_raw)):
                                            _ff_ple = float(_ff_raw)
                                    _disk_cache_put(
                                        db_path, _ff_w_canon, protein, _ff_score,
                                        product_name=_ff_w_row.get('product_name'),
                                        apb=_ff_apb if isinstance(_ff_apb, (int, float)) else None,
                                        apv=_ff_apv if isinstance(_ff_apv, (int, float)) else None,
                                        ligand_iptm=_ff_li if isinstance(_ff_li, (int, float)) else None,
                                        boltz_le_std=_compute_le_std(_ff_comps),
                                        confidence_score=_ff_cs if isinstance(_ff_cs, (int, float)) else None,
                                        boltz_embedding=_emb_to_bytes(_ff_comps),
                                        psichic_le=_ff_ple,  # §IIIIIIIIII
                                    )
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

    # §BBBBBBBBBB: Include §CC warm-start molecule in §MM seed pool when it was NOT
    # re-scored this epoch.  The warm-start molecule (top SQLite-cached entry) may
    # have been evicted from global_candidate_pool by later PSICHIC chunks or
    # filtered out by scaffold-diversity selection — meaning it never appears in
    # all_scores or boltz_cache.  Adding it here lets §MM explore its chemical
    # neighbourhood via SALSA, potentially finding a better SAVI-2020 neighbor.
    try:
        _bbbbbbbbbb_ws = _disk_cache_get_best(db_path, protein)
        if _bbbbbbbbbb_ws is not None:
            _bbbbbbbbbb_sc, _bbbbbbbbbb_sm, _bbbbbbbbbb_pn = _bbbbbbbbbb_ws
            _bbbbbbbbbb_can = get_canonical_smiles(_bbbbbbbbbb_sm) or _bbbbbbbbbb_sm
            if _bbbbbbbbbb_can not in _mm_all_scored and math.isfinite(_bbbbbbbbbb_sc):
                _mm_all_scored[_bbbbbbbbbb_can] = _bbbbbbbbbb_sc
                bt.logging.info(
                    f"[§BBBBBBBBBB] Added §CC warm-start to §MM seed pool: "
                    f"{_bbbbbbbbbb_pn} (cached_score={_bbbbbbbbbb_sc:.4f})"
                )
    except Exception as _bbbbbbbbbb_err:
        bt.logging.debug(f"[§BBBBBBBBBB] Warm-start §MM seed injection (non-fatal): {_bbbbbbbbbb_err}")

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
    _mm_savi_pool = _hhhhhh_pool if _hhhhhh_pool is not None else state.get('savi_stream_pool')
    # §KKKKKK: Hardware-adaptive §MM max rounds.  BoltzWrapper (instantiated above)
    # already probed VRAM and patched config: §XXXXX sets num_subsampled_msa=4096 on
    # H100 (≥70 GiB); §AAA sets 2048 on A100 (≥38 GiB).  Reading back the patched
    # config value reuses that detection without a second GPU probe and stays consistent
    # with the VRAM tiers used throughout the wrapper.
    #
    # Budget analysis (first epoch, 1200 s trigger window, all cache misses):
    #   H100 (~25 s/mol): initial 125 s + §FF 75 s → 1000 s §MM budget → ~17 rounds
    #                     Previous cap of 10 left ~7 rounds unused on H100.
    #   A100 (~45 s/mol): initial 225 s + §FF 100 s → 875 s §MM budget → ~9–10 rounds
    #                     Cap of 10 coincides with time ceiling on A100.
    #   RTX 3090 (~150 s/mol): time guard fires after 0–2 rounds — cap not relevant.
    #
    # Epochs 2+ (adaptive trigger, warm disk cache): §MM gets 1–3 rounds on all
    # hardware tiers due to the shorter trigger window; cap is not binding.
    _kkkkkk_msa = wrapper.config.get('num_subsampled_msa', 1024)
    if _kkkkkk_msa >= 4096:    # §XXXXX H100 tier (≥70 GiB VRAM)
        _mm_max_rounds = 20
        bt.logging.info("[§KKKKKK] H100 tier detected → _mm_max_rounds=20")
    else:
        _mm_max_rounds = 10    # A100 / RTX 3090 / default — time guard is active limit
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
                    5,   # top_k — §NNNN: wider net for scaffold-diversity selection below
                    _hhhhhh_score_col,  # §HHHHHH: surrogate-blended or 'combined_score' fallback
                    'product_smiles',
                    'product_name',
                    _salsa_operator_weights(_mm_seed_smiles),  # §ZZZZZ
                )
                # §NNNN: scaffold-diverse selection — each §MM fast-screen slot tests
                # a different chemical family, maximising coverage per GPU budget.
                if not _mm_salsa_hits.empty and len(_mm_salsa_hits) > 3:
                    _mm_salsa_hits = _scaffold_diverse_candidates(_mm_salsa_hits, max_k=3)
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
            # §FFFFFF: cache misses are batched into ONE score_molecules_target call so the
            # expensive Boltz2 checkpoint load is paid once per round instead of N times.
            _mm_screen: Dict[str, float] = {}  # smiles -> score
            _mm_row_map: Dict[str, Any] = {}   # smiles -> row
            _mm_misses: List[Tuple[str, Any]] = []  # (smiles, row) for cache-miss molecules
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
                elif _mm_canon in _epoch_fast_cache:
                    # §GGGGGG: molecule was already fast-screened this epoch (in §FF or
                    # an earlier §MM round) — reuse the cached fast score.
                    _mm_screen[_mm_smiles] = _epoch_fast_cache[_mm_canon]
                    bt.logging.debug(
                        f"§MM §NN §GGGGGG fast-cache hit: {_epoch_fast_cache[_mm_canon]:.4f}"
                    )
                else:
                    _mm_misses.append((_mm_smiles, _mm_row))
            # §FFFFFF: batch all cache-miss fast-screens into ONE Boltz call.
            if _mm_misses and not _mm_stop:
                try:
                    _mm_blk2 = await state['subtensor'].get_current_block()
                    _mm_ep2 = ((_mm_blk2 // state['epoch_length']) + 1) * state['epoch_length']
                    if _mm_ep2 - _mm_blk2 < 5:
                        bt.logging.info("§MM §NN: epoch ends in <5 blocks — stopping.")
                        _mm_stop = True
                    else:
                        _mm_batch_vmbu = {
                            _uid: {"smiles": [_sm], "names": [_row.get('product_name', '')]}
                            for _uid, (_sm, _row) in enumerate(_mm_misses)
                        }
                        _mm_batch_sd: Dict[str, Any] = {_uid: {} for _uid in _mm_batch_vmbu}
                        await asyncio.to_thread(
                            wrapper.score_molecules_target,
                            _mm_batch_vmbu, _mm_batch_sd, subnet_config, '0x' + '0' * 64, True,
                        )
                        for _uid, (_sm, _row) in enumerate(_mm_misses):
                            _mm_screen[_sm] = wrapper.per_molecule_metric.get(_uid, {}).get(_sm, -math.inf)
                        bt.logging.info(
                            f"§MM §NN §FFFFFF [{_mm_round_idx + 1}/{_mm_max_rounds}]: "
                            f"batch fast-screened {len(_mm_misses)} cache-miss molecules in 1 Boltz call"
                        )
                        # §GGGGGG: populate fast-cache so later §MM rounds skip re-screening
                        # molecules already evaluated in this pass.
                        for _uid, (_sm, _row) in enumerate(_mm_misses):
                            _fc = get_canonical_smiles(_sm)
                            _epoch_fast_cache[_fc] = _mm_screen.get(_sm, -math.inf)
                except Exception as _mm_es:
                    bt.logging.error(f"§MM §NN batch fast-screen error: {_mm_es}")
                    for _sm, _ in _mm_misses:
                        _mm_screen.setdefault(_sm, -math.inf)

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
                            _mm_comps = wrapper.per_molecule_components.get(_mm_uid_f, {}).get(_mm_round_winner, {})
                            _mm_apb = _mm_comps.get('affinity_probability_binary')
                            _mm_apv = _mm_comps.get('affinity_pred_value')
                            _mm_li = _mm_comps.get('ligand_iptm')
                            _mm_cs = _mm_comps.get('confidence_score')
                            # §IIIIIIIIII: prefer map (PSICHIC candidates), fall back to row (SAVI pool).
                            _mm_ple = _psichic_le_map.get(_mm_w_canon)
                            if _mm_ple is None:
                                _mm_raw = _mm_w_row.get('combined_score')
                                if isinstance(_mm_raw, (int, float)) and math.isfinite(float(_mm_raw)):
                                    _mm_ple = float(_mm_raw)
                            _disk_cache_put(
                                db_path, _mm_w_canon, protein, _mm_score,
                                product_name=_mm_w_row.get('product_name'),
                                apb=_mm_apb if isinstance(_mm_apb, (int, float)) else None,
                                apv=_mm_apv if isinstance(_mm_apv, (int, float)) else None,
                                ligand_iptm=_mm_li if isinstance(_mm_li, (int, float)) else None,
                                boltz_le_std=_compute_le_std(_mm_comps),
                                confidence_score=_mm_cs if isinstance(_mm_cs, (int, float)) else None,
                                boltz_embedding=_emb_to_bytes(_mm_comps),
                                psichic_le=_mm_ple,  # §IIIIIIIIII
                            )
                            if wrapper.last_inference_duration > 0:
                                state['boltz_time_per_mol'] = wrapper.last_inference_duration
                                _save_miner_state(db_path, 'boltz_time_per_mol',
                                                  wrapper.last_inference_duration)
                            bt.logging.info(
                                f"§MM §NN [{_mm_round_idx + 1}/{_mm_max_rounds}] full-scored winner: "
                                f"{_mm_w_row.get('product_name', '?')} boltz={_mm_score:.4f} "
                                f"(screened {len(_mm_screen)} hits)"
                            )
                            # §IIIIII: Online surrogate refresh — retrain the dual surrogate
                            # immediately after each new §MM full-score so subsequent rounds
                            # benefit from the freshest possible Boltz-calibrated signal.
                            # Only updates _mm_savi_pool when the RF tier is active (≥100 cache
                            # points); Ridge surrogates ignore additional points almost entirely.
                            try:
                                _ii_dual = fit_dual_surrogate(db_path, protein)
                                if _ii_dual is not None:
                                    _ii_src = (
                                        _mm_savi_pool
                                        if _mm_savi_pool is not None
                                        else state.get('savi_stream_pool')
                                    )
                                    if _ii_src is not None and not _ii_src.empty:
                                        _ii_pool = augment_pool_with_surrogate_blend(
                                            _ii_src, _ii_dual
                                        )
                                        if 'surrogate_salsa_score' in _ii_pool.columns:
                                            _mm_savi_pool = _ii_pool
                                            _hhhhhh_score_col = 'surrogate_salsa_score'
                                            bt.logging.debug(
                                                f"[§IIIIII] Surrogate refreshed after §MM "
                                                f"round {_mm_round_idx + 1} — pool re-blended."
                                            )
                            except Exception as _ii_exc:
                                bt.logging.debug(
                                    f"[§IIIIII] Surrogate refresh (non-fatal): {_ii_exc}"
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
                    # §CCCCCCCCCC: Before stopping, try the top surrogate-scored SAVI
                    # molecule not yet used as an §MM seed.  When all Boltz-scored
                    # candidates are exhausted, the surrogate-blended pool can nominate a
                    # fresh chemical basin to explore without an extra Boltz call upfront.
                    _cccccccccc_seed = None
                    try:
                        if _mm_savi_pool is not None and not _mm_savi_pool.empty:
                            _cc_col = (
                                _hhhhhh_score_col
                                if _hhhhhh_score_col in _mm_savi_pool.columns
                                else 'combined_score'
                            )
                            if _cc_col in _mm_savi_pool.columns:
                                for _, _cc_row in _mm_savi_pool.nlargest(50, _cc_col).iterrows():
                                    _cc_smi = _cc_row.get('product_smiles', '')
                                    _cc_can = get_canonical_smiles(_cc_smi) or _cc_smi
                                    if _cc_can and _cc_can not in _mm_tried_seeds:
                                        _ok2, _ = is_boltz_safe_smiles(_cc_can)
                                        if _ok2:
                                            _cccccccccc_seed = _cc_can
                                            bt.logging.info(
                                                f"[§CCCCCCCCCC] Surrogate-pool basin-hop: "
                                                f"{_cc_row.get('product_name', '?')} "
                                                f"({_cc_col}="
                                                f"{_cc_row.get(_cc_col, float('nan')):.4f}) "
                                                f"— Boltz-seed pool exhausted, extending §MM."
                                            )
                                            break
                    except Exception as _cc_err:
                        bt.logging.debug(
                            f"[§CCCCCCCCCC] Surrogate-pool fallback (non-fatal): {_cc_err}"
                        )
                    if _cccccccccc_seed is not None:
                        _mm_seed_smiles = _cccccccccc_seed
                        continue  # start next §MM round from surrogate-nominated basin
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
    # binder without any PSICHIC cost.
    # §NNNNNN: uses get_cached_pool_fps (§MMMMMM FP cache) and §NN two-phase
    # screening — batch fast-screen all tautomer SAVI neighbours in one Boltz
    # call, then full-score only the winner.  Reduces §XX from up to 6 full
    # Boltz calls to 1 fast-batch + 1 full call.  Time guard lowered from
    # +60 s to +30 s (fast-screen is cheap; only 1 full call is guaranteed).
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
            if (_xx_ep0 - _xx_blk0) * 12 > _xx_t_mol + 30:
                from rdkit.Chem.MolStandardize import rdMolStandardize as _rdMSt
                from utils.salsa import nearest_pool_molecules, get_cached_pool_fps

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
                            f"§XX §NNNNNN: {len(_xx_novel)} novel tautomers of epoch best — "
                            f"mapping to SAVI-2020 neighbours"
                        )
                        # §NNNNNN: reuse FP cache from §MM (§MMMMMM) — saves ~2-4 s.
                        _xx_valid_pool, _xx_pool_fps = get_cached_pool_fps(_xx_savi_pool)
                        _xx_seen_neighbours: set = set()

                        # Phase 0: collect up to 6 unique SAVI neighbours.
                        _xx_candidates: list = []  # [(smiles, pname, row), ...]
                        for _xx_t_smi in _xx_novel[:6]:
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
                            _xx_candidates.append((_xx_n_smi, _xx_n_pname, _xx_n_row))

                        if _xx_candidates:
                            # Phase 1: check caches; batch fast-screen all misses.
                            _xx_screen: Dict[str, float] = {}
                            _xx_misses: list = []  # [(smi, pname, row), ...]
                            for _xx_c_smi, _xx_c_pname, _xx_c_row in _xx_candidates:
                                _xx_c_canon = get_canonical_smiles(_xx_c_smi)
                                _xx_c_key = (_xx_c_canon, protein)
                                if _xx_c_key in boltz_cache:
                                    _xx_screen[_xx_c_smi] = boltz_cache[_xx_c_key]
                                    bt.logging.debug(
                                        f"§XX §NNNNNN cache hit: {boltz_cache[_xx_c_key]:.4f}"
                                    )
                                elif _xx_c_canon in _epoch_fast_cache:
                                    # §GGGGGG: reuse fast score from §FF/§MM this epoch.
                                    _xx_screen[_xx_c_smi] = _epoch_fast_cache[_xx_c_canon]
                                    bt.logging.debug(
                                        f"§XX §NNNNNN §GGGGGG fast-cache hit: "
                                        f"{_epoch_fast_cache[_xx_c_canon]:.4f}"
                                    )
                                else:
                                    _xx_misses.append((_xx_c_smi, _xx_c_pname, _xx_c_row))

                            if _xx_misses:
                                try:
                                    _xx_blk1 = await state['subtensor'].get_current_block()
                                    _xx_ep1 = (
                                        (_xx_blk1 // state['epoch_length']) + 1
                                    ) * state['epoch_length']
                                    if (_xx_ep1 - _xx_blk1) * 12 >= _xx_t_mol + 30:
                                        # §FFFFFF: one batched fast-screen call for all misses.
                                        _xx_batch_vmbu = {
                                            _uid: {"smiles": [_sm], "names": [_pn]}
                                            for _uid, (_sm, _pn, _rw) in enumerate(_xx_misses)
                                        }
                                        _xx_batch_sd: Dict[str, Any] = {
                                            _uid: {} for _uid in _xx_batch_vmbu
                                        }
                                        await asyncio.to_thread(
                                            wrapper.score_molecules_target,
                                            _xx_batch_vmbu, _xx_batch_sd, subnet_config,
                                            '0x' + '0' * 64, True,
                                        )
                                        for _uid, (_sm, _pn, _rw) in enumerate(_xx_misses):
                                            _sc = wrapper.per_molecule_metric.get(
                                                _uid, {}
                                            ).get(_sm, -math.inf)
                                            _xx_screen[_sm] = _sc
                                            _epoch_fast_cache[get_canonical_smiles(_sm)] = _sc
                                        bt.logging.info(
                                            f"§XX §NNNNNN §FFFFFF: batch fast-screened "
                                            f"{len(_xx_misses)} tautomer neighbours "
                                            f"in 1 Boltz call"
                                        )
                                except Exception as _xx_fe:
                                    bt.logging.error(
                                        f"§XX §NNNNNN fast-screen error: {_xx_fe}"
                                    )
                                    for _sm, _, _ in _xx_misses:
                                        _xx_screen.setdefault(_sm, -math.inf)

                            # Phase 2: full-score only the best fast-screened candidate.
                            _xx_winner = max(
                                (_s for _s, _v in _xx_screen.items() if math.isfinite(_v)),
                                key=lambda _s: _xx_screen[_s],
                                default=None,
                            )
                            if _xx_winner is not None:
                                _xx_w_canon = get_canonical_smiles(_xx_winner)
                                _xx_w_key = (_xx_w_canon, protein)
                                _xx_w_pname = next(
                                    (pn for sm, pn, _ in _xx_candidates if sm == _xx_winner),
                                    '',
                                )
                                if _xx_w_key in boltz_cache:
                                    _xx_w_score = boltz_cache[_xx_w_key]
                                else:
                                    _xx_disk_s = _disk_cache_get(
                                        db_path, _xx_w_canon, protein
                                    )
                                    if _xx_disk_s is not None:
                                        boltz_cache[_xx_w_key] = _xx_disk_s
                                        _xx_w_score = _xx_disk_s
                                    else:
                                        _xx_uid = 0
                                        _xx_vmbu = {
                                            _xx_uid: {
                                                "smiles": [_xx_winner],
                                                "names": [_xx_w_pname],
                                            }
                                        }
                                        _xx_sd: Dict[str, Any] = {_xx_uid: {}}
                                        try:
                                            await asyncio.to_thread(
                                                wrapper.score_molecules_target,
                                                _xx_vmbu, _xx_sd, subnet_config,
                                                '0x' + '0' * 64,
                                            )
                                            _xx_w_score = wrapper.per_molecule_metric.get(
                                                _xx_uid, {}
                                            ).get(_xx_winner, -math.inf)
                                            boltz_cache[_xx_w_key] = _xx_w_score
                                            _xx_comps = wrapper.per_molecule_components.get(
                                                _xx_uid, {}
                                            ).get(_xx_winner, {})
                                            _xx_apb = _xx_comps.get(
                                                'affinity_probability_binary'
                                            )
                                            _xx_apv = _xx_comps.get('affinity_pred_value')
                                            _xx_li = _xx_comps.get('ligand_iptm')
                                            _xx_cs = _xx_comps.get('confidence_score')
                                            _disk_cache_put(
                                                db_path, _xx_w_canon, protein, _xx_w_score,
                                                product_name=_xx_w_pname or None,
                                                apb=_xx_apb if isinstance(
                                                    _xx_apb, (int, float)
                                                ) else None,
                                                apv=_xx_apv if isinstance(
                                                    _xx_apv, (int, float)
                                                ) else None,
                                                ligand_iptm=_xx_li if isinstance(
                                                    _xx_li, (int, float)
                                                ) else None,
                                                boltz_le_std=_compute_le_std(_xx_comps),
                                                confidence_score=_xx_cs if isinstance(
                                                    _xx_cs, (int, float)
                                                ) else None,
                                                boltz_embedding=_emb_to_bytes(_xx_comps),
                                                psichic_le=None,  # §IIIIIIIIII: tautomer
                                            )
                                            if wrapper.last_inference_duration > 0:
                                                state['boltz_time_per_mol'] = (
                                                    wrapper.last_inference_duration
                                                )
                                                _save_miner_state(
                                                    db_path, 'boltz_time_per_mol',
                                                    wrapper.last_inference_duration,
                                                )
                                            bt.logging.info(
                                                f"§XX §NNNNNN: tautomer SAVI winner "
                                                f"{_xx_w_pname!r} boltz={_xx_w_score:.4f}"
                                            )
                                        except Exception as _xx_be:
                                            bt.logging.error(
                                                f"§XX §NNNNNN Boltz error: {_xx_be}"
                                            )
                                            _xx_w_score = -math.inf

                                all_scores[_xx_w_canon] = _xx_w_score
                                if math.isfinite(_xx_w_score) and _xx_w_score > _xx_best_epoch:
                                    _xx_prev_epoch = _xx_best_epoch
                                    _xx_best_epoch = _xx_w_score
                                    if _xx_w_pname:
                                        _xx_orig = state['candidate_product'].split(',')
                                        state['candidate_product'] = ','.join(
                                            [_xx_w_pname]
                                            + [n for n in _xx_orig if n != _xx_w_pname]
                                        )
                                        bt.logging.info(
                                            f"§XX §NNNNNN: new epoch best from tautomer "
                                            f"search — {_xx_w_pname} "
                                            f"(boltz={_xx_w_score:.4f} > "
                                            f"prev={_xx_prev_epoch:.4f})"
                                        )
        except Exception as _xx_err:
            bt.logging.warning(f"§XX tautomer search failed (non-fatal): {_xx_err}")

    # §TTTTTT — Extended Tautomer Search for 2nd/3rd Epoch Best Molecules.
    # §XX above enumerates tautomers of the single epoch-best molecule.  §TTTTTT
    # extends this to the 2nd and 3rd best molecules in all_scores (sorted by Boltz
    # LE), subject to a per-seed time guard.  On hardware with ample GPU budget
    # (H100: ~25 s/mol, up to 20 §MM rounds), distinct scaffolds often survive from
    # §QQ/§VV basin-hops; their tautomers represent genuinely unexplored chemical
    # space.  Each seed uses the §NNNNNN two-phase screen (batch fast-screen → one
    # full Boltz call for the winner), reusing the §MMMMMM FP cache.  Cross-seed
    # SAVI neighbour deduplication via a shared `_tt_seen` set prevents re-scoring
    # the same product name across both seeds.
    _tt_pool = state.get('savi_stream_pool')
    if (
        _tt_pool is not None
        and not _tt_pool.empty
        and len([v for v in all_scores.values() if math.isfinite(v)]) >= 2
    ):
        _tt_sorted_all = sorted(
            [(s, v) for s, v in all_scores.items() if math.isfinite(v)],
            key=lambda kv: kv[1], reverse=True,
        )
        # Indices 1 and 2 — index 0 is the §XX seed already processed above.
        _tt_extra_seeds = [s for s, _v in _tt_sorted_all[1:3]]
        if _tt_extra_seeds:
            try:
                from rdkit.Chem.MolStandardize import rdMolStandardize as _rdMSt_tt
                from utils.salsa import (
                    nearest_pool_molecules as _tt_nnm,
                    get_cached_pool_fps as _tt_gfps,
                )
                _tt_valid_pool, _tt_pool_fps = _tt_gfps(_tt_pool)
                _tt_seen: set = set()

                for _tt_seed_smi in _tt_extra_seeds:
                    # Per-seed time guard: abort before starting if < t_mol + 30 s remain.
                    try:
                        _tt_blk = await state['subtensor'].get_current_block()
                        _tt_ep  = (
                            (_tt_blk // state['epoch_length']) + 1
                        ) * state['epoch_length']
                        _tt_t_mol = state.get('boltz_time_per_mol', 150.0)
                        if (_tt_ep - _tt_blk) * 12 <= _tt_t_mol + 30:
                            bt.logging.info("§TTTTTT: time guard fired — stopping.")
                            break
                    except Exception:
                        break

                    _tt_mol = Chem.MolFromSmiles(_tt_seed_smi)
                    if _tt_mol is None:
                        continue
                    _tt_enumerator = _rdMSt_tt.TautomerEnumerator()
                    _tt_tautomers  = _tt_enumerator.Enumerate(_tt_mol)
                    _tt_seed_canon = Chem.MolToSmiles(_tt_mol)

                    _tt_novel: list = []
                    for _tt in _tt_tautomers:
                        if _tt is None:
                            continue
                        _tt_smi = Chem.MolToSmiles(_tt)
                        if _tt_smi == _tt_seed_canon:
                            continue
                        _tt_ok, _ = is_boltz_safe_smiles(_tt_smi)
                        if not _tt_ok:
                            continue
                        _tt_ha = get_heavy_atom_count(_tt_smi)
                        if _tt_ha is None or _tt_ha < 10 or _tt_ha > 35:
                            continue
                        _tt_novel.append(_tt_smi)

                    if not _tt_novel:
                        bt.logging.debug(
                            f"§TTTTTT: no novel tautomers for seed {_tt_seed_smi[:30]!r}"
                        )
                        continue

                    # Map tautomers to nearest SAVI-2020 neighbours (up to 6 probes).
                    _tt_cands: list = []
                    for _tt_tsmi in _tt_novel[:6]:
                        _tt_near = _tt_nnm(
                            _tt_tsmi, _tt_valid_pool, top_k=1, pool_fps=_tt_pool_fps
                        )
                        if _tt_near.empty:
                            continue
                        _tt_nr     = _tt_near.iloc[0]
                        _tt_nsmi   = _tt_nr['product_smiles']
                        _tt_npname = _tt_nr.get('product_name', '')
                        if _tt_npname in _tt_seen:
                            continue
                        _tt_seen.add(_tt_npname)
                        _tt_cands.append((_tt_nsmi, _tt_npname, _tt_nr))

                    if not _tt_cands:
                        continue

                    bt.logging.info(
                        f"§TTTTTT: {len(_tt_cands)} SAVI candidate(s) from "
                        f"tautomers of rank-2/3 seed — phase-1 fast-screen"
                    )

                    # Phase 1: cache hit check + batch fast-screen for misses.
                    _tt_screen: Dict[str, float] = {}
                    _tt_misses: list = []
                    for _tt_sm, _tt_pn, _tt_rw in _tt_cands:
                        _tt_can = get_canonical_smiles(_tt_sm)
                        _tt_key = (_tt_can, protein)
                        if _tt_key in boltz_cache:
                            _tt_screen[_tt_sm] = boltz_cache[_tt_key]
                            bt.logging.debug(
                                f"§TTTTTT cache hit: {boltz_cache[_tt_key]:.4f}"
                            )
                        elif _tt_can in _epoch_fast_cache:
                            _tt_screen[_tt_sm] = _epoch_fast_cache[_tt_can]
                            bt.logging.debug(
                                f"§TTTTTT fast-cache hit: {_epoch_fast_cache[_tt_can]:.4f}"
                            )
                        else:
                            _tt_misses.append((_tt_sm, _tt_pn, _tt_rw))

                    if _tt_misses:
                        try:
                            _tt_blk2 = await state['subtensor'].get_current_block()
                            _tt_ep2  = (
                                (_tt_blk2 // state['epoch_length']) + 1
                            ) * state['epoch_length']
                            if (_tt_ep2 - _tt_blk2) * 12 >= _tt_t_mol + 30:
                                _tt_bvmbu = {
                                    _uid: {"smiles": [_sm], "names": [_pn]}
                                    for _uid, (_sm, _pn, _rw) in enumerate(_tt_misses)
                                }
                                _tt_bsd: Dict[str, Any] = {
                                    _uid: {} for _uid in _tt_bvmbu
                                }
                                await asyncio.to_thread(
                                    wrapper.score_molecules_target,
                                    _tt_bvmbu, _tt_bsd, subnet_config,
                                    '0x' + '0' * 64, True,  # fast=True
                                )
                                for _uid, (_sm, _pn, _rw) in enumerate(_tt_misses):
                                    _sc = wrapper.per_molecule_metric.get(
                                        _uid, {}
                                    ).get(_sm, -math.inf)
                                    _tt_screen[_sm] = _sc
                                    _epoch_fast_cache[get_canonical_smiles(_sm)] = _sc
                                bt.logging.info(
                                    f"§TTTTTT §FFFFFF: batch fast-screened "
                                    f"{len(_tt_misses)} tautomer neighbour(s)"
                                )
                        except Exception as _tt_fe:
                            bt.logging.error(f"§TTTTTT fast-screen error: {_tt_fe}")
                            for _sm, _, _ in _tt_misses:
                                _tt_screen.setdefault(_sm, -math.inf)

                    # Phase 2: full Boltz call for the best fast-screened candidate.
                    _tt_winner_sm = max(
                        (_s for _s, _v in _tt_screen.items() if math.isfinite(_v)),
                        key=lambda _s: _tt_screen[_s],
                        default=None,
                    )
                    if _tt_winner_sm is None:
                        continue
                    _tt_w_can   = get_canonical_smiles(_tt_winner_sm)
                    _tt_w_key   = (_tt_w_can, protein)
                    _tt_w_pname = next(
                        (pn for sm, pn, _ in _tt_cands if sm == _tt_winner_sm), ''
                    )

                    if _tt_w_key in boltz_cache:
                        _tt_w_score = boltz_cache[_tt_w_key]
                    else:
                        _tt_disk = _disk_cache_get(db_path, _tt_w_can, protein)
                        if _tt_disk is not None:
                            boltz_cache[_tt_w_key] = _tt_disk
                            _tt_w_score = _tt_disk
                        else:
                            try:
                                _tt_blk3 = await state['subtensor'].get_current_block()
                                _tt_ep3  = (
                                    (_tt_blk3 // state['epoch_length']) + 1
                                ) * state['epoch_length']
                                if (_tt_ep3 - _tt_blk3) * 12 < _tt_t_mol + 30:
                                    bt.logging.info(
                                        "§TTTTTT: time guard before full-score — stopping."
                                    )
                                    break
                            except Exception:
                                break

                            _tt_uid  = 0
                            _tt_vmbu = {
                                _tt_uid: {
                                    "smiles": [_tt_winner_sm],
                                    "names":  [_tt_w_pname],
                                }
                            }
                            _tt_vsd: Dict[str, Any] = {_tt_uid: {}}
                            try:
                                await asyncio.to_thread(
                                    wrapper.score_molecules_target,
                                    _tt_vmbu, _tt_vsd, subnet_config,
                                    '0x' + '0' * 64,
                                )
                                _tt_w_score = wrapper.per_molecule_metric.get(
                                    _tt_uid, {}
                                ).get(_tt_winner_sm, -math.inf)
                                boltz_cache[_tt_w_key] = _tt_w_score
                                _tt_comps = wrapper.per_molecule_components.get(
                                    _tt_uid, {}
                                ).get(_tt_winner_sm, {})
                                _tt_apb = _tt_comps.get('affinity_probability_binary')
                                _tt_apv = _tt_comps.get('affinity_pred_value')
                                _tt_li  = _tt_comps.get('ligand_iptm')
                                _tt_cs  = _tt_comps.get('confidence_score')
                                _disk_cache_put(
                                    db_path, _tt_w_can, protein, _tt_w_score,
                                    product_name=_tt_w_pname or None,
                                    apb=_tt_apb if isinstance(
                                        _tt_apb, (int, float)
                                    ) else None,
                                    apv=_tt_apv if isinstance(
                                        _tt_apv, (int, float)
                                    ) else None,
                                    ligand_iptm=_tt_li if isinstance(
                                        _tt_li, (int, float)
                                    ) else None,
                                    boltz_le_std=_compute_le_std(_tt_comps),
                                    confidence_score=_tt_cs if isinstance(
                                        _tt_cs, (int, float)
                                    ) else None,
                                    boltz_embedding=_emb_to_bytes(_tt_comps),
                                    psichic_le=None,  # §IIIIIIIIII: tautomer
                                )
                                if wrapper.last_inference_duration > 0:
                                    state['boltz_time_per_mol'] = (
                                        wrapper.last_inference_duration
                                    )
                                    _save_miner_state(
                                        db_path,
                                        'boltz_time_per_mol',
                                        wrapper.last_inference_duration,
                                    )
                                bt.logging.info(
                                    f"§TTTTTT: full Boltz — "
                                    f"{_tt_w_pname!r} score={_tt_w_score:.4f}"
                                )
                            except Exception as _tt_be:
                                bt.logging.error(f"§TTTTTT Boltz error: {_tt_be}")
                                _tt_w_score = -math.inf

                    # Update all_scores and promote to pos-0 if new epoch best.
                    if math.isfinite(_tt_w_score):
                        _tt_epoch_best = max(
                            (v for v in all_scores.values() if math.isfinite(v)),
                            default=-math.inf,
                        )
                        all_scores[_tt_w_can] = _tt_w_score
                        if _tt_w_score > _tt_epoch_best and _tt_w_pname:
                            _tt_orig = state['candidate_product'].split(',')
                            state['candidate_product'] = ','.join(
                                [_tt_w_pname]
                                + [n for n in _tt_orig if n != _tt_w_pname]
                            )
                            bt.logging.info(
                                f"§TTTTTT: new epoch best from rank-2/3 "
                                f"tautomer search — {_tt_w_pname} "
                                f"(boltz={_tt_w_score:.4f} > "
                                f"prev={_tt_epoch_best:.4f})"
                            )

            except Exception as _tttttt_err:
                bt.logging.warning(
                    f"§TTTTTT tautomer search failed (non-fatal): {_tttttt_err}"
                )

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
                    # §ZZZZZZZZ: Skip extra seeds when the cached inter-seed std is
                    # already very low (< 0.003) — the molecule is diffusion-stable
                    # and re-running seeds 42/123 would just confirm what we know.
                    # Saves 1–2 Boltz call budgets for §MM/§TTTTTT on epoch 3+.
                    _zz_extra_seeds = list(_ww_extra_seeds)
                    try:
                        _zz_can = get_canonical_smiles(_ww_sm) or _ww_sm
                        with sqlite3.connect(db_path) as _zz_conn:
                            _zz_row = _zz_conn.execute(
                                "SELECT boltz_ww_std FROM boltz_cache "
                                "WHERE smiles=? AND protein=?",
                                (_zz_can, protein),
                            ).fetchone()
                        if _zz_row and _zz_row[0] is not None and float(_zz_row[0]) < 0.003:
                            _zz_extra_seeds = []
                            bt.logging.info(
                                f"[§ZZZZZZZZ] {_ww_pname}: cached ww_std={_zz_row[0]:.4f}"
                                " < 0.003 — skipping extra seeds (diffusion-stable)."
                            )
                    except Exception:
                        pass
                    for _ww_seed in _zz_extra_seeds:
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
                    # §XXXXXXXX: Persist inter-seed std to SQLite for surrogate
                    # confidence weighting.  Only stored when ≥2 seeds returned
                    # finite scores so the std is meaningful.  Uses UPDATE (not
                    # INSERT OR REPLACE) to preserve all other columns unchanged.
                    if len(_ww_mol_scores) >= 2:
                        _xx_ww_std = float(np.std(_ww_mol_scores, ddof=0))
                        _xx_can = get_canonical_smiles(_ww_sm) or _ww_sm
                        try:
                            with sqlite3.connect(db_path) as _xx_conn:
                                _xx_conn.execute(
                                    "UPDATE boltz_cache SET boltz_ww_std=? "
                                    "WHERE smiles=? AND protein=?",
                                    (_xx_ww_std, _xx_can, protein),
                                )
                            bt.logging.debug(
                                f"[§XXXXXXXX] {_ww_pname}: ww_std={_xx_ww_std:.4f} "
                                f"({len(_ww_mol_scores)} seeds)"
                            )
                        except Exception as _xx_e:
                            bt.logging.debug(
                                f"[§XXXXXXXX] ww_std persist failed (non-fatal): {_xx_e}"
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

    # §UUUU — Antitarget Boltz Selectivity Scoring.
    # PSICHIC already penalises antitarget binders in the streaming loop, but the
    # Boltz prescoring only evaluates the TARGET protein.  A top candidate could
    # score well on the target while also binding the antitarget strongly — a
    # disadvantage discovered only at validation time.  §UUUU runs a fast Boltz
    # inference on the weekly antitarget for the top-2 Boltz candidates and adjusts
    # submission ordering using:
    #   selectivity_score = target_LE − antitarget_weight × antitarget_LE
    # Time guard: only fires when ≥ 2 fast-Boltz calls + 60 s of runway remain.
    _uuuu_antitargets = state.get('current_challenge_antitargets', [])
    _uuuu_valid = {s: v for s, v in all_scores.items() if math.isfinite(v)}
    if _uuuu_antitargets and len(_uuuu_valid) >= 2:
        try:
            _uuuu_blk = await state['subtensor'].get_current_block()
            _uuuu_ep = ((_uuuu_blk // state['epoch_length']) + 1) * state['epoch_length']
            _uuuu_remaining = (_uuuu_ep - _uuuu_blk) * 12
            # Fast Boltz ≈ 1/3 of full inference time (50 vs 150 sampling steps).
            _uuuu_t_fast = state.get('boltz_time_per_mol', 150.0) / 3.0
            if _uuuu_remaining > _uuuu_t_fast * 2 + 60:
                _uuuu_at_protein = _uuuu_antitargets[0]
                bt.logging.info(
                    f"§UUUU: {_uuuu_remaining:.0f}s remaining — antitarget selectivity "
                    f"check on top-2 candidates (antitarget={_uuuu_at_protein})..."
                )
                # Build SMILES→product_name lookup (same pattern as §WW)
                _uuuu_name_lookup: Dict[str, str] = {}
                for _uuuu_frame in (candidates, state.get('global_candidate_pool'),
                                    state.get('savi_stream_pool')):
                    if _uuuu_frame is None or _uuuu_frame.empty:
                        continue
                    if 'product_smiles' not in _uuuu_frame.columns:
                        continue
                    if 'product_name' not in _uuuu_frame.columns:
                        continue
                    for _, _uuuu_r in _uuuu_frame.iterrows():
                        _uuuu_ps = str(_uuuu_r.get('product_smiles', ''))
                        _uuuu_pn = str(_uuuu_r.get('product_name', ''))
                        if _uuuu_ps and _uuuu_pn:
                            _uuuu_name_lookup.setdefault(_uuuu_ps, _uuuu_pn)
                            _uuuu_can_ps = get_canonical_smiles(_uuuu_ps)
                            if _uuuu_can_ps:
                                _uuuu_name_lookup.setdefault(_uuuu_can_ps, _uuuu_pn)

                # Subnet config for antitarget inference — same settings as target
                # but override weekly_target and clear binding_pocket (we have no
                # pocket data for the antitarget).
                _uuuu_at_config = {
                    **subnet_config,
                    'weekly_target': _uuuu_at_protein,
                    'binding_pocket': None,
                    'max_distance': None,
                    'force': False,
                }

                _uuuu_selectivity: Dict[str, float] = {}
                _uuuu_top2 = sorted(
                    _uuuu_valid.items(), key=lambda kv: kv[1], reverse=True
                )[:2]
                _uuuu_at_weight = getattr(state.get('config'), 'antitarget_weight', 0.9)
                _uuuu_stop = False

                for _uuuu_sm, _uuuu_target_le in _uuuu_top2:
                    if _uuuu_stop:
                        break
                    try:
                        _uuuu_blk2 = await state['subtensor'].get_current_block()
                        _uuuu_ep2 = (
                            (_uuuu_blk2 // state['epoch_length']) + 1
                        ) * state['epoch_length']
                        if (_uuuu_ep2 - _uuuu_blk2) * 12 < _uuuu_t_fast + 30:
                            bt.logging.info("§UUUU: time guard fired — stopping early.")
                            _uuuu_stop = True
                            break
                    except Exception:
                        pass

                    _uuuu_pname = (
                        _uuuu_name_lookup.get(_uuuu_sm)
                        or _uuuu_name_lookup.get(get_canonical_smiles(_uuuu_sm) or '', '')
                    )
                    _uuuu_uid = 0
                    _uuuu_vmbu = {
                        _uuuu_uid: {"smiles": [_uuuu_sm], "names": [_uuuu_pname or '']}
                    }
                    _uuuu_sd: Dict[str, Any] = {_uuuu_uid: {}}
                    try:
                        await asyncio.to_thread(
                            wrapper.score_molecules_target,
                            _uuuu_vmbu, _uuuu_sd, _uuuu_at_config,
                            '0x' + '0' * 64, True,  # fast=True
                        )
                        _uuuu_at_le = wrapper.per_molecule_metric.get(
                            _uuuu_uid, {}
                        ).get(_uuuu_sm)
                        if _uuuu_at_le is not None and math.isfinite(_uuuu_at_le):
                            _uuuu_sel = _uuuu_target_le - _uuuu_at_weight * _uuuu_at_le
                            _uuuu_selectivity[_uuuu_sm] = _uuuu_sel
                            bt.logging.info(
                                f"§UUUU {_uuuu_sm[:40]!r}: "
                                f"target_LE={_uuuu_target_le:.4f}, "
                                f"antitarget_LE={_uuuu_at_le:.4f} "
                                f"→ selectivity={_uuuu_sel:.4f}"
                            )
                        else:
                            _uuuu_selectivity[_uuuu_sm] = _uuuu_target_le
                            bt.logging.debug(
                                f"§UUUU: no valid antitarget score for "
                                f"{_uuuu_sm[:40]!r} — keeping target_LE."
                            )
                    except Exception as _uuuu_be:
                        bt.logging.error(f"§UUUU antitarget inference error: {_uuuu_be}")
                        _uuuu_selectivity[_uuuu_sm] = _uuuu_target_le

                # Reorder submission when both top-2 have selectivity estimates
                if len(_uuuu_selectivity) >= 2:
                    _uuuu_best_sm = max(_uuuu_selectivity, key=_uuuu_selectivity.get)
                    # §VVVV: target-LE priority guard.  The validator weights 100% on
                    # target Boltz LE (pure target score, no antitarget adjustment).
                    # When num_molecules_boltz=1 only pos-0 is scored, so swapping to a
                    # molecule with lower target_LE — even if it has better selectivity —
                    # would decrease our validator score.  Only allow the swap when the
                    # selectivity winner also has ≥ target_LE as the best target-only
                    # candidate (or when multiple molecules are scored, so selectivity
                    # matters for entropy bonus).
                    _uuuu_best_le = _uuuu_valid.get(_uuuu_best_sm, -math.inf)
                    _uuuu_top_le = _uuuu_top2[0][1]
                    _uuuu_n_boltz = subnet_config.get('num_molecules_boltz', 1)
                    if _uuuu_n_boltz <= 1 and _uuuu_best_le < _uuuu_top_le - 1e-6:
                        bt.logging.info(
                            f"§UUUU+§VVVV: selectivity winner target_LE={_uuuu_best_le:.4f} "
                            f"< top target_LE={_uuuu_top_le:.4f} — swap suppressed; "
                            "validator uses pure target LE (num_molecules_boltz=1)."
                        )
                    else:
                        _uuuu_best_pname = (
                            _uuuu_name_lookup.get(_uuuu_best_sm)
                            or _uuuu_name_lookup.get(
                                get_canonical_smiles(_uuuu_best_sm) or '', ''
                            )
                        )
                        if _uuuu_best_pname:
                            _uuuu_orig = (state.get('candidate_product') or '').split(',')
                            if _uuuu_orig and _uuuu_orig[0] != _uuuu_best_pname:
                                state['candidate_product'] = ','.join(
                                    [_uuuu_best_pname]
                                    + [n for n in _uuuu_orig if n != _uuuu_best_pname]
                                )
                                _uuuu_other_sel = next(
                                    (v for s, v in _uuuu_selectivity.items()
                                     if s != _uuuu_best_sm),
                                    -math.inf,
                                )
                                bt.logging.info(
                                    f"§UUUU: pos-0 swapped → {_uuuu_best_pname} "
                                    f"(selectivity={_uuuu_selectivity[_uuuu_best_sm]:.4f}) "
                                    f"over {_uuuu_orig[0]} "
                                    f"(selectivity={_uuuu_other_sel:.4f})"
                                )
                            else:
                                bt.logging.info(
                                    f"§UUUU: pos-0 confirmed ({_uuuu_best_pname}); "
                                    "selectivity ordering agrees with target-only ordering."
                                )
        except Exception as _uuuu_err:
            bt.logging.warning(
                f"§UUUU antitarget selectivity check failed (non-fatal): {_uuuu_err}"
            )

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
                # §CCCCCC: persist across restarts so the bias survives crashes/auto-updates.
                _db_yy = state.get('boltz_cache_db', BOLTZ_CACHE_DB)
                _save_miner_state_text(_db_yy, 'best_boltz_rxn_class', _yy_rxn)
                # §EEEEEE: also record the winning Boltz score for this class so the
                # per-class history drives top-K rank-weighted SAVI sampling next epoch.
                # Use the best finite score in all_scores (the score that caused this class
                # to win) as the representative score for this run.
                _yy_best = max(
                    (v for v in all_scores.values() if math.isfinite(v)), default=None
                )
                if _yy_best is not None and _yy_best > 0:
                    _save_rxn_class_scores(_db_yy, _yy_rxn, _yy_best)
                    state['rxn_class_weights'] = _load_rxn_class_weights(_db_yy)
                    _ee_top = sorted(
                        state['rxn_class_weights'],
                        key=state['rxn_class_weights'].get,
                        reverse=True,
                    )[:3]
                    bt.logging.debug(
                        f"[§EEEEEE] Updated rxn scores: {_yy_rxn}={_yy_best:.4f} | "
                        f"top-3 weights: {[(c, state['rxn_class_weights'][c]) for c in _ee_top]}"
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
        'startup_dual_surrogate': None,  # §YYYYYY: dual RF surrogate fitted at startup from GitHub cache
        'best_boltz_rxn_class': None,    # §YY: winning rxn class for SAVI streaming bias (persists across epochs)
        'rxn_class_weights': {},          # §EEEEEE: {rxn_class: weight} for top-K multi-class sampling bias
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

        # §WWWWW: SMILES from homologous prior-target proteins (populated before
        # cache cleanup so prior-protein rows are still readable).
        'cross_target_seeds': [],

        # §OOOOOO: Adaptive §TTTT fragment-slot quota (adapted at startup from
        # Boltz cache evidence; 1000 default until sufficient data exists).
        'tttt_fragment_quota': 1000,
    }

    # Ensure persistent Boltz cache DB exists
    _init_boltz_cache_db(state['boltz_cache_db'])

    # §WWWWW: harvest cross-target seeds from homologous prior-target proteins
    # BEFORE cleanup removes all non-current-protein entries.  On the first epoch
    # after a target rotation this gives SALSA confirmed binders from structurally
    # related targets (≥40% seq-identity) as warm seeds — a no-op when the target
    # has not rotated or no prior-target entries exist.
    state['cross_target_seeds'] = _cross_target_seeds_from_cache(
        state['boltz_cache_db'], config.weekly_target
    )
    if state['cross_target_seeds']:
        bt.logging.info(
            f"§WWWWW: {len(state['cross_target_seeds'])} cross-target seed(s) "
            f"saved from homologous prior-target(s)."
        )

    _cleanup_boltz_cache(state['boltz_cache_db'], keep_protein=config.weekly_target)
    bt.logging.info(f"Boltz persistent cache initialised: {state['boltz_cache_db']}")

    # §PPPPPP: Download and import Boltz cache export from the miner's GitHub repo.
    # boltz_score_cache.db is gitignored and ephemeral: every fresh container start
    # loses all scored molecules, making §BBBBB/§CCCCCC/§EEEEEE start cold for epoch 1.
    # §PPPPPP uploads a compact JSON export (top-500 entries + miner_state) at each
    # successful submission and re-imports it here so the surrogate, adaptive timing,
    # and reaction-class weights are warm from the very first epoch of any new session.
    # Runs BEFORE §BBBBB/§CCCCCC/§EEEEEE so those restores find populated miner_state
    # rows even on a fresh container with an otherwise empty SQLite DB.
    try:
        _pppppp_data = download_boltz_cache_export(config.weekly_target)
        if _pppppp_data:
            _pppppp_matched = _pppppp_data.get('_protein_matched', True)
            if _pppppp_matched:
                _pppppp_entries = _pppppp_data.get('entries', [])
                _pppppp_imported = 0
                if _pppppp_entries:
                    with sqlite3.connect(state['boltz_cache_db']) as _pp_conn:
                        for _e in _pppppp_entries:
                            try:
                                cur = _pp_conn.execute(
                                    "INSERT OR IGNORE INTO boltz_cache "
                                    "(smiles, protein, score, ts, product_name, "
                                    "affinity_prob_binary, affinity_pred_val, ligand_iptm, "
                                    "boltz_le_std, boltz_ww_std, confidence_score) "
                                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                    (
                                        _e['smiles'], config.weekly_target,
                                        _e['score'], 0,
                                        _e.get('product_name') or '',
                                        _e.get('apb'), _e.get('apv'), _e.get('ligand_iptm'),
                                        _e.get('le_std'), _e.get('ww_std'),
                                        _e.get('conf_score'),
                                    ),
                                )
                                if cur.rowcount:
                                    _pppppp_imported += 1
                            except Exception:
                                pass
                _pppppp_state = _pppppp_data.get('state', {})
                for _pk in ('boltz_time_per_mol', 'boltz_trigger_blocks'):
                    if _pk in _pppppp_state and _load_miner_state(state['boltz_cache_db'], _pk) is None:
                        _save_miner_state(state['boltz_cache_db'], _pk, float(_pppppp_state[_pk]))
                for _pk in ('best_boltz_rxn_class', 'rxn_class_scores_json'):
                    if _pk in _pppppp_state and _load_miner_state_text(state['boltz_cache_db'], _pk) is None:
                        _save_miner_state_text(state['boltz_cache_db'], _pk, str(_pppppp_state[_pk]))
                bt.logging.info(
                    f"[§PPPPPP] Imported {_pppppp_imported}/{len(_pppppp_entries)} cache entries "
                    f"from GitHub export (protein={config.weekly_target!r})."
                )
            # §RRRRRR: Import cross-target history regardless of protein match.
            # On fresh container + protein rotation, §WWWWW found nothing (empty SQLite).
            # History entries for prior proteins enable a second §WWWWW pass here so
            # SALSA gets Boltz-validated seeds from structurally related prior targets
            # even when the current target's own cache is missing.
            _rrrrrr_history = _pppppp_data.get('history', {})
            if _rrrrrr_history:
                import time as _rr_time
                _rr_ts = int(_rr_time.time())
                with sqlite3.connect(state['boltz_cache_db']) as _rr_conn:
                    for _rr_protein, _rr_entries in _rrrrrr_history.items():
                        for _e in _rr_entries:
                            try:
                                _rr_conn.execute(
                                    "INSERT OR IGNORE INTO boltz_cache "
                                    "(smiles, protein, score, ts, product_name, "
                                    "affinity_prob_binary, affinity_pred_val, ligand_iptm, "
                                    "boltz_le_std, boltz_ww_std, confidence_score) "
                                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                    (
                                        _e['smiles'], _rr_protein,
                                        _e['score'], _rr_ts,
                                        _e.get('product_name') or '',
                                        _e.get('apb'), _e.get('apv'), _e.get('ligand_iptm'),
                                        _e.get('le_std'), _e.get('ww_std'),
                                        _e.get('conf_score'),
                                    ),
                                )
                            except Exception:
                                pass
                # Re-run cross-target seeding now that history entries are in SQLite.
                _rrrrrr_seeds = _cross_target_seeds_from_cache(
                    state['boltz_cache_db'], config.weekly_target
                )
                _rrrrrr_new = [s for s in _rrrrrr_seeds if s not in state['cross_target_seeds']]
                if _rrrrrr_new:
                    state['cross_target_seeds'].extend(_rrrrrr_new)
                    bt.logging.info(
                        f"[§RRRRRR] {len(_rrrrrr_new)} cross-target seed(s) from "
                        f"GitHub history ({len(_rrrrrr_history)} prior protein(s))."
                    )
        else:
            bt.logging.debug("[§PPPPPP] No GitHub cache export found — cold start.")
    except Exception as _pppppp_err:
        bt.logging.warning(f"[§PPPPPP] Cache import failed (non-fatal): {_pppppp_err}")

    # §BBBBB: Restore adaptive timing from disk so fast-GPU miners (A100/H100) use
    # the correct boltz_trigger_blocks from epoch 1 after a process restart instead
    # of defaulting to 100 blocks (20 min) and wasting 12–16 min of PSICHIC streaming.
    _bbbbb_tpm = _load_miner_state(state['boltz_cache_db'], 'boltz_time_per_mol')
    _bbbbb_trg = _load_miner_state(state['boltz_cache_db'], 'boltz_trigger_blocks')
    if _bbbbb_tpm is not None and _bbbbb_tpm > 0:
        state['boltz_time_per_mol'] = _bbbbb_tpm
        bt.logging.info(
            f"[§BBBBB] Restored boltz_time_per_mol={_bbbbb_tpm:.1f}s from disk."
        )
    if _bbbbb_trg is not None and _bbbbb_trg >= 30:
        state['boltz_trigger_blocks'] = int(_bbbbb_trg)
        bt.logging.info(
            f"[§BBBBB] Restored boltz_trigger_blocks={int(_bbbbb_trg)} from disk."
        )

    # §CCCCCC: Restore winning reaction class so §YY SAVI streaming bias survives restarts.
    # Without this, every restart resets best_boltz_rxn_class=None and falls back to
    # uniform SAVI sampling for one full epoch, discarding the epoch-over-epoch learning
    # of which reaction template produces the best Boltz-validated binders.
    _cccccc_rxn = _load_miner_state_text(state['boltz_cache_db'], 'best_boltz_rxn_class')
    if _cccccc_rxn:
        state['best_boltz_rxn_class'] = _cccccc_rxn
        bt.logging.info(
            f"[§CCCCCC] Restored best_boltz_rxn_class={_cccccc_rxn!r} from disk — "
            f"SAVI streaming pre-biased toward this reaction class."
        )

    # §EEEEEE: Load per-class score history and compute rank-based sampling weights.
    # When history exists (epoch 2+), this replaces the §YY single-class 2× bias with
    # a richer 4×/2×/1.5×/1× gradient across the top-3 reaction classes.
    _eeeeee_weights = _load_rxn_class_weights(state['boltz_cache_db'])
    if _eeeeee_weights:
        state['rxn_class_weights'] = _eeeeee_weights
        _eeeeee_top = sorted(_eeeeee_weights, key=_eeeeee_weights.get, reverse=True)[:3]
        bt.logging.info(
            "[§EEEEEE] Restored rxn class weights from disk — "
            f"top-3: {_eeeeee_top} "
            f"(weights {[_eeeeee_weights[c] for c in _eeeeee_top]})"
        )

    # §OOOOOO: Adapt §TTTT fragment-slot quota using Boltz cache evidence.
    # Default: 1000/10000 slots for ≤18-HA molecules (§TTTT).
    # With ≥10 scored fragments AND ≥10 scored drug-like molecules in cache:
    #   avg_le_frag > avg_le_drug × 1.20 → quota=2500 (fragments outperform)
    #   avg_le_frag < avg_le_drug          → quota=500  (drug-like outperform)
    #   otherwise                           → quota=1000 (parity, keep default)
    # Requires ≥10 data points per bucket to avoid adapting on a handful of noisy
    # measurements.  Falls back to 1000 silently on insufficient data or any error.
    _oooooo_af, _oooooo_ad, _oooooo_nf, _oooooo_nd = _compute_ha_bucket_le(
        state['boltz_cache_db'], config.weekly_target
    )
    if (
        _oooooo_af is not None and _oooooo_ad is not None
        and _oooooo_nf >= 10 and _oooooo_nd >= 10
    ):
        if _oooooo_af > _oooooo_ad * 1.20:
            state['tttt_fragment_quota'] = 2500
            bt.logging.info(
                f"[§OOOOOO] Fragments outperform drug-like "
                f"(avg_LE {_oooooo_af:.4f} vs {_oooooo_ad:.4f}, "
                f"n={_oooooo_nf}/{_oooooo_nd}) → §TTTT quota=2500"
            )
        elif _oooooo_af < _oooooo_ad:
            state['tttt_fragment_quota'] = 500
            bt.logging.info(
                f"[§OOOOOO] Drug-like outperform fragments "
                f"(avg_LE {_oooooo_af:.4f} vs {_oooooo_ad:.4f}, "
                f"n={_oooooo_nf}/{_oooooo_nd}) → §TTTT quota=500"
            )
        else:
            state['tttt_fragment_quota'] = 1000
            bt.logging.info(
                f"[§OOOOOO] Fragment/drug-like parity "
                f"(avg_LE {_oooooo_af:.4f} vs {_oooooo_ad:.4f}) → §TTTT quota=1000"
            )
    else:
        state['tttt_fragment_quota'] = 1000  # insufficient cache data — keep default

    # §YYYYYY: Fit dual RF surrogate at startup from GitHub-imported cache so
    # the PSICHIC streaming loop can blend its Boltz-calibrated signal from the
    # very first SAVI chunk.  Only activates for RF models (≥100 cache points);
    # falls back to None on cold starts or when only Ridge quality is available
    # (augment_pool_with_surrogate_blend handles this guard internally).
    try:
        _yyyyyy_dual = fit_dual_surrogate(state['boltz_cache_db'], config.weekly_target)
        if _yyyyyy_dual is not None:
            state['startup_dual_surrogate'] = _yyyyyy_dual
            bt.logging.info(
                "[§YYYYYY] Startup dual surrogate fitted — PSICHIC streaming will "
                "blend 0.4×PSICHIC + 0.6×surrogate from chunk 1 (RF tier only)."
            )
    except Exception as _yyyyyy_err:
        bt.logging.debug(f"[§YYYYYY] Startup surrogate fit skipped: {_yyyyyy_err}")

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

        # §UUUU: Fetch MSA for antitarget proteins so antitarget Boltz scoring
        # has evolutionary context when §UUUU fires inside run_boltz_prescoring.
        for _uuuu_at_p in startup_proteins.get("antitargets", []):
            try:
                _uuuu_at_seq = get_sequence_from_protein_code(_uuuu_at_p)
                if _uuuu_at_seq:
                    ensure_msa(_uuuu_at_p, _uuuu_at_seq)
            except Exception as _uuuu_at_msa_err:
                bt.logging.warning(
                    f"[§UUUU] Antitarget MSA fetch failed for {_uuuu_at_p} "
                    f"(non-fatal): {_uuuu_at_msa_err}"
                )

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

                    # §UUUU: Ensure MSA exists for new antitarget proteins.
                    for _uuuu_new_at in new_proteins.get("antitargets", []):
                        try:
                            _uuuu_new_at_seq = get_sequence_from_protein_code(_uuuu_new_at)
                            if _uuuu_new_at_seq:
                                ensure_msa(_uuuu_new_at, _uuuu_new_at_seq)
                        except Exception as _uuuu_new_at_err:
                            bt.logging.warning(
                                f"[§UUUU] Antitarget MSA fetch failed for "
                                f"{_uuuu_new_at}: {_uuuu_new_at_err}"
                            )

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

                # §PPPP: Pre-seed savi_stream_pool with top-50 Boltz-validated
                # molecules from disk cache.  These act as attractor anchors in
                # Morgan fingerprint space: when SALSA maps any perturbation back
                # to the pool via Tanimoto, a prior Boltz winner that is the
                # nearest neighbour will be selected and used as the next SALSA
                # seed — directing hill-climbing toward validated binders from
                # the very first SALSA round.  The Boltz score is used directly
                # as combined_score so SALSA's score-based seed-advance logic
                # naturally promotes these anchors over random SAVI molecules.
                _pppp_rows = _disk_cache_get_candidates(
                    state['boltz_cache_db'], config.weekly_target, limit=50
                )
                if _pppp_rows:
                    try:
                        _pppp_df = pd.DataFrame(_pppp_rows)
                        _pppp_df['heavy_atoms'] = _pppp_df['product_smiles'].apply(
                            lambda s: get_heavy_atom_count(s) or 0
                        )
                        _pppp_df = _pppp_df[_pppp_df['heavy_atoms'] > 0].reset_index(
                            drop=True
                        )
                        if not _pppp_df.empty:
                            state['savi_stream_pool'] = _pppp_df
                            bt.logging.info(
                                f"[§PPPP] Pre-seeded savi_stream_pool with "
                                f"{len(_pppp_df)} Boltz-validated anchors "
                                f"from disk cache."
                            )
                    except Exception as _pppp_exc:
                        bt.logging.debug(
                            f"[§PPPP] Pool pre-seeding failed (non-fatal): "
                            f"{_pppp_exc}"
                        )

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
