"""
SALSA — Stochastic Approximate Ligand Scoring and Optimisation

Iterative hill-climbing over chemical space, constrained to molecules already
present in the SAVI-2020 streamed pool.  Starting from a high-PSICHIC seed,
SALSA generates bioisosteric perturbations and maps each back to the nearest
molecule in the streamed pool via Tanimoto similarity.  The resulting hits are
valid SAVI-2020 product names and can be added directly to the
global_candidate_pool for Boltz-2 pre-scoring.

Typical call from miner.py:

    hits_df = run_salsa_search(
        seed_smiles=state['global_candidate_pool'].iloc[0]['product_smiles'],
        savi_pool_df=state['savi_stream_pool'],
        rounds=3,
        n_perturb=60,
        top_k=5,
    )
    # hits_df rows are valid entries from savi_stream_pool; add to global_candidate_pool.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bioisosteric substitution table
# ---------------------------------------------------------------------------
_BIOISOSTERES: dict[int, list[int]] = {
    6:  [7, 8, 16],   # C  → N, O, S
    7:  [6, 8],       # N  → C, O
    8:  [6, 7, 16],   # O  → C, N, S
    17: [9, 35],      # Cl → F, Br
    35: [17, 9],      # Br → Cl, F
    9:  [17, 35],     # F  → Cl, Br
}


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------
def _morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    """Return Morgan fingerprint or None on failure."""
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def precompute_pool_fps(
    pool_df: pd.DataFrame,
    smiles_col: str = 'product_smiles',
) -> Tuple[pd.DataFrame, List]:
    """
    Pre-compute Morgan fingerprints for every molecule in *pool_df*.

    Returns a (valid_df, fps_list) tuple where:
    - valid_df  — rows of pool_df whose SMILES parsed successfully (same order as fps_list)
    - fps_list  — list of ExplicitBitVect fingerprints, one per valid_df row

    Call this once before a SALSA/GA run and pass the results to
    nearest_pool_molecules() via the pool_fps argument to avoid recomputing
    fingerprints on every query (typically 180+ queries per SALSA run over a
    5000-molecule pool — ~900k redundant FP computations without caching).
    """
    fps: List = []
    valid_indices: List[int] = []

    for idx, smiles in enumerate(pool_df[smiles_col]):
        mol = Chem.MolFromSmiles(smiles)
        fp = _morgan_fp(mol) if mol is not None else None
        if fp is not None:
            fps.append(fp)
            valid_indices.append(idx)

    valid_df = pool_df.iloc[valid_indices].reset_index(drop=True)
    return valid_df, fps


# ---------------------------------------------------------------------------
# Perturbation operators
# ---------------------------------------------------------------------------
def generate_perturbations(smiles: str, n_max: int = 100) -> List[str]:
    """
    Generate up to n_max unique canonical SMILES variants of *smiles* via
    bioisosteric atom substitution.

    Returns a list of valid canonical SMILES strings (excluding the input).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    seen = {Chem.MolToSmiles(mol)}
    results: List[str] = []

    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        for target_an in _BIOISOSTERES.get(an, []):
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(target_an)
            try:
                Chem.SanitizeMol(rw)
                canonical = Chem.MolToSmiles(rw.GetMol())
                if canonical not in seen:
                    seen.add(canonical)
                    results.append(canonical)
            except Exception:
                pass
            if len(results) >= n_max:
                return results

    return results


# ---------------------------------------------------------------------------
# Nearest-neighbour pool search
# ---------------------------------------------------------------------------
def nearest_pool_molecules(
    target_smiles: str,
    pool_df: pd.DataFrame,
    top_k: int = 1,
    smiles_col: str = 'product_smiles',
    pool_fps: Optional[List] = None,
) -> pd.DataFrame:
    """
    Return the *top_k* rows from *pool_df* most similar to *target_smiles*
    by Tanimoto coefficient on Morgan fingerprints (radius=2, 2048 bits).

    When *pool_fps* is provided (a pre-computed list of ExplicitBitVect objects
    aligned with *pool_df*'s rows, as returned by precompute_pool_fps()), the
    function skips per-call FP computation and uses BulkTanimotoSimilarity for
    a vectorised C++ similarity sweep — dramatically faster for large pools.

    Returns an empty DataFrame if *target_smiles* is invalid or pool is empty.
    """
    if pool_df is None or pool_df.empty:
        return pd.DataFrame()

    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        return pd.DataFrame()
    target_fp = _morgan_fp(target_mol)
    if target_fp is None:
        return pd.DataFrame()

    if pool_fps is not None:
        # Fast path: pre-computed FPs + vectorised BulkTanimotoSimilarity
        if not pool_fps:
            return pd.DataFrame()
        sims = DataStructs.BulkTanimotoSimilarity(target_fp, pool_fps)
    else:
        # Slow path: compute FPs on the fly (used when pool_fps not provided)
        sims = []
        for smiles in pool_df[smiles_col]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                sims.append(0.0)
                continue
            fp = _morgan_fp(mol)
            sims.append(DataStructs.TanimotoSimilarity(target_fp, fp) if fp is not None else 0.0)

    result = pool_df.copy()
    result['_tanimoto'] = sims
    result.sort_values('_tanimoto', ascending=False, inplace=True)
    return result.head(top_k).drop(columns=['_tanimoto']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main SALSA algorithm
# ---------------------------------------------------------------------------
def run_salsa_search(
    seed_smiles: str,
    savi_pool_df: pd.DataFrame,
    rounds: int = 3,
    n_perturb: int = 60,
    top_k: int = 5,
    score_col: str = 'combined_score',
    smiles_col: str = 'product_smiles',
    name_col: str = 'product_name',
) -> pd.DataFrame:
    """
    SALSA: Stochastic Approximate Ligand Scoring and Optimisation.

    Starting from *seed_smiles*, iteratively:
      1. Generate *n_perturb* bioisosteric perturbations.
      2. For each perturbation, find its nearest molecule in *savi_pool_df*
         (Tanimoto similarity, Morgan r=2).
      3. Collect all unique pool-hits and pick the one with the highest
         *score_col* value as the new seed for the next round.

    After *rounds* iterations, return the *top_k* pool molecules discovered
    across all rounds, sorted by *score_col* descending.

    Pool fingerprints are pre-computed once before the rounds loop and reused
    for all perturbation queries via BulkTanimotoSimilarity, reducing FP
    computations from O(rounds × n_perturb × pool_size) to O(pool_size).

    All returned rows come from *savi_pool_df*, so their *product_name* values
    are valid for miner submission.

    Args:
        seed_smiles: Starting SMILES — typically the best global_candidate_pool entry.
        savi_pool_df: DataFrame of SAVI-2020 molecules already streamed and PSICHIC-scored.
        rounds: Number of hill-climbing iterations.
        n_perturb: Max perturbation variants generated per round.
        top_k: Number of pool molecules to return.
        score_col: Column used to rank discovered pool molecules.
        smiles_col: Column containing SMILES strings in savi_pool_df.
        name_col: Column containing submittable product names.

    Returns:
        DataFrame of up to *top_k* rows from *savi_pool_df*, sorted by
        *score_col* descending.  May be empty if no perturbations could be
        mapped to pool molecules.
    """
    if savi_pool_df is None or savi_pool_df.empty:
        logger.debug("SALSA: savi_pool_df is empty, skipping.")
        return pd.DataFrame()

    # Pre-compute pool fingerprints once — reused for all perturbation queries.
    # Filters out any rows with unparseable SMILES; valid_pool and pool_fps are aligned.
    valid_pool, pool_fps = precompute_pool_fps(savi_pool_df, smiles_col)
    if valid_pool.empty or not pool_fps:
        logger.debug("SALSA: no valid pool molecules after FP pre-computation, skipping.")
        return pd.DataFrame()

    best_smiles = seed_smiles
    all_hits: List[pd.DataFrame] = []
    seen_names: set[str] = set()

    for round_idx in range(rounds):
        perturbations = generate_perturbations(best_smiles, n_max=n_perturb)
        if not perturbations:
            logger.debug(f"SALSA round {round_idx + 1}: no perturbations generated from {best_smiles!r}")
            break

        round_hits: List[pd.Series] = []
        for pert_smiles in perturbations:
            nearest = nearest_pool_molecules(
                pert_smiles, valid_pool, top_k=1, smiles_col=smiles_col, pool_fps=pool_fps
            )
            if nearest.empty:
                continue
            row = nearest.iloc[0]
            name = row.get(name_col, '')
            if name and name not in seen_names:
                seen_names.add(name)
                round_hits.append(row)

        if not round_hits:
            logger.debug(f"SALSA round {round_idx + 1}: no pool hits found")
            break

        hits_df = pd.DataFrame(round_hits)
        if score_col not in hits_df.columns:
            break

        hits_df = hits_df.sort_values(score_col, ascending=False).reset_index(drop=True)
        all_hits.append(hits_df)

        # New seed: pool molecule nearest to the best perturbation hit
        best_smiles = hits_df.iloc[0].get(smiles_col, best_smiles)
        logger.debug(
            f"SALSA round {round_idx + 1}: best hit = "
            f"{hits_df.iloc[0].get(name_col, '?')} "
            f"(score={hits_df.iloc[0].get(score_col, float('nan')):.4f})"
        )

    if not all_hits:
        return pd.DataFrame()

    combined = pd.concat(all_hits, ignore_index=True)
    combined.drop_duplicates(subset=[name_col], inplace=True)
    combined.sort_values(score_col, ascending=False, inplace=True)
    result = combined.head(top_k).reset_index(drop=True)
    _best = f"{result.iloc[0][score_col]:.4f}" if not result.empty else 'n/a'
    logger.info(
        f"SALSA complete: {len(result)} hits from {len(savi_pool_df)}-molecule pool "
        f"(best score={_best})"
    )
    return result
