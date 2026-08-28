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
# §MMMMMM: Cross-call pool fingerprint cache.
# precompute_pool_fps on a 10 000-molecule pool takes ~2–4 s in Python/RDKit.
# In §MM, the same pool DataFrame object is passed on every round (up to 20 on
# H100) when §IIIIII is inactive (Ridge tier, epochs 1–2).  This cache keyed
# by DataFrame object id() short-circuits the recompute and returns the prior
# result directly.  When §IIIIII fires (RF tier, ≥100 pts) it creates a new
# DataFrame object → new id() → cache miss → recompute.  The cache is bounded
# to 10 entries to prevent unbounded memory growth across epochs.
# ---------------------------------------------------------------------------
_fp_cache: dict = {}   # {(pool_id, smiles_col): (valid_pool, fps_list)}

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

# Atomic numbers of single heavy atoms to *append* at positions that still
# carry an implicit hydrogen.  The resulting SMILES are used only as query
# probes for the nearest-SAVI-2020 Tanimoto search — they are never submitted
# directly.  Appending these atoms expands the search radius beyond pure
# bioisosteric substitution: F/Cl add electron-withdrawing groups; C adds a
# methyl; N adds a primary amine; O adds a hydroxyl.
_FG_ATOMS: list[int] = [9, 17, 6, 7, 8]  # F, Cl, C (methyl), N (amine), O (hydroxyl)

# Atom types eligible as attachment points for FG addition.
_FG_ATTACHMENT_ATOMS: frozenset[int] = frozenset([6, 7, 8, 16])  # C, N, O, S


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


def get_cached_pool_fps(
    pool_df: pd.DataFrame,
    smiles_col: str = 'product_smiles',
) -> Tuple[pd.DataFrame, List]:
    """
    §NNNNNN: Cache-backed wrapper for precompute_pool_fps.
    Uses the same module-level _fp_cache (§MMMMMM) as run_salsa_search, so
    callers outside run_salsa_search (e.g., §XX in miner.py) can reuse pool
    FPs already computed during §MM rounds on the same DataFrame object.
    """
    _cache_key = (id(pool_df), smiles_col)
    if _cache_key in _fp_cache:
        logger.debug(f"[§NNNNNN/get_cached_pool_fps] FP cache hit for pool id={id(pool_df)}")
        return _fp_cache[_cache_key]
    valid_pool, pool_fps = precompute_pool_fps(pool_df, smiles_col)
    _fp_cache[_cache_key] = (valid_pool, pool_fps)
    if len(_fp_cache) > 10:
        _fp_cache.pop(next(iter(_fp_cache)))
    return valid_pool, pool_fps


# ---------------------------------------------------------------------------
# Perturbation operators
# ---------------------------------------------------------------------------
def generate_perturbations(
    smiles: str,
    n_max: int = 100,
    operator_weights: Optional[dict] = None,
    return_tags: bool = False,
) -> List:
    """
    Generate up to n_max unique canonical SMILES variants of *smiles* via
    four complementary operators:

    1. **Bioisosteric substitution** — replace each heavy atom with its
       bioisosteric equivalents (C↔N, O↔S, Cl↔F, …).  Explores the same
       molecular size with altered electronics/H-bonding.

    2. **Functional group addition** — append a single heavy atom (F, Cl,
       methyl-C, amine-N, hydroxyl-O) at every position that still carries
       an implicit hydrogen.  Explores molecules one atom larger, sweeping
       a broader radius in the Tanimoto fingerprint space.

    3. **Terminal atom removal** — remove each terminal (degree-1) heavy atom.
       Produces probes one atom smaller, targeting the Boltz-2 scoring formula's
       heavy_atom_count denominator.

    4. **Ring walk** — expand 4–6-membered rings by inserting CH₂ into a single
       ring bond (+1 ring size), and contract 5–7-membered rings by removing a
       degree-2 ring carbon (-1 ring size).  Covers 5↔6 and 6↔7 transitions
       orthogonal to all three operators above.

    All operators produce *probe* SMILES used only for nearest-SAVI-2020
    Tanimoto lookup — they are never submitted directly.

    operator_weights: optional dict with keys 'bioisostere', 'fg_add',
        'terminal_remove', 'ring_walk'.  Values are relative weights (higher =
        more budget allocated).  §ZZZZZ uses HA-adaptive weights so that
        large seeds (>25 HA) bias toward terminal_remove (find smaller
        neighbours) and small seeds (<15 HA) bias toward fg_add (find
        molecules with more pharmacophore features).  When None, the budget
        is divided equally across operators — identical to previous behaviour.

    return_tags: if True, return List[Tuple[str, str]] of (operator_tag, smiles)
        instead of List[str].  operator_tag is one of 'bioisostere', 'fg_add',
        'terminal_remove', 'ring_walk'.  Used by §OOOO bandit operator tracking.

    Returns a list of valid canonical SMILES strings (excluding the input),
    or (when return_tags=True) a list of (operator_tag, smiles) tuples.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    seen = {Chem.MolToSmiles(mol)}

    # §ZZZZZ: Per-operator budget allocation.
    # Default equal-weight split preserves original behaviour when caller
    # passes operator_weights=None.
    _def_w = {'bioisostere': 1.0, 'fg_add': 1.0, 'terminal_remove': 1.0, 'ring_walk': 1.0}
    if operator_weights:
        _w = {k: max(0.0, float(operator_weights.get(k, _def_w[k]))) for k in _def_w}
    else:
        _w = _def_w
    _tw = max(sum(_w.values()), 1e-9)
    _n_bio = max(2, round(n_max * _w['bioisostere'] / _tw))
    _n_fga = max(2, round(n_max * _w['fg_add'] / _tw))
    _n_ter = max(2, round(n_max * _w['terminal_remove'] / _tw))
    _n_rng = max(2, n_max - _n_bio - _n_fga - _n_ter)

    bio_res: List[str] = []
    fga_res: List[str] = []
    ter_res: List[str] = []
    rng_res: List[str] = []

    # --- 1. Bioisosteric substitution ---
    for atom in mol.GetAtoms():
        if len(bio_res) >= _n_bio:
            break
        an = atom.GetAtomicNum()
        for target_an in _BIOISOSTERES.get(an, []):
            if len(bio_res) >= _n_bio:
                break
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(target_an)
            try:
                Chem.SanitizeMol(rw)
                canonical = Chem.MolToSmiles(rw.GetMol())
                if canonical not in seen:
                    seen.add(canonical)
                    bio_res.append(canonical)
            except Exception:
                pass

    # --- 2. Functional group addition ---
    # Append one heavy atom at each position with available implicit hydrogens.
    # SanitizeMol discards valence violations silently via the try/except.
    for atom in mol.GetAtoms():
        if len(fga_res) >= _n_fga:
            break
        if atom.GetTotalNumHs() == 0:
            continue
        if atom.GetAtomicNum() not in _FG_ATTACHMENT_ATOMS:
            continue
        for fg_an in _FG_ATOMS:
            if len(fga_res) >= _n_fga:
                break
            rw = Chem.RWMol(mol)
            new_idx = rw.AddAtom(Chem.Atom(fg_an))
            rw.AddBond(atom.GetIdx(), new_idx, Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(rw)
                canonical = Chem.MolToSmiles(rw.GetMol())
                if canonical not in seen:
                    seen.add(canonical)
                    fga_res.append(canonical)
            except Exception:
                pass

    # --- 3. Terminal atom removal ---
    # Remove each terminal heavy atom (degree=1, not H) to generate probes that
    # are one atom smaller.  Stripping a halogen, methyl, or hydroxyl from the
    # seed and mapping back to SAVI-2020 finds molecules with fewer heavy atoms —
    # directly targeting the scoring formula's heavy_atom_count denominator.
    # These probes are never submitted; they are query vectors for Tanimoto search.
    for atom in mol.GetAtoms():
        if len(ter_res) >= _n_ter:
            break
        if atom.GetDegree() != 1 or atom.GetAtomicNum() <= 1:
            continue  # only terminal non-hydrogen heavy atoms
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(atom.GetIdx())
        try:
            Chem.SanitizeMol(rw)
            canonical = Chem.MolToSmiles(rw.GetMol())
            if canonical not in seen:
                seen.add(canonical)
                ter_res.append(canonical)
        except Exception:
            pass

    # --- 4. Ring walk (ring size ±1) ---
    # 4a. Ring expansion: insert CH₂ into each single bond within a 4–6 membered
    #     ring, producing a ring one atom larger (5–7 membered).  Avoids expanding
    #     rings already ≥7 atoms to prevent macrocycles.
    # 4b. Ring contraction: remove each degree-2 ring carbon from a 5–7 membered
    #     ring, reconnecting its two neighbours.  Avoids contracting below 4-membered
    #     rings (5-membered is the smallest we contract from).
    # Like all operators above, results are query probes for nearest-SAVI-2020
    # Tanimoto search — never submitted directly.
    _ring_info = mol.GetRingInfo()

    # 4a — expansion: collect bonds in rings of size 4–6
    _small_ring_bonds: set = set()
    for _r in _ring_info.BondRings():
        if 4 <= len(_r) <= 6:
            _small_ring_bonds.update(_r)

    for _bond_idx in _small_ring_bonds:
        if len(rng_res) >= _n_rng:
            break
        _bond = mol.GetBondWithIdx(_bond_idx)
        if _bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        _bi = _bond.GetBeginAtomIdx()
        _ei = _bond.GetEndAtomIdx()
        rw = Chem.RWMol(mol)
        rw.RemoveBond(_bi, _ei)
        _ni = rw.AddAtom(Chem.Atom(6))  # insert CH₂
        rw.AddBond(_bi, _ni, Chem.BondType.SINGLE)
        rw.AddBond(_ni, _ei, Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(rw)
            canonical = Chem.MolToSmiles(rw.GetMol())
            if canonical not in seen:
                seen.add(canonical)
                rng_res.append(canonical)
        except Exception:
            pass

    # 4b — contraction: remove degree-2 ring carbons from 5–7 membered rings
    for _ring in _ring_info.AtomRings():
        if len(rng_res) >= _n_rng:
            break
        if not (5 <= len(_ring) <= 7):
            continue
        for _rpos, _ai in enumerate(_ring):
            if len(rng_res) >= _n_rng:
                break
            _atom = mol.GetAtomWithIdx(_ai)
            if _atom.GetAtomicNum() != 6:
                continue  # only remove unsubstituted carbons
            if _atom.GetDegree() != 2:
                continue  # substituent present — removal would lose a group
            _prev = _ring[(_rpos - 1) % len(_ring)]
            _next = _ring[(_rpos + 1) % len(_ring)]
            # Adjust neighbour indices after _ai is removed
            _adj_prev = _prev - (1 if _prev > _ai else 0)
            _adj_next = _next - (1 if _next > _ai else 0)
            rw = Chem.RWMol(mol)
            rw.RemoveAtom(_ai)
            # Reconnect the two former neighbours to close the smaller ring
            if rw.GetBondBetweenAtoms(_adj_prev, _adj_next) is None:
                rw.AddBond(_adj_prev, _adj_next, Chem.BondType.SINGLE)
            try:
                Chem.SanitizeMol(rw)
                canonical = Chem.MolToSmiles(rw.GetMol())
                if canonical not in seen:
                    seen.add(canonical)
                    rng_res.append(canonical)
            except Exception:
                pass

    # §OOOO: tagged output for bandit operator tracking in §MM.
    if return_tags:
        return (
            [('bioisostere', s) for s in bio_res]
            + [('fg_add', s) for s in fga_res]
            + [('terminal_remove', s) for s in ter_res]
            + [('ring_walk', s) for s in rng_res]
        )
    return bio_res + fga_res + ter_res + rng_res


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
# §UUUUUUUUUUUU: Surrogate pre-filter for perturbation probes
# ---------------------------------------------------------------------------

def _prefilter_perturbations_with_surrogate(
    perturbations: List[str],
    dual_surrogate,
    top_fraction: float = 0.25,
    min_keep: int = 10,
    max_keep: int = 50,
) -> List[str]:
    """
    §UUUUUUUUUUUU: Pre-score perturbation probe SMILES with the dual RF surrogate
    and keep only the top fraction before the Tanimoto pool-lookup step.

    Scoring 200 probes takes ~20 ms (vectorized RF predict); the lookup savings
    (~75% fewer pool queries) free CPU for 3–4 additional §MM rounds per epoch.

    Only activates at the RF tier (≥100 training points). Falls back to returning
    perturbations unchanged when the surrogate is None, uses Ridge, or scoring fails.
    """
    if dual_surrogate is None or len(perturbations) <= min_keep:
        return perturbations

    try:
        from sklearn.ensemble import RandomForestRegressor
        model_apb, model_apv = dual_surrogate
        if not (
            isinstance(model_apb.named_steps.get('model'), RandomForestRegressor)
            and isinstance(model_apv.named_steps.get('model'), RandomForestRegressor)
        ):
            return perturbations
    except Exception:
        return perturbations

    try:
        import numpy as np
        from utils.surrogate import _descriptor_vector

        vecs: List = []
        ha_vals: List[float] = []
        valid_idxs: List[int] = []
        for i, smi in enumerate(perturbations):
            vec = _descriptor_vector(smi)
            if vec is None:
                continue
            mol = Chem.MolFromSmiles(smi)
            ha = mol.GetNumHeavyAtoms() if mol is not None else 25
            vecs.append(vec + [0.0])  # psichic_le=0.0 neutral prior for probe SMILES
            ha_vals.append(float(max(1, ha)))
            valid_idxs.append(i)

        if len(vecs) < min_keep:
            return perturbations

        apb_preds = model_apb.predict(vecs)
        apv_preds = model_apv.predict(vecs)
        scores = (apb_preds - apv_preds) / np.array(ha_vals)

        n_keep = min(max_keep, max(min_keep, int(len(vecs) * top_fraction)))
        top_local = set(np.argsort(scores)[::-1][:n_keep].tolist())
        kept_orig = {valid_idxs[li] for li in top_local}
        filtered = [smi for i, smi in enumerate(perturbations) if i in kept_orig]
        logger.debug(
            f"[§UUUUUUUUUUUU] Surrogate pre-filter: {len(perturbations)} → {len(filtered)} perturbations"
        )
        return filtered
    except Exception as _e:
        logger.debug(f"[§UUUUUUUUUUUU] Pre-filter skipped ({_e})")
        return perturbations


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
    operator_weights: Optional[dict] = None,
    out_operator_tags: Optional[dict] = None,
    dual_surrogate=None,
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
        operator_weights: Optional dict mapping operator names to relative weights
            (bioisostere, fg_add, terminal_remove, ring_walk).  Passed through to
            generate_perturbations to allocate per-operator budgets.  None → equal
            weights (§ZZZZZ).
        out_operator_tags: §OOOO — if not None, populated in-place with
            {product_name: operator_tag} recording which operator first discovered
            each returned hit.  Enables the §MM caller to credit bandit wins per
            operator and bias subsequent rounds toward productive operators.
        dual_surrogate: §UUUUUUUUUUUU — optional (model_apb, model_apv) pair from
            fit_dual_surrogate.  When provided at RF tier (≥100 pts), each round's
            perturbation probes are pre-scored and the bottom 75% are dropped before
            pool lookup, concentrating Tanimoto queries on the most Boltz-promising
            chemical directions.  None or Ridge → no-op.

    Returns:
        DataFrame of up to *top_k* rows from *savi_pool_df*, sorted by
        *score_col* descending.  May be empty if no perturbations could be
        mapped to pool molecules.
    """
    if savi_pool_df is None or savi_pool_df.empty:
        logger.debug("SALSA: savi_pool_df is empty, skipping.")
        return pd.DataFrame()

    # §MMMMMM: Check cross-call FP cache before recomputing.  The cache is keyed
    # by (id(pool_df), smiles_col) so it hits when the same DataFrame object is
    # passed on consecutive §MM rounds (Ridge-tier, where §IIIIII is inactive and
    # _mm_savi_pool stays the same object).  A new DataFrame id — from §IIIIII's
    # surrogate-blended copy or a new PSICHIC streaming chunk — is a cache miss,
    # triggering a fresh precompute_pool_fps.  Bounded to 10 entries.
    _cache_key = (id(savi_pool_df), smiles_col)
    if _cache_key in _fp_cache:
        valid_pool, pool_fps = _fp_cache[_cache_key]
        logger.debug(f"[§MMMMMM] FP cache hit for pool id={id(savi_pool_df)}, "
                     f"size={len(savi_pool_df)}")
    else:
        valid_pool, pool_fps = precompute_pool_fps(savi_pool_df, smiles_col)
        _fp_cache[_cache_key] = (valid_pool, pool_fps)
        if len(_fp_cache) > 10:
            _fp_cache.pop(next(iter(_fp_cache)))
    if valid_pool.empty or not pool_fps:
        logger.debug("SALSA: no valid pool molecules after FP pre-computation, skipping.")
        return pd.DataFrame()

    best_smiles = seed_smiles
    all_hits: List[pd.DataFrame] = []
    seen_names: set[str] = set()
    # §AAAAAAAAA: track previous round's best so we can detect convergence.
    _prev_best_smiles: Optional[str] = None

    for round_idx in range(rounds):
        # §OOOO: use tagged output when caller wants operator tracking; plain list otherwise.
        if out_operator_tags is not None:
            _tagged = generate_perturbations(
                best_smiles, n_max=n_perturb,
                operator_weights=operator_weights, return_tags=True,
            )
            _probe_to_op: dict = {smi: op for op, smi in _tagged}
            perturbations = [smi for _, smi in _tagged]
        else:
            _probe_to_op = {}
            perturbations = generate_perturbations(
                best_smiles, n_max=n_perturb, operator_weights=operator_weights,
            )
        if not perturbations:
            logger.debug(f"SALSA round {round_idx + 1}: no perturbations generated from {best_smiles!r}")
            break

        # §UUUUUUUUUUUU: Drop the bottom 75% of probes by predicted surrogate LE before
        # Tanimoto pool lookup.  _probe_to_op (keyed by SMILES) remains valid after
        # filtering since only the probe-SMILES subset changes, not the dict mapping.
        if dual_surrogate is not None:
            perturbations = _prefilter_perturbations_with_surrogate(perturbations, dual_surrogate)

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
                # §OOOO: record which operator first discovered this hit
                if out_operator_tags is not None and name not in out_operator_tags:
                    out_operator_tags[name] = _probe_to_op.get(pert_smiles, 'unknown')

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

        # §AAAAAAAAA: convergence guard — if the best seed didn't change from
        # the previous round, every subsequent round would generate identical
        # perturbations and re-visit the same pool neighbors.  Stop early and
        # free the CPU budget for §MM's next hill-climbing molecule or another
        # SALSA instance.  Only fires when ≥1 full round has already run
        # (_prev_best_smiles is set) so we always complete at least one round.
        if _prev_best_smiles is not None and best_smiles == _prev_best_smiles:
            logger.debug(
                f"[§AAAAAAAAA] SALSA converged at round {round_idx + 1} "
                f"(seed unchanged: {best_smiles!r}) — stopping early."
            )
            break
        _prev_best_smiles = best_smiles

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
