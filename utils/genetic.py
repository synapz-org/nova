"""
GradientGA — Gradient-Guided Genetic Algorithm for SN68 Mining

Population-based search over SAVI-2020 chemical space.  Maintains a pool of
molecules across generations, using PSICHIC-derived combined_score (already
computed for the savi_stream_pool in the miner) as the cheap fitness proxy.
Crossover is via BRICS fragment exchange; mutation reuses SALSA's bioisosteric
perturbations.  All offspring are mapped back to the nearest SAVI-2020 molecule
in the streamed pool, so every returned candidate has a valid product_name and
can be submitted directly or added to global_candidate_pool for Boltz-2 scoring.

Typical call from miner.py:

    ga_hits = await asyncio.to_thread(
        run_gradient_ga,
        seed_df=state['global_candidate_pool'],
        savi_pool_df=state['savi_stream_pool'],
        n_generations=5,
        pop_size=50,
        top_k=5,
    )
    if not ga_hits.empty:
        # merge into global_candidate_pool
        ...

Runtime: ~1–3 s (CPU) for 5 generations × 50-molecule population.
"""
from __future__ import annotations

import logging
import random
from typing import List, Optional, Set, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS

from utils.salsa import generate_perturbations, nearest_pool_molecules

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BRICS crossover
# ---------------------------------------------------------------------------

def _brics_fragments(mol: Chem.Mol) -> List[str]:
    """Decompose *mol* into BRICS fragment SMILES."""
    try:
        return list(BRICS.BRICSDecompose(mol))
    except Exception:
        return []


def brics_crossover(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    max_offspring: int = 4,
    max_pairs: int = 20,
) -> List[str]:
    """
    Exchange BRICS fragments between *mol_a* and *mol_b* and return up to
    *max_offspring* unique canonical SMILES for valid offspring molecules.

    Caps the fragment-pair search to *max_pairs* to avoid combinatorial blow-up.
    Returns an empty list if either parent cannot be fragmented or no valid
    offspring can be assembled.
    """
    frags_a = _brics_fragments(mol_a)
    frags_b = _brics_fragments(mol_b)
    if not frags_a or not frags_b:
        return []

    pairs: List[Tuple[str, str]] = [(fa, fb) for fa in frags_a for fb in frags_b]
    random.shuffle(pairs)

    seen: Set[str] = set()
    offspring: List[str] = []

    for fa_smi, fb_smi in pairs[:max_pairs]:
        try:
            fa_mol = Chem.MolFromSmiles(fa_smi)
            fb_mol = Chem.MolFromSmiles(fb_smi)
            if fa_mol is None or fb_mol is None:
                continue
            built = list(BRICS.BRICSBuild([fa_mol, fb_mol]))
            for m in built[:3]:
                if m is None:
                    continue
                try:
                    Chem.SanitizeMol(m)
                    smi = Chem.MolToSmiles(m)
                    if smi not in seen:
                        seen.add(smi)
                        offspring.append(smi)
                        if len(offspring) >= max_offspring:
                            return offspring
                except Exception:
                    pass
        except Exception:
            pass

    return offspring


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------

def tournament_select(
    population: pd.DataFrame,
    n_pairs: int,
    k: int = 3,
    score_col: str = 'combined_score',
    smiles_col: str = 'product_smiles',
) -> List[Tuple[str, str]]:
    """
    Select *n_pairs* parent pairs from *population* via tournament selection.

    Each parent is chosen by sampling *k* candidates and picking the one with
    the highest *score_col* value.  The two parents in each pair are guaranteed
    to be distinct.

    Returns a list of (smiles_a, smiles_b) tuples.
    """
    n = len(population)
    if n < 2:
        return []

    pairs: List[Tuple[str, str]] = []
    for _ in range(n_pairs):
        # Parent A
        idx_a = population.sample(min(k, n))[score_col].idxmax()
        smi_a = population.loc[idx_a, smiles_col]

        # Parent B (must differ from A)
        rest = population[population[smiles_col] != smi_a]
        if rest.empty:
            continue
        idx_b = rest.sample(min(k, len(rest)))[score_col].idxmax()
        smi_b = rest.loc[idx_b, smiles_col]

        pairs.append((smi_a, smi_b))

    return pairs


# ---------------------------------------------------------------------------
# Main GA
# ---------------------------------------------------------------------------

def run_gradient_ga(
    seed_df: pd.DataFrame,
    savi_pool_df: pd.DataFrame,
    n_generations: int = 5,
    pop_size: int = 50,
    top_k: int = 5,
    score_col: str = 'combined_score',
    smiles_col: str = 'product_smiles',
    name_col: str = 'product_name',
) -> pd.DataFrame:
    """
    Gradient-Guided Genetic Algorithm over SAVI-2020 chemical space.

    Maintains a population of *pop_size* SAVI-2020 molecules.  Each generation:
      1. Select ~25 parent pairs by tournament selection (fitness = combined_score).
      2. BRICS crossover on each pair → offspring SMILES.
      3. Bioisosteric mutation on each parent A → additional SMILES.
      4. Map every offspring/mutant to its nearest SAVI-2020 molecule in
         *savi_pool_df* (Tanimoto, Morgan r=2).
      5. Merge new hits with the population; keep top *pop_size* by score.
      6. Elitism: the best molecule found so far is never displaced.

    All returned molecules come from *savi_pool_df* and have valid product_name
    values for submission.

    Args:
        seed_df:      Initial elites (e.g. global_candidate_pool).
        savi_pool_df: All PSICHIC-scored SAVI-2020 molecules streamed this epoch.
        n_generations: Number of GA iterations.
        pop_size:     Maximum population size per generation.
        top_k:        Number of best-discovered molecules to return.
        score_col:    Column used as fitness proxy (combined_score).
        smiles_col:   SMILES column name in both DataFrames.
        name_col:     Product name column name (must be submittable).

    Returns:
        DataFrame of up to *top_k* rows from *savi_pool_df*, sorted by
        *score_col* descending.  May be empty if the pool has no scorable rows.
    """
    if savi_pool_df is None or savi_pool_df.empty:
        logger.debug("GradientGA: savi_pool_df is empty — skipping.")
        return pd.DataFrame()
    if seed_df is None or seed_df.empty:
        logger.debug("GradientGA: seed_df is empty — skipping.")
        return pd.DataFrame()

    # ------------------------------------------------------------------ init
    # Seed population: elites + random sample from the stream pool
    sample_n = min(pop_size, len(savi_pool_df))
    sampled = savi_pool_df.sample(sample_n, random_state=42)
    init = pd.concat([seed_df, sampled], ignore_index=True)
    init.drop_duplicates(subset=[name_col], inplace=True)
    init.sort_values(score_col, ascending=False, inplace=True)
    population = init.head(pop_size).reset_index(drop=True)

    # Track every unique molecule found (including initial elites)
    all_found: List[pd.DataFrame] = [seed_df.copy()]
    elite_score: float = (
        population.iloc[0][score_col] if not population.empty else float('-inf')
    )

    # ------------------------------------------------------------------ loop
    for gen_idx in range(n_generations):
        if len(population) < 2:
            break

        n_pairs = min(25, len(population) // 2)
        pairs = tournament_select(
            population, n_pairs=n_pairs, k=3,
            score_col=score_col, smiles_col=smiles_col,
        )

        candidate_smiles: List[str] = []
        for smi_a, smi_b in pairs:
            mol_a = Chem.MolFromSmiles(smi_a)
            mol_b = Chem.MolFromSmiles(smi_b)

            # Crossover
            if mol_a is not None and mol_b is not None:
                children = brics_crossover(mol_a, mol_b, max_offspring=3)
                candidate_smiles.extend(children)

            # Mutation on parent A (bioisosteric substitution)
            mutants = generate_perturbations(smi_a, n_max=5)
            candidate_smiles.extend(mutants[:3])

        if not candidate_smiles:
            logger.debug(f"GA gen {gen_idx + 1}: no candidates generated")
            continue

        # Map offspring → nearest SAVI-2020 molecule
        seen_names: Set[str] = set(population[name_col].tolist())
        new_hit_frames: List[pd.DataFrame] = []

        for cand_smi in candidate_smiles:
            nearest = nearest_pool_molecules(
                cand_smi, savi_pool_df, top_k=1, smiles_col=smiles_col
            )
            if nearest.empty:
                continue
            row_name = nearest.iloc[0].get(name_col, '')
            if row_name and row_name not in seen_names:
                seen_names.add(row_name)
                new_hit_frames.append(nearest)

        if not new_hit_frames:
            logger.debug(f"GA gen {gen_idx + 1}: no new SAVI hits mapped")
            continue

        new_hits = pd.concat(new_hit_frames, ignore_index=True)

        # Merge and apply elitism: best ever molecule stays in the pool
        combined = pd.concat([population, new_hits], ignore_index=True)
        combined.drop_duplicates(subset=[name_col], inplace=True)
        combined.sort_values(score_col, ascending=False, inplace=True)
        population = combined.head(pop_size).reset_index(drop=True)

        # Update elite score tracker
        if not population.empty:
            elite_score = max(elite_score, population.iloc[0][score_col])

        all_found.append(new_hits)
        logger.info(
            f"GA gen {gen_idx + 1}: {len(new_hits)} new hits mapped; "
            f"pop best={population.iloc[0][score_col]:.4f}"
        )

    # ---------------------------------------------------------------- result
    if not all_found:
        return pd.DataFrame()

    combined_all = pd.concat(all_found, ignore_index=True)
    combined_all.drop_duplicates(subset=[name_col], inplace=True)
    combined_all.sort_values(score_col, ascending=False, inplace=True)
    result = combined_all.head(top_k).reset_index(drop=True)

    _best = f"{result.iloc[0][score_col]:.4f}" if not result.empty else 'n/a'
    logger.info(
        f"GradientGA complete: {len(result)} hits returned "
        f"(best score={_best})"
    )
    return result
