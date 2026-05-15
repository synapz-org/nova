"""Molecule scoring: proxy (offline) + Boltz2 (real inference, gated)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem


def _heavy_atom_count(smiles: str) -> int:
    """Mirror of utils.molecules.get_heavy_atom_count. Inlined to avoid utils/__init__ heavy deps."""
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


@dataclass
class ScoredMolecule:
    name: str            # "rxn:r:m1:m2[:m3]"
    smiles: str
    heavy_atoms: int
    score: float         # combined score, higher is better (boltz_mode='max' from config)
    raw: Optional[dict] = None  # {affinity_probability_binary, affinity_pred_value, ...} when real


def _combined_score(affinity_prob_binary: float, affinity_pred_value: float, heavy_atoms: int) -> float:
    """Mirror validator's heavy_atom_normalization combination:
    (boltz_metric[0] - boltz_metric[1]) / heavy_atom_count

    For config (boltz_metric = [affinity_probability_binary, affinity_pred_value]):
        score = (prob_binary - pred_value) / heavy_atoms

    Higher is better (config boltz_mode = 'max').
    """
    if heavy_atoms == 0:
        return -math.inf
    return (affinity_prob_binary - affinity_pred_value) / heavy_atoms


class ProxyScorer:
    """Cheap offline proxy. Used both for ranking pre-Boltz2 candidates and
    standalone (v1) scoring.

    Important: the validator's heavy_atom_normalization combines as
    (prob_binary - pred_value) / heavy_atoms. A naive uniform-random proxy
    biases toward the smallest molecules (denominator) and hurts us against
    random sampling. So:
      - Prefer mid-range heavy-atom counts (~25-35 is drug-like sweet spot)
      - Penalize extremes (very small = denominator artifact; very large = bad ADMET)
    """

    # Drug-like sweet spot for heavy atoms (typical small-molecule binders)
    HA_SWEET_LOW = 20
    HA_SWEET_HIGH = 40

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def _ha_penalty(self, ha: int) -> float:
        """0 inside [low, high]; quadratic penalty outside."""
        if ha < self.HA_SWEET_LOW:
            return (self.HA_SWEET_LOW - ha) ** 2 * 0.005
        if ha > self.HA_SWEET_HIGH:
            return (ha - self.HA_SWEET_HIGH) ** 2 * 0.005
        return 0.0

    def score(self, name: str, smiles: str) -> ScoredMolecule:
        ha = _heavy_atom_count(smiles)
        # Mock affinity components — fixed distribution so randomness adds
        # tie-breaking variation but not magnitude
        pb = self.rng.uniform(0.4, 0.7)
        pv = self.rng.uniform(-1.0, -0.3)
        base = _combined_score(pb, pv, ha)
        # Subtract penalty for non-drug-like size (higher is better, so penalty reduces score)
        s = base - self._ha_penalty(ha)
        return ScoredMolecule(
            name=name,
            smiles=smiles,
            heavy_atoms=ha,
            score=s,
            raw={"affinity_probability_binary": pb, "affinity_pred_value": pv},
        )

    def score_batch(self, candidates: list[tuple[str, str]]) -> list[ScoredMolecule]:
        return [self.score(n, s) for n, s in candidates]


class Boltz2Scorer:
    """Wraps the vendored Boltz2 predict() for single-target miner-side scoring.

    Lazy-imports torch/boltz so the rest of the miner can be tested without
    a CUDA env. Activate via --use-inference at the run.py level.
    """

    def __init__(self, target: str, clip_interval=None, config_path: Optional[str] = None):
        self.target = target
        self.clip_interval = clip_interval
        self.config_path = config_path
        self._wrapper = None  # lazy

    def _ensure_loaded(self):
        if self._wrapper is not None:
            return
        # Lazy imports — only triggered when caller has GPU + boltz deps.
        from external_tools.boltz.boltz_wrapper import BoltzWrapper
        self._wrapper = BoltzWrapper()

    def score_batch(self, candidates: list[tuple[str, str]], subnet_config: dict) -> list[ScoredMolecule]:
        """Run Boltz2 on a batch of candidates against self.target.

        subnet_config must include: small_molecule_target, small_molecule_target_clip_interval,
        boltz_metric, combination_strategy, boltz_mode.
        Returns a list of ScoredMolecule in the same order as `candidates`.
        """
        self._ensure_loaded()

        # Build the validator-shape input the wrapper expects, but with one fake "uid=0"
        # so the bookkeeping works. We unpack the per-molecule components afterwards.
        smiles_list = [s for _, s in candidates]
        valid_molecules_by_uid = {0: {"smiles": smiles_list, "names": [n for n, _ in candidates]}}
        score_dict = {0: {"molecule_scores": [[]]}}

        self._wrapper.score_molecules(valid_molecules_by_uid, score_dict, subnet_config)

        # Map per-molecule components back to our ScoredMolecule objects
        out: list[ScoredMolecule] = []
        per_mol = getattr(self._wrapper, "per_molecule_components", {}) or {}
        for name, smiles in candidates:
            ha = _heavy_atom_count(smiles)
            metrics = per_mol.get(smiles, {}).get(self.target, {})
            if not metrics:
                out.append(ScoredMolecule(name=name, smiles=smiles, heavy_atoms=ha, score=-math.inf))
                continue
            pb = metrics.get("affinity_probability_binary", 0.0)
            pv = metrics.get("affinity_pred_value", 0.0)
            out.append(
                ScoredMolecule(
                    name=name,
                    smiles=smiles,
                    heavy_atoms=ha,
                    score=_combined_score(pb, pv, ha),
                    raw=metrics,
                )
            )
        return out


def rank(scored: list[ScoredMolecule]) -> list[ScoredMolecule]:
    """Sort descending by score (validator uses boltz_mode='max')."""
    return sorted(scored, key=lambda s: s.score, reverse=True)
