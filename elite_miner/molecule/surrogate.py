"""LightGBM surrogate scorer for Boltz2.

Predicts the validator's heavy-atom-normalized combined score *numerator*
(affinity_probability_binary - affinity_pred_value), then divides by RDKit-computed
heavy_atoms at scoring time. Decouples model from the size confound and makes
the surrogate transferable across heavy-atom-count ranges.

Interface mirrors ProxyScorer for drop-in replacement:
    scorer.score(name, smiles) -> ScoredMolecule
    scorer.score_batch(candidates) -> list[ScoredMolecule]

Cold start (no model file present) falls back to a ProxyScorer instance.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .features import mol_features, mol_features_batch, feature_dim
from .scorer import ProxyScorer, ScoredMolecule, _heavy_atom_count


@dataclass
class SurrogateMetrics:
    """Holdout metrics for the trained surrogate. Loaded from holdout_metrics.json."""
    spearman_rho: float
    top5_recall: float
    per_ha_bucket_rho: dict  # {"10-20": 0.42, "20-30": 0.51, ...}
    n_train: int
    n_holdout: int
    trained_at: str
    target: Optional[str] = None  # if model is single-target

    @classmethod
    def load(cls, path: str) -> "SurrogateMetrics":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


class SurrogateScorer:
    """LightGBM-backed scorer.

    Parameters
    ----------
    model_dir : str
        Directory containing model.txt + holdout_metrics.json + feature_version.json
        from elite_miner/scripts/train_surrogate.py.
    target_features : np.ndarray, optional
        Per-target protein feature vector. Concatenated with mol features at predict
        time. If None, no protein features are used (single-target model).
    min_spearman : float, default 0.0
        If model's holdout spearman is below this, fall back to ProxyScorer.
        Used at startup to refuse a weak model.
    """

    def __init__(
        self,
        model_dir: str,
        target_features: Optional[np.ndarray] = None,
        min_spearman: float = 0.0,
        proxy_rng: Optional[random.Random] = None,
    ):
        self.model_dir = model_dir
        self.target_features = target_features
        self.min_spearman = min_spearman
        self.proxy = ProxyScorer(proxy_rng)

        self._model = None
        self._metrics: Optional[SurrogateMetrics] = None
        self._fallback_reason: Optional[str] = None

        self._try_load()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def metrics(self) -> Optional[SurrogateMetrics]:
        return self._metrics

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    # ---- I/O ----
    def _try_load(self) -> None:
        if not os.path.isdir(self.model_dir):
            self._fallback_reason = f"model_dir {self.model_dir} not found"
            return

        model_path = os.path.join(self.model_dir, "model.txt")
        metrics_path = os.path.join(self.model_dir, "holdout_metrics.json")
        if not os.path.exists(model_path):
            self._fallback_reason = f"model.txt missing in {self.model_dir}"
            return

        try:
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=model_path)
        except Exception as e:
            self._fallback_reason = f"failed to load LightGBM model: {e}"
            return

        if os.path.exists(metrics_path):
            try:
                self._metrics = SurrogateMetrics.load(metrics_path)
                if self._metrics.spearman_rho < self.min_spearman:
                    self._fallback_reason = (
                        f"holdout spearman {self._metrics.spearman_rho:.3f} "
                        f"< min_spearman {self.min_spearman}"
                    )
                    self._model = None
                    return
            except Exception as e:
                # Model loaded but metrics didn't — still serviceable, just unknown quality
                self._metrics = None

    # ---- Inference ----
    def _build_X(self, smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
        mat, kept = mol_features_batch(smiles_list)
        if self.target_features is not None and len(kept) > 0:
            tile = np.tile(self.target_features, (mat.shape[0], 1))
            mat = np.concatenate([mat, tile], axis=1)
        return mat, kept

    def score(self, name: str, smiles: str) -> ScoredMolecule:
        if not self.is_ready:
            return self.proxy.score(name, smiles)
        return self.score_batch([(name, smiles)])[0]

    def score_batch(self, candidates: list[tuple[str, str]]) -> list[ScoredMolecule]:
        if not self.is_ready:
            return self.proxy.score_batch(candidates)

        smiles_list = [s for _, s in candidates]
        X, kept = self._build_X(smiles_list)

        # Predict numerator = (pb - pv). Divide by HA at score time for the final score.
        if len(kept) == 0:
            # All SMILES unparseable — proxy them so callers get something
            return self.proxy.score_batch(candidates)

        try:
            preds = self._model.predict(X)
        except Exception as e:
            # Inference failed (shape mismatch etc) — degrade to proxy and log via fallback_reason
            self._fallback_reason = f"predict failed: {e}"
            return self.proxy.score_batch(candidates)

        out: list[ScoredMolecule] = []
        kept_set = set(kept)
        pred_iter = iter(preds)
        for i, (name, smiles) in enumerate(candidates):
            if i not in kept_set:
                # Unparseable — proxy this one
                out.append(self.proxy.score(name, smiles))
                continue
            numerator_pred = float(next(pred_iter))
            ha = _heavy_atom_count(smiles)
            score = -math.inf if ha == 0 else (numerator_pred / ha)
            out.append(
                ScoredMolecule(
                    name=name,
                    smiles=smiles,
                    heavy_atoms=ha,
                    score=score,
                    raw={"numerator_pred": numerator_pred},
                )
            )
        return out
