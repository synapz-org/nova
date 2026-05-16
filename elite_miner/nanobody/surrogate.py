"""LightGBM surrogate scorer for BoltzGen.

Predicts the combined min-is-better score from sequence + target features.
Drop-in replacement for ProxyNanobodyScorer; falls back to proxy on cold start
or model load failure.

Architecture choice: we predict the **combined scalar** (signed normalization
of the 10 raw BoltzGen metrics, see scorer.combined_score_from_metrics) rather
than each metric independently. Simpler, lower-variance, and matches the
quantity NanobodyTrack actually ranks on.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .features import nb_features, nb_features_batch, feature_dim
from .scorer import ProxyNanobodyScorer, ScoredNanobody


@dataclass
class NanobodySurrogateMetrics:
    spearman_rho: float
    top_decile_recall: float
    n_train: int
    n_holdout: int
    trained_at: str
    target: Optional[str] = None

    @classmethod
    def load(cls, path: str) -> "NanobodySurrogateMetrics":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


class NanobodySurrogateScorer:
    """LightGBM-backed scorer. Interface mirrors ProxyNanobodyScorer."""

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
        self.proxy = ProxyNanobodyScorer(proxy_rng)

        self._model = None
        self._metrics: Optional[NanobodySurrogateMetrics] = None
        self._fallback_reason: Optional[str] = None
        self._try_load()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def metrics(self) -> Optional[NanobodySurrogateMetrics]:
        return self._metrics

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

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
                self._metrics = NanobodySurrogateMetrics.load(metrics_path)
                if self._metrics.spearman_rho < self.min_spearman:
                    self._fallback_reason = (
                        f"holdout spearman {self._metrics.spearman_rho:.3f} "
                        f"< min_spearman {self.min_spearman}"
                    )
                    self._model = None
                    return
            except Exception:
                self._metrics = None

    def _build_X(self, sequences: list[str]) -> tuple[np.ndarray, list[int]]:
        mat, kept = nb_features_batch(sequences)
        if self.target_features is not None and len(kept) > 0:
            tile = np.tile(self.target_features, (mat.shape[0], 1))
            mat = np.concatenate([mat, tile], axis=1)
        return mat, kept

    def score(self, seq: str) -> ScoredNanobody:
        if not self.is_ready:
            return self.proxy.score(seq)
        return self.score_batch([seq])[0]

    def score_batch(self, sequences: list[str]) -> list[ScoredNanobody]:
        if not self.is_ready:
            return self.proxy.score_batch(sequences)
        X, kept = self._build_X(sequences)
        if len(kept) == 0:
            return self.proxy.score_batch(sequences)
        try:
            preds = self._model.predict(X)
        except Exception as e:
            self._fallback_reason = f"predict failed: {e}"
            return self.proxy.score_batch(sequences)

        out: list[ScoredNanobody] = []
        kept_set = set(kept)
        pred_iter = iter(preds)
        for i, seq in enumerate(sequences):
            if i not in kept_set:
                out.append(self.proxy.score(seq))
                continue
            score = float(next(pred_iter))
            out.append(ScoredNanobody(sequence=seq, score=score, raw={"surrogate_pred": score}))
        return out
