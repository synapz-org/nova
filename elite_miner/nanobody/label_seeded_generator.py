"""Generator that mutates from sequences we've actually measured to score well.

Unlike ArchiveSeededGenerator (which uses the top-200 of the public
Submission-Archive — known winners but also what every competitor is targeting),
LabelSeededGenerator reads seeds from `cache/offline_labels/<target>.parquet`,
keeping only entries with `real_iiptm >= threshold`.

The hope: by mutating from sequences WE labeled at iiptm 0.80+, we avoid the
"shared archive neighborhood" effect ([[archive-seeded-strategy-likely-shared]]
in kb/notes) and move into less-saturated parts of sequence space.

Risk: with only a handful of labels, the seed pool is small and similar. If
all 5-10 winners cluster tightly, this generator is no more diverse than
ArchiveSeededGenerator with seeds=10. Watch for low diversity in submissions.

The generator falls back gracefully: if not enough labels above threshold, it
defers to the parent ArchiveSeededGenerator's behaviour (or raises if no
fallback seeds available).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .archive_seeded_generator import ArchiveSeededGenerator, ArchiveSeededConfig
from .filters import NanobodyValidityConfig


@dataclass
class LabelSeededConfig(ArchiveSeededConfig):
    """Same knobs as ArchiveSeededConfig + a path/threshold for the label parquet."""
    labels_path: str = "cache/offline_labels/Q9NZQ7.parquet"
    real_iiptm_threshold: float = 0.80
    # If we have fewer than min_seeds passing the threshold, fall back to the
    # archive seeds_path. Avoids running on 1-2 seeds and producing very narrow output.
    min_seeds: int = 3
    target: str = "Q9NZQ7"


def _load_label_seeds(
    labels_path: str,
    target: str,
    real_iiptm_threshold: float,
) -> list[tuple[str, float]]:
    """Read offline_labels parquet, return [(seq, iiptm), ...] above threshold."""
    if not os.path.exists(labels_path):
        return []
    try:
        df = pd.read_parquet(labels_path)
    except Exception:
        return []
    if "real_iiptm" not in df.columns or "sequence" not in df.columns:
        return []
    df = df[df.get("target", target) == target]
    df = df.dropna(subset=["real_iiptm", "sequence"])
    df = df[df["real_iiptm"] >= real_iiptm_threshold]
    if df.empty:
        return []
    # Dedupe by sequence (keep highest iiptm for each)
    df = df.sort_values("real_iiptm", ascending=False).drop_duplicates(
        subset=["sequence"], keep="first"
    )
    return [(row.sequence, float(row.real_iiptm)) for row in df.itertuples()]


class LabelSeededGenerator(ArchiveSeededGenerator):
    """ArchiveSeededGenerator but seeds come from our measured winners."""

    def __init__(
        self,
        validity_cfg: NanobodyValidityConfig,
        gen_cfg: Optional[LabelSeededConfig] = None,
        rng: Optional[random.Random] = None,
    ):
        cfg = gen_cfg or LabelSeededConfig()
        label_seeds = _load_label_seeds(
            cfg.labels_path, cfg.target, cfg.real_iiptm_threshold
        )

        if len(label_seeds) >= cfg.min_seeds:
            # Use label-derived seeds. ArchiveSeededGenerator's seed loader is
            # bypassed by passing `seeds=` explicitly.
            super().__init__(
                validity_cfg, gen_cfg=cfg, rng=rng, seeds=label_seeds,
            )
            self._seed_source = "labels"
            self._n_label_seeds = len(label_seeds)
        else:
            # Not enough measured winners yet — fall back to archive seeds.
            super().__init__(validity_cfg, gen_cfg=cfg, rng=rng)
            self._seed_source = f"archive_fallback (only {len(label_seeds)} labels >= {cfg.real_iiptm_threshold})"
            self._n_label_seeds = len(label_seeds)

    def info(self) -> str:
        return f"LabelSeededGenerator: source={self._seed_source} n_label_seeds={self._n_label_seeds}"
