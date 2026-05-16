"""Append-only writer for nanobody surrogate training labels.

Every real BoltzGen call during mining or label collection produces a
(sequence, target, {metric_name: value, ...}) tuple. We store these as
parquet rows for later training.

Schema:
    sequence : str  (canonical normalized form)
    target : str    (UniProt ID)
    seq_length : int
    {metric_name} : float (one column per BoltzGen metric)
    rank_sum : float (sum-of-ranks; None if labels collected before ranking ran)
    run_id : str
    timestamp : str (ISO)

Concrete BoltzGen metrics from config/boltzgen_config.yaml:
  confidence:
    design_iiptm, design_ptm, design_to_target_iptm,
    min_design_to_target_pae, interaction_pae
  physical_interaction:
    plip_hbonds_refolded, plip_saltbridge_refolded, delta_sasa_refolded
  developability:
    liability_score, liability_num_violations
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from threading import Lock
from typing import Optional

import pandas as pd

# Mirrors keys + modes from config/boltzgen_config.yaml
BOLTZGEN_METRICS = {
    "design_iiptm": "max",
    "design_ptm": "max",
    "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min",
    "interaction_pae": "min",
    "plip_hbonds_refolded": "max",
    "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max",
    "liability_score": "min",
    "liability_num_violations": "min",
}


_LOCK = Lock()


def _streaming_run_id() -> str:
    if not hasattr(_streaming_run_id, "_id"):
        _streaming_run_id._id = "nb-stream-" + hashlib.sha256(
            dt.datetime.now(dt.timezone.utc).isoformat().encode()
        ).hexdigest()[:10]
    return _streaming_run_id._id


def append_labels(output_path: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    rid = _streaming_run_id()
    for r in rows:
        r.setdefault("timestamp", now)
        r.setdefault("run_id", rid)
    df = pd.DataFrame(rows)
    with _LOCK:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if os.path.exists(output_path):
            try:
                existing = pd.read_parquet(output_path)
                df = pd.concat([existing, df], ignore_index=True)
            except Exception:
                pass
        df.to_parquet(output_path)
    return len(rows)


def default_path(target: str, cache_dir: Optional[str] = None) -> str:
    cache_dir = cache_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "cache", "nb_labels",
    )
    return os.path.join(cache_dir, f"{target}.parquet")


def make_label_row(
    sequence: str,
    target: str,
    components: dict,
    rank_sum: Optional[float] = None,
) -> dict:
    """Build a single label row from BoltzGen components dict.

    `components` is the validator's per_nanobody_components[uid][seq][target] structure —
    a flat dict of metric_name -> float value.
    """
    row = {
        "sequence": sequence,
        "target": target,
        "seq_length": len(sequence),
    }
    for metric in BOLTZGEN_METRICS:
        v = components.get(metric)
        row[metric] = float(v) if v is not None else None
    if rank_sum is not None:
        row["rank_sum"] = float(rank_sum)
    return row
