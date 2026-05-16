"""Append-only writer for surrogate training labels.

Every real Boltz2 call during mining produces (smiles, target, pb, pv) tuples.
We're already paying for those scores; capturing them as future surrogate
training data is free.

Schema must match elite_miner/scripts/collect_labels.py output so that
train_surrogate.py can load everything from cache/labels/*.parquet uniformly.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from threading import Lock
from typing import Optional

import pandas as pd


_LOCK = Lock()


def _streaming_run_id() -> str:
    if not hasattr(_streaming_run_id, "_id"):
        _streaming_run_id._id = "stream-" + hashlib.sha256(
            dt.datetime.now(dt.timezone.utc).isoformat().encode()
        ).hexdigest()[:10]
    return _streaming_run_id._id


def append_labels(
    output_path: str,
    rows: list[dict],
) -> int:
    """Append rows to a parquet file. Thread-safe.

    Each row should have: smiles, name, heavy_atoms, target, pb, pv, raw_score.
    timestamp + run_id are added here if missing.

    Returns number of rows written.
    """
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
                # Existing file is corrupt — overwrite with new data
                pass
        df.to_parquet(output_path)
    return len(rows)


def default_path(target: str, cache_dir: Optional[str] = None) -> str:
    cache_dir = cache_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "cache", "labels",
    )
    return os.path.join(cache_dir, f"{target}.parquet")
