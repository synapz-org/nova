"""Append-only queue of our chain-submitted nb sequences.

Purpose: ground-truth measurement loop. We submit a sequence to chain (surrogate's
top pick), then asynchronously run real BoltzGen on it and record the actual
design_iiptm. Lets us answer "did the n=5000 batch boost translate to real iiptm?"
without waiting for the public archive to surface our sequences.

Storage: JSONL. One line per submission. Append-only, cheap to tail.

Format:
    {
      "ts": "2026-05-17T12:19:38Z",
      "epoch": 22787,
      "landed_block": 8203401,
      "target": "Q9NZQ7",
      "sequence": "DVTLAES...",
      "predicted_iiptm": 0.8253,
      "generator": "ArchiveSeededGenerator",
      "fast_batch_size": 5000
    }

Consumers (the BoltzGen labeling worker, analysis scripts) follow file position
via a sidecar `.cursor` file; new entries appended after their last cursor are
fresh work.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Optional


DEFAULT_PATH = "cache/submission_queue/nb.jsonl"


def append(
    *,
    target: str,
    sequence: str,
    predicted_iiptm: float,
    epoch: Optional[int] = None,
    landed_block: Optional[int] = None,
    generator: Optional[str] = None,
    fast_batch_size: Optional[int] = None,
    variant_id: Optional[str] = None,
    path: str = DEFAULT_PATH,
) -> None:
    """Append one nb submission record. Best-effort; never raises out of process.

    The miner calls this on the hot path immediately after a successful chain
    commit, so we swallow IO errors rather than risk crashing the submission loop.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        row = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "epoch": epoch,
            "landed_block": landed_block,
            "target": target,
            "sequence": sequence,
            "predicted_iiptm": float(predicted_iiptm),
            "generator": generator,
            "fast_batch_size": fast_batch_size,
            "variant_id": variant_id,
        }
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        # Never let queue IO break the submission path.
        pass
