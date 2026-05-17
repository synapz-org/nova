"""Background worker: real BoltzGen on each nb sequence we submitted to chain.

Reads `cache/submission_queue/nb.jsonl` (written by `submission_queue.append`),
runs BoltzGen on each new sequence, appends a row to a separate labels file
keyed by the queue position so we can join.

Runs forever. Designed to share a GPU with the live miner — the miner uses
surrogate-only when `--nb_disable_inference` is set, so the worker has BoltzGen
to itself.

Output:
    cache/submission_labels/Q9NZQ7.parquet
        columns: queue_ts, sequence, predicted_iiptm, real_iiptm, fast_batch_size, ...

Usage:
    python -m elite_miner.scripts.label_submissions_worker \
        --queue cache/submission_queue/nb.jsonl \
        --output-dir cache/submission_labels \
        --poll-seconds 60

Stop cleanly with SIGTERM. State (cursor position) is the line count in the
queue file recorded in `<queue>.cursor`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import sys
import time
from typing import Optional

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Heavy imports deferred until first label run so the worker starts fast.
_scorer_cache: dict = {}


def _stop_flag() -> dict:
    state = {"stop": False}
    def handler(signum, frame):
        state["stop"] = True
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    return state


def _read_cursor(cursor_path: str) -> int:
    if not os.path.exists(cursor_path):
        return 0
    try:
        with open(cursor_path) as f:
            return int(f.read().strip() or "0")
    except Exception:
        return 0


def _write_cursor(cursor_path: str, pos: int) -> None:
    os.makedirs(os.path.dirname(cursor_path) or ".", exist_ok=True)
    tmp = cursor_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(pos))
    os.replace(tmp, cursor_path)


def _read_new_entries(queue_path: str, cursor: int) -> tuple[list[dict], int]:
    """Read lines from cursor onward. Returns (entries, new_cursor)."""
    if not os.path.exists(queue_path):
        return [], cursor
    entries: list[dict] = []
    new_cursor = cursor
    with open(queue_path) as f:
        for i, line in enumerate(f):
            if i < cursor:
                continue
            line = line.strip()
            if not line:
                new_cursor = i + 1
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                # Skip malformed line, advance cursor
                pass
            new_cursor = i + 1
    return entries, new_cursor


def _get_scorer(target: str, clip_interval):
    """Lazy-construct a BoltzGenScorer per target."""
    if target in _scorer_cache:
        return _scorer_cache[target]
    from elite_miner.nanobody.scorer import BoltzGenScorer
    sc = BoltzGenScorer(target=target, clip_interval=clip_interval, stream_labels=False)
    _scorer_cache[target] = sc
    return sc


def _subnet_config_for(target: str, clip_interval):
    """Minimal subnet config dict BoltzGenScorer expects."""
    return {
        "nanobody_target": [target],
        "nanobody_target_clip_interval": [clip_interval],
    }


def _output_path(output_dir: str, target: str) -> str:
    return os.path.join(output_dir, f"{target}.parquet")


def _already_labeled(output_path: str, target: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    try:
        df = pd.read_parquet(output_path, columns=["sequence"])
        return set(df["sequence"].tolist())
    except Exception:
        return set()


def _append_label_rows(output_path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    new_df = pd.DataFrame(rows)
    if os.path.exists(output_path):
        try:
            existing = pd.read_parquet(output_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        except Exception:
            pass
    new_df.to_parquet(output_path, index=False)


def label_one(entry: dict, clip_intervals: dict, output_dir: str) -> Optional[dict]:
    """Run BoltzGen on one queue entry. Returns label row, or None on failure."""
    target = entry.get("target")
    seq = entry.get("sequence")
    if not target or not seq:
        return None
    clip = clip_intervals.get(target)
    if clip is None:
        print(f"[worker] no clip interval for {target}, skipping", flush=True)
        return None

    out_path = _output_path(output_dir, target)
    if seq in _already_labeled(out_path, target):
        return None  # already done, just advance cursor

    scorer = _get_scorer(target, clip)
    subnet_cfg = _subnet_config_for(target, clip)

    # Clean BoltzGen's tmp inputs/ + outputs/ before each call. Wrapper accumulates
    # yaml input files and intermediate designs across calls, which makes each
    # subsequent label process all prior sequences and grow latency unboundedly.
    # See kb/gotchas/install-deps-uv-not-in-path.md for context; same wrapper bug.
    import shutil
    bg_tmp = os.path.join(BASE_DIR, "external_tools", "boltzgen", "boltzgen_tmp_files")
    for sub in ("inputs", "outputs"):
        p = os.path.join(bg_tmp, sub)
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
        os.makedirs(p, exist_ok=True)

    t0 = time.time()
    try:
        scored = scorer.score_batch([seq], subnet_cfg)
    except Exception as e:
        print(f"[worker] BoltzGen failed for seq={seq[:30]}…: {e}", flush=True)
        return None
    elapsed = time.time() - t0
    if not scored:
        return None
    s = scored[0]
    raw = s.raw or {}
    row = {
        "queue_ts": entry.get("ts"),
        "epoch": entry.get("epoch"),
        "landed_block": entry.get("landed_block"),
        "target": target,
        "sequence": seq,
        "predicted_iiptm": entry.get("predicted_iiptm"),
        "real_iiptm": raw.get("design_iiptm"),
        "fast_batch_size": entry.get("fast_batch_size"),
        "generator": entry.get("generator"),
        "elapsed_s": elapsed,
        "labeled_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    # Carry through every metric BoltzGen returned for completeness
    for k, v in raw.items():
        if k not in row:
            row[f"bg_{k}"] = v
    _append_label_rows(out_path, [row])
    pred = entry.get("predicted_iiptm")
    real = raw.get("design_iiptm")
    pred_str = f"{pred:.4f}" if pred is not None else "n/a"
    real_str = f"{real:.4f}" if real is not None else "n/a"
    print(
        f"[worker] labeled epoch={entry.get('epoch')} pred={pred_str} "
        f"real={real_str} ({elapsed:.0f}s)",
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="cache/submission_queue/nb.jsonl")
    parser.add_argument("--output-dir", default="cache/submission_labels")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--clip-interval",
        action="append",
        default=[],
        help="target:start,end pairs e.g. Q9NZQ7:17,132. Repeatable.",
    )
    args = parser.parse_args()

    # Default clip interval matches config.yaml for Q9NZQ7
    clip_intervals = {"Q9NZQ7": [17, 132]}
    for spec in args.clip_interval:
        try:
            tgt, rng = spec.split(":")
            a, b = rng.split(",")
            clip_intervals[tgt] = [int(a), int(b)]
        except Exception:
            print(f"[worker] bad --clip-interval spec: {spec}", flush=True)

    cursor_path = args.queue + ".cursor"
    stop = _stop_flag()
    print(f"[worker] starting. queue={args.queue} output={args.output_dir}", flush=True)

    while not stop["stop"]:
        cursor = _read_cursor(cursor_path)
        entries, new_cursor = _read_new_entries(args.queue, cursor)
        if entries:
            print(f"[worker] {len(entries)} new entries (cursor {cursor} → {new_cursor})", flush=True)
        for entry in entries:
            if stop["stop"]:
                break
            label_one(entry, clip_intervals, args.output_dir)
            # Advance cursor incrementally so a crash doesn't redo work
            _write_cursor(cursor_path, cursor + 1)
            cursor += 1
        if not entries:
            # idle wait, but stay responsive to signals
            for _ in range(args.poll_seconds):
                if stop["stop"]:
                    break
                time.sleep(1)

    print("[worker] shutdown clean", flush=True)


if __name__ == "__main__":
    main()
