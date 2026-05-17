"""Offline label collector: generate candidates, run BoltzGen, write labels.

Independent of the live miner — just produces (sequence, real_iiptm, surrogate_pred,
all_boltzgen_metrics) rows so we can:
- Retrain a high-band-accurate nb surrogate
- Build a feedback-loop generator that mutates from empirically-good seeds
- Compare variants without waiting on the live submission cadence

Generates candidates from ArchiveSeededGenerator (default) or any other
generator the caller picks, surrogate-ranks them, then runs real BoltzGen on
top-K. Loops forever. Designed to run on a dedicated GPU box alongside
(not replacing) the live miner.

Usage:
    python -m elite_miner.scripts.offline_label_collector \
        --target Q9NZQ7 \
        --target-clip 17,132 \
        --topk 1 \
        --batch-size 5000 \
        --min-muts 1 --max-muts 3 \
        --output cache/offline_labels/Q9NZQ7.parquet \
        --machine-id boxB

Stop cleanly with SIGTERM. Idempotent — already-labeled sequences are skipped.
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


def _stop_flag() -> dict:
    state = {"stop": False}
    def handler(signum, frame):
        state["stop"] = True
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    return state


def already_labeled(output_path: str, target: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    try:
        df = pd.read_parquet(output_path, columns=["sequence"])
        return set(df["sequence"].tolist())
    except Exception:
        return set()


def append_rows(output_path: str, rows: list[dict]) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Q9NZQ7")
    parser.add_argument("--target-clip", default="17,132", help="comma-separated start,end")
    parser.add_argument("--topk", type=int, default=1, help="how many surrogate-top candidates to label per outer iteration")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--min-muts", type=int, default=1)
    parser.add_argument("--max-muts", type=int, default=3)
    parser.add_argument("--output", default="cache/offline_labels/Q9NZQ7.parquet")
    parser.add_argument("--surrogate-model-dir", default="models/nb_surrogate_Q9NZQ7_archive")
    parser.add_argument("--pssm-path", default="cache/q9nzq7_pssm.json")
    parser.add_argument("--seeds-path", default="cache/q9nzq7_top_seeds.json")
    parser.add_argument("--machine-id", default="boxA", help="tag rows with which box produced them")
    args = parser.parse_args()

    a, b = args.target_clip.split(",")
    clip_interval = [int(a), int(b)]

    print(f"offline label collector starting:", flush=True)
    print(f"  target={args.target} clip={clip_interval}", flush=True)
    print(f"  generator: ArchiveSeededGenerator m={args.min_muts}-{args.max_muts}", flush=True)
    print(f"  batch_size={args.batch_size} topk={args.topk}", flush=True)
    print(f"  output={args.output} machine_id={args.machine_id}", flush=True)
    print(flush=True)

    # Build generator and surrogate
    from elite_miner.nanobody.archive_seeded_generator import (
        ArchiveSeededGenerator, ArchiveSeededConfig,
    )
    from elite_miner.nanobody.filters import NanobodyValidityConfig
    from elite_miner.nanobody.surrogate import NanobodySurrogateScorer
    from elite_miner.nanobody.scorer import rank as nb_rank, BoltzGenScorer
    from elite_miner.protein.features import target_features as protein_target_features

    vcfg = NanobodyValidityConfig()
    gen = ArchiveSeededGenerator(
        vcfg,
        gen_cfg=ArchiveSeededConfig(
            min_mutations=args.min_muts,
            max_mutations=args.max_muts,
            pssm_path=args.pssm_path,
            seeds_path=args.seeds_path,
        ),
    )
    try:
        tgt_feats = protein_target_features(args.target)
    except Exception as e:
        print(f"warning: target_features({args.target}) failed: {e}", flush=True)
        tgt_feats = None
    surrogate = NanobodySurrogateScorer(
        model_dir=args.surrogate_model_dir,
        target_features=tgt_feats,
        min_spearman=0.0,
    )
    print(f"surrogate ready: {surrogate.is_ready}", flush=True)

    scorer = BoltzGenScorer(target=args.target, clip_interval=clip_interval, stream_labels=False)
    subnet_cfg = {
        "nanobody_target": [args.target],
        "nanobody_target_clip_interval": [clip_interval],
    }

    stop = _stop_flag()
    iteration = 0
    while not stop["stop"]:
        iteration += 1
        seen_seqs = already_labeled(args.output, args.target)

        # Generate, surrogate-rank, pick top-K not already labeled
        t0 = time.time()
        seqs = gen.generate_batch(args.batch_size)
        t_gen = time.time() - t0
        if not seqs:
            print(f"[iter {iteration}] generator returned no sequences, sleeping 30s", flush=True)
            for _ in range(30):
                if stop["stop"]:
                    break
                time.sleep(1)
            continue

        t0 = time.time()
        scored = surrogate.score_batch(seqs)
        t_score = time.time() - t0
        ranked = nb_rank(scored)
        # ranked is ascending — best is index 0 (lowest score = highest predicted iiptm)
        picks = []
        for r in ranked:
            if r.sequence not in seen_seqs:
                picks.append(r)
            if len(picks) >= args.topk:
                break

        if not picks:
            print(f"[iter {iteration}] all top-K already labeled, regenerating", flush=True)
            continue

        top_pred = (picks[0].raw or {}).get("surrogate_pred", 0.0)
        print(
            f"[iter {iteration}] gen={t_gen:.1f}s score={t_score:.1f}s "
            f"top_pred_iiptm={top_pred:.4f}",
            flush=True,
        )

        # Clean BoltzGen's tmp INPUT and OUTPUT dirs before each outer iteration.
        # Its wrapper accumulates yaml input files in `inputs/` and intermediate
        # designs in `outputs/` across calls, so the next call re-processes ALL
        # prior sequences (1 → 3 → 5 → 10 …) and latency grows unboundedly.
        # config/boltzgen_config.yaml has remove_files=true but the wrapper
        # doesn't honor it.
        import shutil
        bg_tmp = os.path.join(BASE_DIR, "external_tools", "boltzgen", "boltzgen_tmp_files")
        for sub in ("inputs", "outputs"):
            p = os.path.join(bg_tmp, sub)
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)
            # Recreate the directory — the wrapper's _write_yaml_files uses
            # open(path, "w") which doesn't auto-create parent dirs.
            os.makedirs(p, exist_ok=True)

        for i, pick in enumerate(picks):
            if stop["stop"]:
                break
            seq = pick.sequence
            pred = pick.raw.get("surrogate_pred", 0.0) if pick.raw else 0.0
            t0 = time.time()
            try:
                scored_real = scorer.score_batch([seq], subnet_cfg)
            except Exception as e:
                print(f"  [pick {i}] BoltzGen failed: {e}", flush=True)
                continue
            elapsed = time.time() - t0
            if not scored_real:
                continue
            s = scored_real[0]
            raw = s.raw or {}
            real_iiptm = raw.get("design_iiptm")
            real_str = f"{real_iiptm:.4f}" if real_iiptm is not None else "n/a"
            print(
                f"  [pick {i}] pred={pred:.4f} real={real_str} ({elapsed:.0f}s)",
                flush=True,
            )

            row = {
                "ts": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                "machine_id": args.machine_id,
                "target": args.target,
                "sequence": seq,
                "predicted_iiptm": pred,
                "real_iiptm": real_iiptm,
                "min_muts": args.min_muts,
                "max_muts": args.max_muts,
                "batch_size": args.batch_size,
                "elapsed_s": elapsed,
            }
            for k, v in raw.items():
                if k not in row:
                    row[f"bg_{k}"] = v
            append_rows(args.output, [row])

    print("offline label collector stopping cleanly", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
