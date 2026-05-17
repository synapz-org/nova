"""High-band-weighted nb surrogate retrain.

Pulls the full Metanova/Submission-Archive nanobody table, merges with our
local box 2 offline labels (cache/offline_labels/<target>.parquet), normalizes
everything to the box-1 nb_labels schema, and trains a new LightGBM surrogate
with sample weights iiptm**weight_exp.

The goal is to get a surrogate that's calibrated *in the high-iiptm band*,
where the current model has Spearman ~0.6 and produces too-clustered top-1
picks (real iiptm spans 0.70-0.82 from predictions in 0.819-0.825).

Output: models/<output-name>/{model.txt, holdout_metrics.json, feature_version.json}

Usage:
    python -m elite_miner.scripts.retrain_nb_surrogate_highband \
        --target Q9NZQ7 \
        --offline-labels cache/offline_labels/Q9NZQ7.parquet \
        --weight-exp 2.0 \
        --output-dir models/nb_surrogate_Q9NZQ7_hb_v1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
from huggingface_hub import hf_hub_download


# Maps box 2 offline_labels columns → box 1 nb_labels columns
OFFLINE_TO_NB = {
    "bg_design_iiptm": "design_iiptm",
    "bg_design_ptm": "design_ptm",
    "bg_design_to_target_iptm": "design_to_target_iptm",
    "bg_min_design_to_target_pae": "min_design_to_target_pae",
    "bg_interaction_pae": "interaction_pae",
    "bg_plip_hbonds_refolded": "plip_hbonds_refolded",
    "bg_plip_saltbridge_refolded": "plip_saltbridge_refolded",
    "bg_delta_sasa_refolded": "delta_sasa_refolded",
    "bg_liability_score": "liability_score",
    "bg_liability_num_violations": "liability_num_violations",
}


def fetch_archive(target: str) -> pd.DataFrame:
    """Pull the public nanobody archive for `target` from Metanova/Submission-Archive."""
    print(f"[retrain] pulling Metanova/Submission-Archive {target}_nanobodies.csv …", flush=True)
    path = hf_hub_download(
        repo_id="Metanova/Submission-Archive",
        filename=f"{target}_nanobodies.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(path)
    df["target"] = target
    df["seq_length"] = df["sequence"].str.len()
    print(f"[retrain] archive: {len(df)} rows", flush=True)
    return df


def normalize_offline(df_offline: pd.DataFrame, target: str) -> pd.DataFrame:
    """Convert box 2's offline_labels schema to box 1's nb_labels schema."""
    df = df_offline[df_offline["target"] == target].copy()
    df = df.rename(columns=OFFLINE_TO_NB)
    df["seq_length"] = df["sequence"].str.len()
    # nb_labels also has run_id; tag offline rows so we can audit later.
    if "run_id" not in df.columns:
        df["run_id"] = "offline_" + df.get("machine_id", "").astype(str)
    # Drop the offline-only columns to align schemas
    drop_cols = ["ts", "machine_id", "predicted_iiptm", "min_muts", "max_muts",
                 "batch_size", "elapsed_s"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def merge_and_save(target: str, offline_path: str | None, out_path: str) -> int:
    """Pull archive + load offline, merge, dedupe by sequence_hash, save."""
    archive = fetch_archive(target)
    if offline_path and os.path.exists(offline_path):
        offline_raw = pd.read_parquet(offline_path)
        offline_norm = normalize_offline(offline_raw, target)
        print(f"[retrain] offline: {len(offline_norm)} rows after schema normalize", flush=True)
        combined = pd.concat([archive, offline_norm], ignore_index=True)
    else:
        print(f"[retrain] no offline file at {offline_path}; archive only", flush=True)
        combined = archive

    # Dedupe by sequence (newest wins — drop earlier duplicates)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["sequence"], keep="last").reset_index(drop=True)
    print(f"[retrain] after dedupe: {len(combined)} rows (dropped {before - len(combined)} dupes)", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"[retrain] wrote merged labels to {out_path}", flush=True)
    return len(combined)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Q9NZQ7")
    parser.add_argument("--offline-labels", default="cache/offline_labels/Q9NZQ7.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weight-exp", type=float, default=2.0)
    parser.add_argument("--high-band-cutoff", type=float, default=0.78)
    parser.add_argument("--keep-merged", action="store_true",
                        help="Don't delete the merged labels file after training")
    args = parser.parse_args()

    # 1) Merge archive + offline labels into a temporary nb_labels-shaped parquet
    with tempfile.TemporaryDirectory(prefix="retrain_") as tmp:
        merged_path = os.path.join(tmp, f"{args.target}_merged.parquet")
        n_total = merge_and_save(args.target, args.offline_labels, merged_path)
        if n_total < 100:
            print(f"[retrain] only {n_total} rows after merge — too few to train meaningfully", flush=True)
            return 1

        # 2) Run train_nb_surrogate with the merged labels and weight_exp
        # Glob includes the directory so it picks up only our merged file.
        labels_glob = merged_path
        cmd = [
            sys.executable, "-m", "elite_miner.scripts.train_nb_surrogate",
            "--labels-glob", labels_glob,
            "--target", args.target,
            "--output-dir", args.output_dir,
            "--weight-exp", str(args.weight_exp),
            "--high-band-cutoff", str(args.high_band_cutoff),
        ]
        print(f"[retrain] running: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=BASE_DIR)
        if rc != 0:
            print(f"[retrain] train_nb_surrogate exited with code {rc}", flush=True)
            return rc

        if args.keep_merged:
            # Save the merged labels alongside the model for reproducibility
            import shutil
            keep_path = os.path.join(args.output_dir, "merged_training_labels.parquet")
            shutil.copy(merged_path, keep_path)
            print(f"[retrain] saved merged labels to {keep_path}", flush=True)

    print(f"[retrain] DONE. Output dir: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
