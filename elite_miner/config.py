"""Argparse + config loading for elite miner.

Wraps config/config_loader.load_config and adds miner-specific knobs.
"""

from __future__ import annotations

import argparse
import os

import bittensor as bt

from config.config_loader import load_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NOVA SN68 elite miner")
    parser.add_argument("--network", default=os.getenv("SUBTENSOR_NETWORK"),
                        help="Subtensor network (e.g. finney, test, local)")
    parser.add_argument("--netuid", type=int, default=68,
                        help="Subnet UID (68=mainnet, 379=testnet)")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Candidates per search batch per track")
    parser.add_argument("--max_batches_per_epoch", type=int, default=20,
                        help="Search loop iterations per epoch (per track)")
    parser.add_argument("--use_inference", action="store_true",
                        help="Use real Boltz2/BoltzGen scoring (requires GPU). Default: proxy scoring.")
    parser.add_argument("--surrogate_model_dir", default=None,
                        help="Path to trained LightGBM surrogate. If unset, proxy scoring is used.")
    parser.add_argument("--surrogate_batch_size", type=int, default=5000,
                        help="Candidate pool size per batch when surrogate is active "
                             "(surrogate inference is cheap, so we widen).")
    parser.add_argument("--surrogate_topk", type=int, default=5,
                        help="Top-K from surrogate to send to real Boltz2 when --use_inference.")
    parser.add_argument("--surrogate_min_spearman", type=float, default=0.0,
                        help="Refuse a surrogate model with holdout spearman below this threshold.")
    parser.add_argument("--nb_surrogate_model_dir", default=None,
                        help="Path to trained nanobody LightGBM surrogate (Phase 3). Falls back to proxy.")
    parser.add_argument("--nb_surrogate_batch_size", type=int, default=1000,
                        help="Candidate pool size per batch when nanobody surrogate is active.")
    parser.add_argument("--nb_surrogate_topk", type=int, default=5,
                        help="Top-K from nb surrogate to send to real BoltzGen when --use_inference.")
    parser.add_argument("--nb_surrogate_min_spearman", type=float, default=0.0,
                        help="Refuse a nb surrogate model with holdout spearman below this threshold.")
    parser.add_argument("--nb_disable_inference", action="store_true",
                        help="Skip BoltzGen real inference for nanobody track (use surrogate only). "
                             "BoltzGen takes ~hours per epoch and blocks new-epoch submissions.")
    parser.add_argument("--fast_batch_size_nb", type=int, default=5000,
                        help="Fast-phase surrogate-only nb candidate pool. Larger = better top-1 "
                             "prediction (5000 → ~0.82 vs 0.80 at n=20); ~0.6s on A100.")
    parser.add_argument("--fast_batch_size_mol", type=int, default=500,
                        help="Fast-phase surrogate-only mol candidate pool. Mol scoring is slower "
                             "(~3s/1000), top-1 plateaus around n=500.")
    parser.add_argument("--disable_molecule_track", action="store_true",
                        help="Disable molecule track (default: enabled)")
    parser.add_argument("--disable_nanobody_track", action="store_true",
                        help="Disable nanobody track (default: enabled)")
    parser.add_argument("--no_uniqueness_check", action="store_true",
                        help="Skip HF archive uniqueness check (faster, but may submit duplicates)")
    parser.add_argument("--log_level", default="info", choices=["debug", "info", "warning", "error"],
                        help="bt.logging level")

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    config = bt.config(parser)
    subnet_cfg = load_config()
    # bt.config is a Munch (dict-like). Use update() so dict access also works.
    config.update(subnet_cfg)

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/elite_miner".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
        )
    )
    os.makedirs(config.full_path, exist_ok=True)
    return config


def setup_logging(config: argparse.Namespace) -> None:
    bt.logging(config=config, logging_dir=config.full_path)
    level = config.log_level.lower()
    if level == "debug":
        bt.logging.set_debug(True)
    bt.logging.info(f"elite miner starting on netuid={config.netuid} network={config.network}")


def subnet_config_dict(config: argparse.Namespace) -> dict:
    """Extract the subnet config subset that Boltz2Scorer / BoltzGenScorer expect."""
    return {
        "small_molecule_target": config.small_molecule_target,
        "small_molecule_target_clip_interval": config.small_molecule_target_clip_interval,
        "nanobody_target": config.nanobody_target,
        "nanobody_target_clip_interval": config.nanobody_target_clip_interval,
        "boltz_metric": config.boltz_metric,
        "combination_strategy": config.combination_strategy,
        "boltz_mode": config.boltz_mode,
        "boltzgen_rank_mode": config.boltzgen_rank_mode,
        "boltzgen_rank_by": config.boltzgen_rank_by,
        "num_molecules": config.num_molecules,
        "num_sequences": config.num_sequences,
    }
