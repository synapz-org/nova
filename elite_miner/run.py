"""Elite miner main loop.

For each epoch:
  1. Detect epoch boundary, fetch block hash, derive challenge (target + allowed reaction)
  2. Refresh HF archive uniqueness caches
  3. Search molecule and nanobody candidates in parallel
  4. Submit best-so-far as soon as we have it (race competitors on tiebreak)
  5. Continue refining; resubmit when rate-limit window opens AND we have a better candidate
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from typing import Any, Optional

from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import bittensor as bt
from substrateinterface import SubstrateInterface

# Direct submodule loading to avoid utils/__init__ which pulls in the heavy
# nanobody/metanano/abnumber chain. Lazy so this module imports clean even
# when the GPU/runtime deps (timelock, datasets, etc.) aren't installed.
import importlib.util as _ilu
import sys as _sys


def _load_submodule(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_quicknet_timelock():
    """Lazy load — only when actually mining (requires `timelock` package)."""
    _btdr = _load_submodule("_em_utils_btdr", os.path.join(BASE_DIR, "utils", "btdr.py"))
    return _btdr.QuicknetBittensorDrandTimelock


def _get_challenge_fn():
    """Lazy load — `datasets` import is heavy."""
    _challenge = _load_submodule("_em_utils_challenge", os.path.join(BASE_DIR, "utils", "challenge.py"))
    return _challenge.get_challenge_params_from_blockhash


def _get_upload_github_fn():
    _files = _load_submodule("_em_utils_files", os.path.join(BASE_DIR, "utils", "files.py"))
    return _files.upload_file_to_github

from elite_miner.config import parse_arguments, setup_logging, subnet_config_dict
from elite_miner.submission import build_github_path, submit, SubmissionResult
from elite_miner.timing import EpochState, SubmissionRateLimiter, get_epoch_state

from elite_miner.molecule import (
    CombinatorialSearcher,
    ValidityFilter as MolValidityFilter,
    ProxyScorer as MolProxyScorer,
    Boltz2Scorer,
    ScoredMolecule,
    parse_allowed_reaction,
    rank as mol_rank,
    warm_archive_cache as warm_mol_cache,
    is_molecule_unique,
)
from elite_miner.nanobody import (
    NanobodyFilter,
    NanobodyValidityConfig,
    NanobodyGenerator,
    ProxyNanobodyScorer,
    BoltzGenScorer,
    ScoredNanobody,
    rank as nb_rank,
    warm_archive_cache as warm_nb_cache,
    is_nanobody_unique,
)


# --------------------------------------------------------------------------- #
# Bittensor setup
# --------------------------------------------------------------------------- #

async def setup_bittensor(config) -> tuple[Any, Any, Any, int, int]:
    bt.logging.info("Setting up Bittensor objects")
    wallet = bt.wallet(config=config)
    bt.logging.info(f"wallet: {wallet}")

    async with bt.async_subtensor(network=config.network) as subtensor:
        bt.logging.info(f"connected to subtensor: {config.network}")
        metagraph = await subtensor.metagraph(config.netuid)
        await metagraph.sync()

        try:
            miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        except ValueError:
            bt.logging.error(
                f"hotkey {wallet.hotkey.ss58_address} not registered on netuid {config.netuid}"
            )
            raise

        node = SubstrateInterface(url=config.network)
        epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1
        bt.logging.info(f"miner_uid={miner_uid} epoch_length={epoch_length}")

    return wallet, subtensor, metagraph, miner_uid, epoch_length


# --------------------------------------------------------------------------- #
# Molecule track
# --------------------------------------------------------------------------- #

class MoleculeTrack:
    """Per-epoch molecule search + scoring + best-candidate tracking."""

    def __init__(self, config):
        self.config = config
        self.subnet_cfg = subnet_config_dict(config)
        self.target = config.small_molecule_target[0]  # single target for now
        self.vfilter = MolValidityFilter.from_config({
            "min_heavy_atoms": config.min_heavy_atoms,
            "min_rotatable_bonds": config.min_rotatable_bonds,
            "max_rotatable_bonds": config.max_rotatable_bonds,
            "banned_atom_types": config.banned_atom_types,
        })
        self.skip_unique = config.no_uniqueness_check
        self.use_inference = config.use_inference

        if self.use_inference:
            self.scorer = Boltz2Scorer(target=self.target)
        else:
            self.scorer = MolProxyScorer()

        self.searcher: Optional[CombinatorialSearcher] = None
        self.best: Optional[ScoredMolecule] = None
        self.candidates_scored = 0

    def reset_for_epoch(self, rxn_id: int) -> None:
        bt.logging.info(f"molecule: starting epoch with rxn_id={rxn_id} target={self.target}")
        self.searcher = CombinatorialSearcher(rxn_id)
        self.best = None
        self.candidates_scored = 0

        if not self.skip_unique:
            try:
                warm_mol_cache(self.target)
            except Exception as e:
                bt.logging.warning(f"molecule: archive cache warm failed: {e}")

    def search_one_batch(self) -> Optional[ScoredMolecule]:
        """Run one search → filter → uniqueness → score → rank batch.
        Returns the best of this batch, or None if no valid candidates survived.
        """
        if self.searcher is None:
            return None

        raw = self.searcher.generate_batch(self.config.batch_size)
        bt.logging.debug(f"molecule: sampled {len(raw)} raw candidates")

        valid = self.vfilter.filter_batch(raw)
        bt.logging.debug(f"molecule: {len(valid)}/{len(raw)} passed validity")

        if not self.skip_unique:
            valid = [(n, s) for n, s in valid if self._safe_is_unique(s)]
            bt.logging.debug(f"molecule: {len(valid)} unique vs archive")

        if not valid:
            return None

        # Real Boltz2 is expensive — only score a small top-N if using inference
        if self.use_inference:
            # In v1 we don't have a surrogate, so just take a sample to score
            # to fit the per-batch time budget
            sample_size = min(len(valid), 5)
            valid = valid[:sample_size]
            scored = self.scorer.score_batch(valid, self.subnet_cfg)
        else:
            scored = self.scorer.score_batch(valid)

        self.candidates_scored += len(scored)
        ranked = mol_rank(scored)
        return ranked[0] if ranked else None

    def update_best(self, candidate: Optional[ScoredMolecule]) -> bool:
        """Update self.best if candidate is better. Returns True if updated."""
        if candidate is None:
            return False
        if self.best is None or candidate.score > self.best.score:
            self.best = candidate
            bt.logging.info(f"molecule: new best {candidate.name} score={candidate.score:.4f}")
            return True
        return False

    def _safe_is_unique(self, smiles: str) -> bool:
        try:
            return is_molecule_unique(self.target, smiles)
        except Exception as e:
            bt.logging.warning(f"molecule: uniqueness check failed for {smiles[:40]}: {e}")
            # Be conservative — assume non-unique on error so we don't submit a likely-duplicate
            return False


# --------------------------------------------------------------------------- #
# Nanobody track
# --------------------------------------------------------------------------- #

class NanobodyTrack:
    """Per-epoch nanobody generation + scoring + best-candidate tracking."""

    def __init__(self, config):
        self.config = config
        self.subnet_cfg = subnet_config_dict(config)
        self.target = config.nanobody_target[0]
        self.validity_cfg = NanobodyValidityConfig.from_config({
            "min_sequence_length": config.min_sequence_length,
            "max_sequence_length": config.max_sequence_length,
            "min_cysteines": config.min_cysteines,
            "cys_pair_min_separation": config.cys_pair_min_separation,
            "cys_pair_max_separation": config.cys_pair_max_separation,
            "max_homopolymer_run": config.max_homopolymer_run,
            "max_di_repeat_pairs": config.max_di_repeat_pairs,
            "reject_signal_peptides": config.reject_signal_peptides,
            "sp_window": config.sp_window,
            "sp_hydro_min_in_window": config.sp_hydro_min_in_window,
            "sp_scan_prefix": config.sp_scan_prefix,
            "enforce_vhh_hallmarks": config.enforce_vhh_hallmarks,
        })
        self.generator = NanobodyGenerator(self.validity_cfg)
        self.skip_unique = config.no_uniqueness_check
        self.use_inference = config.use_inference

        if self.use_inference:
            self.scorer = BoltzGenScorer(target=self.target)
        else:
            self.scorer = ProxyNanobodyScorer()

        self.best: Optional[ScoredNanobody] = None
        self.candidates_scored = 0

    def reset_for_epoch(self) -> None:
        bt.logging.info(f"nanobody: starting epoch with target={self.target}")
        self.generator.reset_seen()
        self.best = None
        self.candidates_scored = 0

        if not self.skip_unique:
            try:
                warm_nb_cache(self.target)
            except Exception as e:
                bt.logging.warning(f"nanobody: archive cache warm failed: {e}")

    def search_one_batch(self) -> Optional[ScoredNanobody]:
        seqs = self.generator.generate_batch(self.config.batch_size)
        bt.logging.debug(f"nanobody: generated {len(seqs)} valid candidates")

        if not self.skip_unique:
            seqs = [s for s in seqs if self._safe_is_unique(s)]
            bt.logging.debug(f"nanobody: {len(seqs)} unique vs archive")

        if not seqs:
            return None

        if self.use_inference:
            sample_size = min(len(seqs), 5)
            seqs = seqs[:sample_size]
            scored = self.scorer.score_batch(seqs, self.subnet_cfg)
        else:
            scored = self.scorer.score_batch(seqs)

        self.candidates_scored += len(scored)
        ranked = nb_rank(scored)
        return ranked[0] if ranked else None

    def update_best(self, candidate: Optional[ScoredNanobody]) -> bool:
        if candidate is None:
            return False
        # Lower is better for nanobodies (boltzgen_rank_mode='min')
        if self.best is None or candidate.score < self.best.score:
            self.best = candidate
            bt.logging.info(f"nanobody: new best score={candidate.score:.4f} len={len(candidate.sequence)}")
            return True
        return False

    def _safe_is_unique(self, sequence: str) -> bool:
        try:
            return is_nanobody_unique(self.target, sequence)
        except Exception as e:
            bt.logging.warning(f"nanobody: uniqueness check failed: {e}")
            return False


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

async def run_epoch(
    state: dict,
    molecule_track: Optional[MoleculeTrack],
    nanobody_track: Optional[NanobodyTrack],
    epoch_state: EpochState,
) -> None:
    """Search + submit + refine for one epoch."""
    config = state["config"]
    subtensor = state["subtensor"]
    rate_limiter: SubmissionRateLimiter = state["rate_limiter"]

    # 1) Derive challenge params from block hash
    challenge = _get_challenge_fn()(
        block_hash=epoch_state.block_hash,
        small_molecule_target=config.small_molecule_target,
        nanobody_target=config.nanobody_target,
        include_reaction=config.random_valid_reaction,
    )
    bt.logging.info(f"epoch: challenge={challenge}")

    # 2) Pick reaction id (either from challenge or from config allowed list)
    if config.random_valid_reaction:
        rxn_id = parse_allowed_reaction(challenge.get("allowed_reaction"))
    else:
        rxn_id = parse_allowed_reaction(config.allowed_reactions[0] if config.allowed_reactions else None)
    if rxn_id is None and molecule_track is not None:
        bt.logging.warning("epoch: no valid reaction id, disabling molecule track this epoch")

    # 3) Reset tracks for the new epoch
    if molecule_track is not None and rxn_id is not None:
        molecule_track.reset_for_epoch(rxn_id)
    if nanobody_track is not None:
        nanobody_track.reset_for_epoch()

    # 4) Search loop
    submitted_at_least_once = False
    for batch_idx in range(config.max_batches_per_epoch):
        current = await subtensor.get_current_block()
        if current >= epoch_state.next_boundary - 2:
            bt.logging.info("epoch: near boundary, stopping search")
            break

        # Run molecule + nanobody batches sequentially (cheap proxy mode) or
        # let them block independently (real inference mode would parallelize via GPU)
        if molecule_track is not None and rxn_id is not None:
            try:
                cand = molecule_track.search_one_batch()
                molecule_track.update_best(cand)
            except Exception as e:
                bt.logging.error(f"epoch: molecule batch {batch_idx} failed: {e}")
                bt.logging.debug(traceback.format_exc())

        if nanobody_track is not None:
            try:
                cand = nanobody_track.search_one_batch()
                nanobody_track.update_best(cand)
            except Exception as e:
                bt.logging.error(f"epoch: nanobody batch {batch_idx} failed: {e}")
                bt.logging.debug(traceback.format_exc())

        # 5) Try to submit if we have anything and rate-limit allows
        current = await subtensor.get_current_block()
        if rate_limiter.can_submit(current) and _has_candidate(molecule_track, nanobody_track):
            result = await _do_submit(state, molecule_track, nanobody_track)
            if result.success:
                rate_limiter.record_submission(result.block_submitted or current)
                submitted_at_least_once = True

        await asyncio.sleep(0.5)

    if not submitted_at_least_once:
        bt.logging.warning("epoch: ended without successful submission")


def _has_candidate(mt: Optional[MoleculeTrack], nt: Optional[NanobodyTrack]) -> bool:
    if mt is not None and mt.best is not None:
        return True
    if nt is not None and nt.best is not None:
        return True
    return False


async def _do_submit(
    state: dict,
    mt: Optional[MoleculeTrack],
    nt: Optional[NanobodyTrack],
) -> SubmissionResult:
    mol_name = mt.best.name if (mt is not None and mt.best is not None) else None
    nb_seq = nt.best.sequence if (nt is not None and nt.best is not None) else None
    return await submit(
        subtensor=state["subtensor"],
        wallet=state["wallet"],
        netuid=state["config"].netuid,
        miner_uid=state["miner_uid"],
        bdt=state["bdt"],
        molecule_name=mol_name,
        nanobody_sequence=nb_seq,
        github_path=state["github_path"],
        upload_github_fn=_get_upload_github_fn(),
    )


async def run_miner(config) -> None:
    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor(config)

    # We re-enter the async_subtensor context for the long-running loop
    async with bt.async_subtensor(network=config.network) as subtensor:
        state = {
            "config": config,
            "wallet": wallet,
            "subtensor": subtensor,
            "metagraph": metagraph,
            "miner_uid": miner_uid,
            "epoch_length": epoch_length,
            "bdt": _get_quicknet_timelock()(),
            "github_path": build_github_path(),
            "rate_limiter": SubmissionRateLimiter(config.no_submission_blocks),
        }

        molecule_track = MoleculeTrack(config) if config.enable_molecule_track else None
        nanobody_track = NanobodyTrack(config) if config.enable_nanobody_track else None

        if molecule_track is None and nanobody_track is None:
            bt.logging.error("Both tracks disabled — nothing to mine. Exiting.")
            return

        bt.logging.info("entering main mining loop")
        last_epoch_block = -1

        while True:
            try:
                epoch_state = await get_epoch_state(subtensor, epoch_length)
                if epoch_state.last_boundary != last_epoch_block:
                    bt.logging.info(
                        f"epoch boundary: last_boundary={epoch_state.last_boundary} "
                        f"next={epoch_state.next_boundary} current={epoch_state.current_block}"
                    )
                    last_epoch_block = epoch_state.last_boundary
                    await run_epoch(state, molecule_track, nanobody_track, epoch_state)
                else:
                    # Mid-epoch: wait for the next boundary
                    blocks_remaining = epoch_state.blocks_remaining
                    if blocks_remaining > 5:
                        await asyncio.sleep(12)
                    else:
                        await asyncio.sleep(2)
            except KeyboardInterrupt:
                bt.logging.success("interrupt — exiting miner")
                break
            except asyncio.CancelledError:
                bt.logging.warning("subtensor connection cancelled, reconnecting")
                subtensor = bt.async_subtensor(network=config.network)
                await subtensor.initialize()
                state["subtensor"] = subtensor
                await asyncio.sleep(1)
            except Exception as e:
                bt.logging.error(f"main loop error: {e}")
                bt.logging.debug(traceback.format_exc())
                await asyncio.sleep(5)


async def main() -> None:
    load_dotenv()
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    asyncio.run(main())
