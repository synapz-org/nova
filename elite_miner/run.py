"""Elite miner main loop.

For each epoch:
  1. Detect epoch boundary; fetch block hash; derive challenge (target + allowed reaction)
  2. Refresh HF archive uniqueness caches (warm cache)
  3. FAST-SUBMIT: one tiny search batch → submit immediately to claim block_submitted
     tiebreak before competitors. The reference miner submits within ~1 block of boundary;
     so must we.
  4. REFINE: more search batches in a thread (doesn't block event loop). Resubmit when
     rate-limit window opens AND we have a better candidate.
  5. FALLBACK: if nothing valid found by mid-epoch, submit a known-good template.
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
import traceback
from typing import Any, Optional

from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import bittensor as bt
from substrateinterface import SubstrateInterface

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
    list_templates,
)


# --------------------------------------------------------------------------- #
# Lazy loaders for heavy `utils.*` submodules.
# --------------------------------------------------------------------------- #

import importlib.util as _ilu
import sys as _sys


def _load_submodule(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_quicknet_timelock():
    _btdr = _load_submodule("_em_utils_btdr", os.path.join(BASE_DIR, "utils", "btdr.py"))
    return _btdr.QuicknetBittensorDrandTimelock


def _get_challenge_fn():
    _challenge = _load_submodule("_em_utils_challenge", os.path.join(BASE_DIR, "utils", "challenge.py"))
    return _challenge.get_challenge_params_from_blockhash


def _get_upload_github_fn():
    _files = _load_submodule("_em_utils_files", os.path.join(BASE_DIR, "utils", "files.py"))
    return _files.upload_file_to_github


# --------------------------------------------------------------------------- #
# Bittensor setup
# --------------------------------------------------------------------------- #

async def setup_bittensor(config) -> tuple[Any, int, int]:
    bt.logging.info("Setting up Bittensor objects")
    wallet = bt.wallet(config=config)
    bt.logging.info(f"wallet: {wallet}")

    async with bt.async_subtensor(network=config.network) as subtensor:
        metagraph = await subtensor.metagraph(config.netuid)
        await metagraph.sync()
        try:
            miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        except ValueError:
            raise RuntimeError(
                f"hotkey {wallet.hotkey.ss58_address} not registered on netuid {config.netuid}"
            )
        node = SubstrateInterface(url=config.network)
        epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1
        bt.logging.info(f"miner_uid={miner_uid} epoch_length={epoch_length}")
    return wallet, miner_uid, epoch_length


# --------------------------------------------------------------------------- #
# Molecule track
# --------------------------------------------------------------------------- #

class MoleculeTrack:
    """Per-epoch molecule search + scoring + best-candidate tracking."""

    def __init__(self, config):
        self.config = config
        self.subnet_cfg = subnet_config_dict(config)
        # Use ALL targets the validator scores against
        self.targets: list[str] = list(config.small_molecule_target)
        self.vfilter = MolValidityFilter.from_config({
            "min_heavy_atoms": config.min_heavy_atoms,
            "min_rotatable_bonds": config.min_rotatable_bonds,
            "max_rotatable_bonds": config.max_rotatable_bonds,
            "banned_atom_types": config.banned_atom_types,
        })
        self.skip_unique = config.no_uniqueness_check
        self.use_inference = config.use_inference
        # In inference mode we score against the *primary* target (first one).
        # Multi-target scoring requires multiple runs and a combine step which we
        # defer to v2.
        if self.use_inference:
            self.scorer = Boltz2Scorer(target=self.targets[0])
        else:
            self.scorer = MolProxyScorer()
        self.searcher: Optional[CombinatorialSearcher] = None
        self.best: Optional[ScoredMolecule] = None
        self.candidates_scored = 0
        self.proxy = MolProxyScorer()  # always available for top-N pre-filter

    def reset_for_epoch(self, rxn_id: int) -> None:
        bt.logging.info(f"molecule: epoch start rxn={rxn_id} targets={self.targets}")
        self.searcher = CombinatorialSearcher(rxn_id)
        self.best = None
        self.candidates_scored = 0
        if not self.skip_unique:
            for tgt in self.targets:
                try:
                    warm_mol_cache(tgt)
                except Exception as e:
                    bt.logging.warning(f"molecule: archive cache warm failed for {tgt}: {e}")

    def _safe_is_unique_all_targets(self, smiles: str) -> bool:
        try:
            return all(is_molecule_unique(t, smiles) for t in self.targets)
        except Exception as e:
            bt.logging.warning(f"molecule: uniqueness check error: {e}")
            return False  # conservative

    def search_one_batch(self, batch_size: Optional[int] = None) -> Optional[ScoredMolecule]:
        """Search → filter → uniqueness → score → rank. Returns best of this batch."""
        if self.searcher is None:
            return None
        size = batch_size if batch_size is not None else self.config.batch_size

        raw = self.searcher.generate_batch(size)
        valid = self.vfilter.filter_batch(raw)
        if not self.skip_unique and valid:
            valid = [(n, s) for n, s in valid if self._safe_is_unique_all_targets(s)]
        if not valid:
            return None

        if self.use_inference:
            # Pre-rank by cheap proxy, then run real Boltz2 on top 5 only.
            proxy_scored = self.proxy.score_batch(valid)
            top = mol_rank(proxy_scored)[:5]
            candidates = [(c.name, c.smiles) for c in top]
            scored = self.scorer.score_batch(candidates, self.subnet_cfg)
        else:
            scored = self.proxy.score_batch(valid)

        self.candidates_scored += len(scored)
        ranked = mol_rank(scored)
        return ranked[0] if ranked else None

    def update_best(self, candidate: Optional[ScoredMolecule]) -> bool:
        if candidate is None:
            return False
        if self.best is None or candidate.score > self.best.score:
            self.best = candidate
            bt.logging.info(f"molecule: new best {candidate.name} score={candidate.score:.4f}")
            return True
        return False

    def fallback_candidate(self) -> Optional[ScoredMolecule]:
        """Emergency: when nothing valid found, sample DB at random for one shot.
        Returns the first reaction product that passes validity (no uniqueness).
        Better to submit *something* than nothing — reference miner submits even less."""
        if self.searcher is None:
            return None
        for _ in range(50):
            raw = self.searcher.generate_batch(20)
            valid = self.vfilter.filter_batch(raw)
            if valid:
                # Pick first valid; score with proxy
                name, smiles = valid[0]
                return self.proxy.score(name, smiles)
        return None


# --------------------------------------------------------------------------- #
# Nanobody track
# --------------------------------------------------------------------------- #

class NanobodyTrack:
    def __init__(self, config):
        self.config = config
        self.subnet_cfg = subnet_config_dict(config)
        self.targets: list[str] = list(config.nanobody_target)
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
            self.scorer = BoltzGenScorer(target=self.targets[0])
        else:
            self.scorer = ProxyNanobodyScorer()
        self.proxy = ProxyNanobodyScorer()  # always available
        self.best: Optional[ScoredNanobody] = None
        self.candidates_scored = 0

    def reset_for_epoch(self) -> None:
        bt.logging.info(f"nanobody: epoch start targets={self.targets}")
        self.generator.reset_seen()
        self.best = None
        self.candidates_scored = 0
        if not self.skip_unique:
            for tgt in self.targets:
                try:
                    warm_nb_cache(tgt)
                except Exception as e:
                    bt.logging.warning(f"nanobody: archive cache warm failed for {tgt}: {e}")

    def _safe_is_unique_all_targets(self, seq: str) -> bool:
        try:
            return all(is_nanobody_unique(t, seq) for t in self.targets)
        except Exception as e:
            bt.logging.warning(f"nanobody: uniqueness check error: {e}")
            return False

    def search_one_batch(self, batch_size: Optional[int] = None) -> Optional[ScoredNanobody]:
        size = batch_size if batch_size is not None else self.config.batch_size
        seqs = self.generator.generate_batch(size)
        if not self.skip_unique and seqs:
            seqs = [s for s in seqs if self._safe_is_unique_all_targets(s)]
        if not seqs:
            return None

        if self.use_inference:
            proxy_scored = self.proxy.score_batch(seqs)
            top = nb_rank(proxy_scored)[:5]
            top_seqs = [c.sequence for c in top]
            scored = self.scorer.score_batch(top_seqs, self.subnet_cfg)
        else:
            scored = self.proxy.score_batch(seqs)
        self.candidates_scored += len(scored)
        ranked = nb_rank(scored)
        return ranked[0] if ranked else None

    def update_best(self, candidate: Optional[ScoredNanobody]) -> bool:
        if candidate is None:
            return False
        if self.best is None or candidate.score < self.best.score:
            self.best = candidate
            bt.logging.info(f"nanobody: new best score={candidate.score:.4f} len={len(candidate.sequence)}")
            return True
        return False

    def fallback_candidate(self) -> Optional[ScoredNanobody]:
        """Emergency: mutate a known-good template once. Always returns something."""
        for _ in range(10):
            seq = self.generator.generate_one()
            if seq is not None:
                return self.proxy.score(seq)
        # Last resort: a literal template (will fail uniqueness if anyone else used it,
        # but reference miner uses it too — at least we get *some* submission)
        templates = list_templates()
        if templates:
            return self.proxy.score(templates[0].sequence)
        return None


# --------------------------------------------------------------------------- #
# Submission helper
# --------------------------------------------------------------------------- #

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


async def _try_submit(state, mt, nt) -> bool:
    """Submit if rate-limit allows AND we have something. Updates rate-limiter on success.
    Returns True if a successful chain commit landed."""
    rl: SubmissionRateLimiter = state["rate_limiter"]
    current = await state["subtensor"].get_current_block()
    if not rl.can_submit(current):
        return False
    if not _has_candidate(mt, nt):
        return False
    result = await _do_submit(state, mt, nt)
    if not result.success:
        return False
    # After awaiting set_commitment, re-read block — the extrinsic may have landed later.
    landed_block = await state["subtensor"].get_current_block()
    rl.record_submission(landed_block)
    return True


# --------------------------------------------------------------------------- #
# Main epoch logic
# --------------------------------------------------------------------------- #

async def run_epoch(
    state: dict,
    molecule_track: Optional[MoleculeTrack],
    nanobody_track: Optional[NanobodyTrack],
    epoch_state: EpochState,
) -> None:
    """Submit-early-then-refine pattern.

    Phase 1 (fast): one tiny batch per track → submit ASAP.
    Phase 2 (refine): full batches until rate-limit allows resubmit, repeat.
    Phase 3 (fallback): if nothing valid found, submit a guaranteed-valid template.
    """
    config = state["config"]
    subtensor = state["subtensor"]

    # 1) Challenge params from block hash
    challenge = _get_challenge_fn()(
        block_hash=epoch_state.block_hash,
        small_molecule_target=config.small_molecule_target,
        nanobody_target=config.nanobody_target,
        include_reaction=config.random_valid_reaction,
    )
    bt.logging.info(f"epoch: challenge={challenge}")

    # 2) Reaction id
    if config.random_valid_reaction:
        rxn_id = parse_allowed_reaction(challenge.get("allowed_reaction"))
    else:
        rxn_id = parse_allowed_reaction(
            config.allowed_reactions[0] if config.allowed_reactions else None
        )
    if rxn_id is None and molecule_track is not None:
        bt.logging.warning("epoch: no valid reaction id — molecule track disabled this epoch")

    # 3) Reset tracks
    if molecule_track is not None and rxn_id is not None:
        molecule_track.reset_for_epoch(rxn_id)
    if nanobody_track is not None:
        nanobody_track.reset_for_epoch()

    submitted_at_least_once = False
    fast_batch_size = max(10, config.batch_size // 10)  # ~10x smaller than refine batches

    # ------------------------------------------------------------------ #
    # Phase 1 — fast initial submit (must run quickly to win block_submitted)
    # ------------------------------------------------------------------ #
    async def fast_search_mol():
        if molecule_track is None or rxn_id is None:
            return
        cand = await asyncio.to_thread(molecule_track.search_one_batch, fast_batch_size)
        molecule_track.update_best(cand)

    async def fast_search_nb():
        if nanobody_track is None:
            return
        cand = await asyncio.to_thread(nanobody_track.search_one_batch, fast_batch_size)
        nanobody_track.update_best(cand)

    try:
        await asyncio.gather(fast_search_mol(), fast_search_nb())
    except Exception as e:
        bt.logging.error(f"epoch: fast search failed: {e}")
        bt.logging.debug(traceback.format_exc())

    if await _try_submit(state, molecule_track, nanobody_track):
        submitted_at_least_once = True
        bt.logging.info("epoch: phase-1 fast submission landed")

    # ------------------------------------------------------------------ #
    # Phase 2 — refine
    # ------------------------------------------------------------------ #
    for batch_idx in range(config.max_batches_per_epoch):
        current = await subtensor.get_current_block()
        if current >= epoch_state.next_boundary - 3:
            break

        async def refine_mol():
            if molecule_track is None or rxn_id is None:
                return
            cand = await asyncio.to_thread(molecule_track.search_one_batch)
            molecule_track.update_best(cand)

        async def refine_nb():
            if nanobody_track is None:
                return
            cand = await asyncio.to_thread(nanobody_track.search_one_batch)
            nanobody_track.update_best(cand)

        try:
            await asyncio.gather(refine_mol(), refine_nb())
        except Exception as e:
            bt.logging.error(f"epoch: refine batch {batch_idx} failed: {e}")
            bt.logging.debug(traceback.format_exc())

        if await _try_submit(state, molecule_track, nanobody_track):
            submitted_at_least_once = True
            bt.logging.info(f"epoch: refine submission {batch_idx} landed")

        await asyncio.sleep(0.5)

    # ------------------------------------------------------------------ #
    # Phase 3 — fallback
    # ------------------------------------------------------------------ #
    if not submitted_at_least_once:
        bt.logging.warning("epoch: no successful submission yet — attempting fallback")
        if molecule_track is not None and molecule_track.best is None:
            molecule_track.best = molecule_track.fallback_candidate()
            if molecule_track.best:
                bt.logging.info(f"epoch: molecule fallback {molecule_track.best.name}")
        if nanobody_track is not None and nanobody_track.best is None:
            nanobody_track.best = nanobody_track.fallback_candidate()
            if nanobody_track.best:
                bt.logging.info(f"epoch: nanobody fallback len={len(nanobody_track.best.sequence)}")

        # Try one more submission — rate limiter may still block, that's life
        if await _try_submit(state, molecule_track, nanobody_track):
            submitted_at_least_once = True
            bt.logging.info("epoch: fallback submission landed")

    if not submitted_at_least_once:
        bt.logging.error("epoch: ended without any successful submission — earning 0 this epoch")
    else:
        mol_cnt = molecule_track.candidates_scored if molecule_track else 0
        nb_cnt = nanobody_track.candidates_scored if nanobody_track else 0
        bt.logging.info(
            f"epoch: complete. molecule_scored={mol_cnt} nanobody_scored={nb_cnt}"
        )


# --------------------------------------------------------------------------- #
# Connection-safe main loop
# --------------------------------------------------------------------------- #

async def _open_subtensor(network: str):
    """Open a fresh async_subtensor connection. Caller owns close()."""
    st = bt.async_subtensor(network=network)
    await st.initialize()
    return st


async def run_miner(config) -> None:
    wallet, miner_uid, epoch_length = await setup_bittensor(config)

    molecule_track = None if config.disable_molecule_track else MoleculeTrack(config)
    nanobody_track = None if config.disable_nanobody_track else NanobodyTrack(config)
    if molecule_track is None and nanobody_track is None:
        bt.logging.error("Both tracks disabled — nothing to mine. Exiting.")
        return

    bdt = _get_quicknet_timelock()()
    github_path = build_github_path()

    bt.logging.info("entering main mining loop")
    last_epoch_block = -1
    consecutive_errors = 0

    subtensor = await _open_subtensor(config.network)
    try:
        while True:
            try:
                state = {
                    "config": config,
                    "wallet": wallet,
                    "subtensor": subtensor,
                    "miner_uid": miner_uid,
                    "epoch_length": epoch_length,
                    "bdt": bdt,
                    "github_path": github_path,
                    "rate_limiter": getattr(run_miner, "_rl", None) or SubmissionRateLimiter(config.no_submission_blocks),
                }
                run_miner._rl = state["rate_limiter"]

                epoch_state = await get_epoch_state(subtensor, epoch_length)
                if epoch_state.last_boundary != last_epoch_block:
                    bt.logging.info(
                        f"epoch boundary: last={epoch_state.last_boundary} "
                        f"next={epoch_state.next_boundary} current={epoch_state.current_block}"
                    )
                    last_epoch_block = epoch_state.last_boundary
                    await run_epoch(state, molecule_track, nanobody_track, epoch_state)
                else:
                    if epoch_state.blocks_remaining > 5:
                        await asyncio.sleep(12)
                    else:
                        await asyncio.sleep(2)
                consecutive_errors = 0
            except KeyboardInterrupt:
                bt.logging.success("interrupt — exiting miner")
                break
            except asyncio.CancelledError:
                bt.logging.warning("subtensor connection cancelled, reconnecting")
                try:
                    await subtensor.close()
                except Exception:
                    pass
                subtensor = await _open_subtensor(config.network)
                await asyncio.sleep(1)
            except Exception as e:
                consecutive_errors += 1
                wait_s = min(60, 2 ** consecutive_errors)
                bt.logging.error(f"main loop error #{consecutive_errors}: {e}; sleeping {wait_s}s")
                bt.logging.debug(traceback.format_exc())
                if consecutive_errors >= 5:
                    # Try a clean reconnect
                    try:
                        await subtensor.close()
                    except Exception:
                        pass
                    subtensor = await _open_subtensor(config.network)
                await asyncio.sleep(wait_s)
    finally:
        try:
            await subtensor.close()
        except Exception:
            pass


async def main() -> None:
    load_dotenv()
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    asyncio.run(main())
