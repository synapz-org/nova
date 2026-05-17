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
from elite_miner.molecule.surrogate import SurrogateScorer
from elite_miner.molecule.diversity import DiversityTracker
from elite_miner.nanobody.surrogate import NanobodySurrogateScorer
from elite_miner.nanobody.diversity import NanobodyDiversityTracker
from elite_miner.protein.features import target_features as protein_target_features
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
    # challenge.py has a relative `from .reactions import get_total_reactions` inside
    # the include_reaction branch. To make that resolve, we ALSO load utils.reactions
    # as a sibling and register both under a synthetic `_em_utils` parent package.
    parent_name = "_em_utils"
    if parent_name not in _sys.modules:
        import types
        parent = types.ModuleType(parent_name)
        parent.__path__ = [os.path.join(BASE_DIR, "utils")]
        _sys.modules[parent_name] = parent

    if f"{parent_name}.reactions" not in _sys.modules:
        _load_submodule(f"{parent_name}.reactions", os.path.join(BASE_DIR, "utils", "reactions.py"))

    _challenge = _load_submodule(f"{parent_name}.challenge", os.path.join(BASE_DIR, "utils", "challenge.py"))
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
        # SubstrateInterface needs a full ws URL, not the named network shorthand.
        # Map common names; otherwise assume caller already passed a ws:// URL.
        _network_urls = {
            "finney": "wss://entrypoint-finney.opentensor.ai:443",
            "test": "wss://test.finney.opentensor.ai:443",
            "local": "ws://127.0.0.1:9944",
            "archive": "wss://archive.chain.opentensor.ai:443",
        }
        substrate_url = _network_urls.get(config.network, config.network)
        node = SubstrateInterface(url=substrate_url)
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

        # Surrogate is the pre-ranker when a model directory is configured + a model
        # is present + holdout spearman exceeds the threshold. Falls back to proxy
        # transparently otherwise. The boundary between proxy and surrogate is here.
        self.surrogate: Optional[SurrogateScorer] = None
        if getattr(config, "surrogate_model_dir", None):
            try:
                tgt_feats = protein_target_features(self.targets[0])
            except Exception as e:
                bt.logging.warning(f"molecule: protein features unavailable for surrogate: {e}")
                tgt_feats = None
            self.surrogate = SurrogateScorer(
                model_dir=config.surrogate_model_dir,
                target_features=tgt_feats,
                min_spearman=getattr(config, "surrogate_min_spearman", 0.0),
            )
            if self.surrogate.is_ready:
                rho_str = (
                    f"{self.surrogate.metrics.spearman_rho:.3f}"
                    if self.surrogate.metrics else "n/a"
                )
                bt.logging.info(f"molecule: surrogate ready (holdout_spearman={rho_str})")
            else:
                bt.logging.warning(
                    f"molecule: surrogate not ready ({self.surrogate.fallback_reason}); using proxy"
                )

        # In inference mode we score against the *primary* target (first one).
        if self.use_inference:
            self.scorer = Boltz2Scorer(target=self.targets[0])
        else:
            self.scorer = MolProxyScorer()
        self.searcher: Optional[CombinatorialSearcher] = None
        self.best: Optional[ScoredMolecule] = None
        self.candidates_scored = 0
        self.proxy = MolProxyScorer()  # always available
        # Diversity guard: detect mode collapse across recent submissions
        self.diversity = DiversityTracker(
            window_size=getattr(config, "diversity_window", 20),
            similarity_threshold=getattr(config, "diversity_threshold", 0.85),
        )

    def _pre_rank_scorer(self):
        """The scorer used to triage large candidate pools before real Boltz2.
        Prefers surrogate when available AND no mode collapse detected.
        Falls back to proxy on either condition."""
        if self.surrogate is None or not self.surrogate.is_ready:
            return self.proxy
        if self.diversity.is_collapsed():
            bt.logging.warning(
                f"molecule: diversity collapse (median sim={self.diversity.median_similarity():.3f}) "
                f"— falling back to proxy this batch"
            )
            return self.proxy
        return self.surrogate

    def record_submission(self, smiles: str) -> None:
        """Hook to record our submitted SMILES for diversity tracking."""
        self.diversity.record(smiles)

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
        """Search → filter → uniqueness → (surrogate pre-rank) → score → rank.
        Returns best of this batch.

        When surrogate is ready, we can sample MUCH bigger batches (10x-100x normal
        batch_size) because surrogate inference is millisecond-fast vs Boltz2's seconds.
        """
        if self.searcher is None:
            return None
        size = batch_size if batch_size is not None else self.config.batch_size
        # If surrogate is live, expand the search budget — that's the whole point
        if self.surrogate is not None and self.surrogate.is_ready and batch_size is None:
            size = max(size, getattr(self.config, "surrogate_batch_size", size * 10))

        raw = self.searcher.generate_batch(size)
        valid = self.vfilter.filter_batch(raw)
        if not self.skip_unique and valid:
            valid = [(n, s) for n, s in valid if self._safe_is_unique_all_targets(s)]
        if not valid:
            return None

        pre_ranker = self._pre_rank_scorer()
        pre_scored = pre_ranker.score_batch(valid)

        if self.use_inference:
            # Take top-K by pre-ranker, then run real Boltz2 to refine
            topk = getattr(self.config, "surrogate_topk", 5)
            top = mol_rank(pre_scored)[:topk]
            candidates = [(c.name, c.smiles) for c in top]
            scored = self.scorer.score_batch(candidates, self.subnet_cfg)
        else:
            scored = pre_scored

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
        # PSSM-guided generator if PSSM file is available; otherwise fall back
        # to CDR-mutator. PSSM produces sequences in the archive-winner neighborhood.
        pssm_path = getattr(config, "nb_pssm_path", "cache/q9nzq7_pssm.json")
        if pssm_path and os.path.exists(pssm_path):
            from elite_miner.nanobody.pssm_generator import PSSMGenerator, PSSMGenerationConfig
            self.generator = PSSMGenerator(
                self.validity_cfg,
                gen_cfg=PSSMGenerationConfig(
                    min_mutations=getattr(config, "nb_min_mutations", 3),
                    max_mutations=getattr(config, "nb_max_mutations", 10),
                    pssm_path=pssm_path,
                ),
            )
            bt.logging.info(f"nanobody: using PSSM-guided generator from {pssm_path}")
        else:
            self.generator = NanobodyGenerator(self.validity_cfg)
            bt.logging.info("nanobody: using CDR-mutator generator (no PSSM found)")
        self.skip_unique = config.no_uniqueness_check
        self.use_inference = config.use_inference

        # Phase 3 surrogate: when configured, replaces proxy as pre-ranker
        self.surrogate: Optional[NanobodySurrogateScorer] = None
        if getattr(config, "nb_surrogate_model_dir", None):
            try:
                tgt_feats = protein_target_features(self.targets[0])
            except Exception as e:
                bt.logging.warning(f"nanobody: protein features unavailable for surrogate: {e}")
                tgt_feats = None
            self.surrogate = NanobodySurrogateScorer(
                model_dir=config.nb_surrogate_model_dir,
                target_features=tgt_feats,
                min_spearman=getattr(config, "nb_surrogate_min_spearman", 0.0),
            )
            if self.surrogate.is_ready:
                rho = self.surrogate.metrics.spearman_rho if self.surrogate.metrics else None
                bt.logging.info(
                    f"nanobody: surrogate ready (holdout_spearman={rho})"
                )
            else:
                bt.logging.warning(
                    f"nanobody: surrogate not ready ({self.surrogate.fallback_reason}); using proxy"
                )

        if self.use_inference:
            self.scorer = BoltzGenScorer(target=self.targets[0])
        else:
            self.scorer = ProxyNanobodyScorer()
        self.proxy = ProxyNanobodyScorer()  # always available
        self.best: Optional[ScoredNanobody] = None
        self.candidates_scored = 0
        # Mode-collapse guard: pairwise sequence identity over recent submissions
        self.diversity = NanobodyDiversityTracker(
            window_size=getattr(config, "nb_diversity_window", 10),
            identity_threshold=getattr(config, "nb_diversity_threshold", 0.95),
        )

    def _pre_rank_scorer(self):
        if self.surrogate is None or not self.surrogate.is_ready:
            return self.proxy
        if self.diversity.is_collapsed():
            bt.logging.warning(
                f"nanobody: diversity collapse (median identity={self.diversity.median_identity():.3f}) "
                f"— falling back to proxy this batch"
            )
            return self.proxy
        return self.surrogate

    def record_submission(self, sequence: str) -> None:
        """Hook to record submitted sequence for diversity tracking."""
        self.diversity.record(sequence)

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
        # If surrogate is live, expand the search budget (surrogate is cheap)
        if self.surrogate is not None and self.surrogate.is_ready and batch_size is None:
            size = max(size, getattr(self.config, "nb_surrogate_batch_size", size * 5))

        seqs = self.generator.generate_batch(size)
        if not self.skip_unique and seqs:
            seqs = [s for s in seqs if self._safe_is_unique_all_targets(s)]
        if not seqs:
            return None

        pre_ranker = self._pre_rank_scorer()
        pre_scored = pre_ranker.score_batch(seqs)

        if self.use_inference:
            topk = getattr(self.config, "nb_surrogate_topk", 5)
            top = nb_rank(pre_scored)[:topk]
            top_seqs = [c.sequence for c in top]
            scored = self.scorer.score_batch(top_seqs, self.subnet_cfg)
        else:
            scored = pre_scored
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
    # Track submitted candidates for diversity guards
    if mt is not None and mt.best is not None:
        mt.record_submission(mt.best.smiles)
    if nt is not None and nt.best is not None:
        nt.record_submission(nt.best.sequence)
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
    # Rate limiter persists across reconnects (block history is global to subnet)
    rate_limiter = SubmissionRateLimiter(config.no_submission_blocks)

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
                    "rate_limiter": rate_limiter,
                }

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
