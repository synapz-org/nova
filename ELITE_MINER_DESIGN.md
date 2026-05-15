# NOVA SN68 Elite Miner — Architecture v2 (May 2026)

> Supersedes the December 2025 design. The competition changed materially:
> PSICHIC removed, nanobody track added, scoring became winner-take-all by rank,
> and payouts moved off-chain. This document is rewritten against upstream
> commit `3a62e98` (May 2026 sync).

## TL;DR

- **Two tracks, one miner.** SN68 now scores a `small_molecule` track and a `nanobody` track independently. Nanobodies are 60% of non-burn incentive, molecules 40%.
- **Boltz2 is the molecule scoring function. PSICHIC is gone.** Optimize the validator's metric directly.
- **BoltzGen scores nanobodies.** Tiebreak is rank-sum across confidence / physical_interaction / developability sub-metrics, then submission block.
- **Winner-take-all per track per epoch.** Rank 1 wins the whole pool. There is no score-weighted distribution and no entropy bonus.
- **Submission timing is a real lever.** Ties are broken by `block_submitted` → `push_time` → `uid`. Submitting earlier than equally-scored competitors wins.
- **Hard uniqueness constraint.** InChIKey (molecules) and sequence hash (nanobodies) must not appear in `Metanova/Submission-Archive` for the current target. You cannot resubmit something already on the leaderboard.
- **Effective payout ≈ 11% (molecule) and ≈ 17% (nanobody) of subnet emissions per epoch**, distributed via the off-chain compound-payout API, not on-chain weights.

---

## 1. What the validator actually scores

### 1.1 Molecule track

Pipeline, in order, from `neurons/validator/molecule_validity.py` and `utils/inference.py`:

1. **Reaction allowed this epoch.** If `random_valid_reaction: true` (current config), one reaction id is selected from the epoch's block hash. Submissions using other reactions are rejected.
2. **Format.** Submission must be `rxn:rxn_id:m1:m2[:m3]` referencing the SAVI combinatorial DB.
3. **No duplicates within a submission** (config: `num_molecules`, currently 1).
4. **Heavy atom count ≥ `min_heavy_atoms`** (10).
5. **`min_rotatable_bonds ≤ NumRotatableBonds ≤ max_rotatable_bonds`** (1..10).
6. **No banned atoms** (`Se, Na, Fe, Zn`).
7. **RDKit-parseable.**
8. **Boltz-safe atom naming** — atom name `symbol + canonical_rank_index` must be ≤ 4 chars. Filters out molecules with very high atom counts and unusual symbols.
9. **InChIKey unique vs `Metanova/Submission-Archive/{target}_molecules.csv`** on HuggingFace. **This is the killer constraint.** Anything previously submitted by anyone is rejected.
10. **Boltz2 inference** (`external_tools/boltz/boltz_wrapper.py`). Configured metric: `["affinity_probability_binary", "affinity_pred_value"]`. Combination strategy: `heavy_atom_normalization`. Mode: `max`.
11. **Rank UIDs by `final_molecule_score`** (sum of combined item scores). Tiebreak: `block_submitted` → `push_time` → `uid`.

### 1.2 Nanobody track

From `neurons/validator/nanobody_validity.py` (path inferred from new `utils/nanobodies.py`) and `utils/inference.py`:

1. **Sequence length** in `[90, 150]`.
2. **Allowed AAs only** (no X / ambiguous).
3. **≥ 1 Cysteine** (currently). If multiple required, plausible cys-pair separation `[35, 80]`.
4. **No homopolymer run > 6** (e.g. `AAAAAAA` rejected).
5. **No di-repeat run > 4 pairs** (e.g. `GSGSGSGSGS` rejected).
6. **No signal-peptide-like N-terminus** — 8+ hydrophobic AAs in any 12-aa window in the first 30 positions.
7. **Nativeness score ≥ 0.45** (via `NOVA-nanobody-filter` submodule, IgBLAST-derived).
8. **Human framework score ≥ 0.75**.
9. **Similarity to top-50 sequences ≤ 0.95** (k-mer/alignment against existing leaderboard top).
10. **VHH FR2 hallmarks enforced** — `E` or `Q` at position 49, `R` at position 50. Failures get neutral placeholder per `3a62e98`.
11. **Sequence-hash unique vs `Metanova/Submission-Archive/{target}_nanobodies.csv`.**
12. **BoltzGen inference.** Reports per-(seq, target) components: `confidence_rank_sum`, `physical_interaction_rank_sum`, `developability_rank_sum`. Final score = sum of combined item scores per UID; mode `min` (lower is better via `rank_sum` aggregation, configured by `boltzgen_rank_by`).
13. **Rank UIDs** by final score, with tiebreak chain `confidence` → `physical_interaction` → `developability` → `block_submitted` → `push_time` → `uid`.

### 1.3 Incentive math

From `neurons/validator/weights.py`:

```
burn_rate = 0.722
nanobody_weight = 0.60      # config.competition.nanobody_weight

molecule_pool = (1 - 0.722) * (1 - 0.60) = 0.278 * 0.40 ≈ 0.1112 (11.1%)
nanobody_pool = (1 - 0.722) * 0.60       = 0.278 * 0.60 ≈ 0.1668 (16.7%)
```

When `payout.enabled: true` (current default, `override_uid: 61`), all non-burn weight is routed on-chain to UID 61, and the actual ranked winners are paid off-chain to their **coldkey** via `https://emission-transfer-api.metanova-labs.ai/payouts/compound-epoch-reward`. **Implication: registration of a hotkey is needed to be ranked, but emission flows to coldkey via API, not to your hotkey stake.**

---

## 2. Why the old design is obsolete

Old design assumption → current reality:

| Old assumption | Reality |
|---|---|
| 50% PSICHIC + 50% Boltz2 | 100% Boltz2 for molecules; PSICHIC removed entirely |
| Score-weighted reward | Winner-take-all per track |
| Entropy maximization for late-epoch diversity | No entropy weight in scoring; uniqueness is binary (pass/fail) |
| Antitarget selectivity ratio | Single target per track; no antitargets in current config |
| Train surrogate on PSICHIC | Surrogate target shifted to Boltz2; much more expensive labels |
| Pareto multi-objective | Single scalar per track; tiebreak is deterministic |
| Single track | **Nanobody track unaddressed — it's 60% of the pool** |

---

## 3. Strategy

### 3.1 First principles

The competition is now: produce **one** combinatorial-DB molecule with the highest Boltz2 score (subject to validity + uniqueness), and **one** novel nanobody sequence with the lowest BoltzGen rank-sum (subject to validity + uniqueness), and commit both before competitors do.

There are three time scales to optimize:

1. **Pre-epoch (offline / continuous):** Build the largest pool of pre-scored candidates you can.
2. **Intra-epoch (after block-hash reveals the allowed reaction):** Filter the pool by reaction, top-rank, submit.
3. **Submission timing:** Win ties by being earlier.

### 3.2 Molecule strategy

**Search space.** SAVI combinatorial DB has ~1.7B products across all reactions. Per-epoch only one reaction is allowed; the per-reaction product count is on the order of 10⁸–10⁹.

**Approach: surrogate-screened combinatorial enumeration.**

1. **Build per-reaction candidate pools offline.** For each reaction id, enumerate building-block combinations, apply cheap validity filters (HA, RB, Boltz-safe, banned atoms) in batch. Cache the survivors. Disk-cheap, lets us skip enumeration cost at epoch boundary.
2. **Train a Boltz2 surrogate.** Sample N candidates per reaction (e.g. N=10k–50k spread across reactions), run real Boltz2, fit an ECFP4/MACCS+target-features regressor (LightGBM or small MLP). Surrogate target = `combined_score` (heavy-atom-normalized Boltz2 output). Retrain when the target rotates (current target rotates weekly).
3. **At epoch start:**
   a. Read block hash → get `allowed_reaction`.
   b. Load that reaction's pre-filtered pool.
   c. Score top K (e.g. 100k–1M) candidates with surrogate.
   d. Filter by **InChIKey ∉ Submission-Archive** (the gating uniqueness check — query HF once, cache the InChIKey set per target).
   e. Run real Boltz2 on top ~50–500 surrogate hits.
   f. Submit best.
4. **Continuously refine.** Every epoch produces new (mol, real Boltz2 score) pairs for the current target — fold those into the surrogate's training set immediately.

**Why this beats the random sampler:**
- The reference miner picks random reactions and random building blocks. Median Boltz2 affinity of random products is well below the leaderboard winner. With a half-decent surrogate (Spearman ρ > 0.5), enriching top-1000 surrogate hits over random sampling gives ≥10× hit rate on real high-scorers within the same Boltz2 budget.

**Where we lose:**
- Uniqueness collisions. The first miner to find each high-scoring InChIKey owns it. We need to either (a) be faster than them on the same surrogate signal, or (b) explore regions of chemical space they aren't covering. Track competitors' submissions over time to identify their coverage gaps.

### 3.3 Nanobody strategy

The nanobody track is more open (no fixed combinatorial DB — any valid VHH sequence is admissible) and is the bigger pool (60% of non-burn). Worth more attention than molecules.

**Generation approaches, in priority order:**

1. **Mutational hill-climbing from a strong template.** Start with a known anti-target VHH (literature, PDB) framework-matched against the current target. Mutate CDR positions (especially CDR3) under the validity rules. Score with BoltzGen, accept improvements, reject.
2. **CDR grafting.** Combine framework regions from a humanized VHH (high humanness, high nativeness) with CDR3 from a target-specific binder. Pass through the VHH hallmark check at FR2 pos 49/50.
3. **AbLang / IgLM language-model generation** with rejection sampling against the validity filters before BoltzGen scoring.

**Hard constraints that drive design:**
- VHH hallmarks at FR2 pos 49 (E/Q) and pos 50 (R). Enforced strictly upstream of BoltzGen.
- Sequence-hash uniqueness vs the archive. Any literal duplicate of a previously-submitted sequence is rejected — but a single AA change makes it unique.
- BoltzGen scores by `rank_sum`, so improvement is **rank-relative across the validator's pool**, not absolute. Submitting a sequence that improves on confidence by a sliver isn't enough; we need it to dominate or come close to dominating across confidence/physical/developability simultaneously.

**Compute cost.** BoltzGen is ~5–15s per (sequence, target) on an A100. Per-epoch budget for a 360-block epoch (~72 min at 12s blocks) running on one A100 is ~300–800 BoltzGen evaluations. Plan for ~100 candidates surviving validity filters per epoch.

### 3.4 Submission timing

The first commitment of an epoch's winning candidate wins ties. Two implications:

- **Don't wait for the "optimal" score.** Once you have a candidate that beats the current leaderboard's known top, submit it. You can resubmit later (subject to `no_submission_blocks: 10` rate limit) — but the first commit is what wins ties.
- **The 10-block rate-limit is per UID.** Means we can update at most every ~2 minutes. So submit early with the best-so-far, then update if surrogate finds something better, until the epoch closes.

---

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            NOVA SN68 Elite Miner                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   ┌──────────────────────────┐         ┌──────────────────────────┐          │
│   │   Offline (continuous)   │         │   Online (per-epoch)     │          │
│   ├──────────────────────────┤         ├──────────────────────────┤          │
│   │ • Combinatorial enum     │         │ • Read block hash        │          │
│   │ • Validity prefilter     │ ──────▶ │ • Pool lookup (reaction) │          │
│   │ • Boltz2 surrogate train │         │ • Uniqueness filter (HF) │          │
│   │ • Nanobody template lib  │         │ • Surrogate top-K        │          │
│   │ • Competitor monitor     │         │ • Real Boltz2 top-N      │          │
│   └──────────────────────────┘         │ • Nanobody mutate+score  │          │
│                                         │ • Submit + repeat        │          │
│                                         └──────────────────────────┘          │
│                                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Module layout

Keep most of the existing `elite_miner/` skeleton; add nanobody track and surrogate.

```
elite_miner/
├── __init__.py
├── run.py                    # Main loop, epoch handling, submission
│
├── molecule/
│   ├── searcher.py           # (exists) reaction enumeration + validity prefilter
│   ├── filters.py            # (exists) HA, RB, banned, Boltz-safe
│   ├── scorer.py             # (exists) proxy ranking; real Boltz2 wrapper call
│   ├── surrogate.py          # NEW. LightGBM/MLP on ECFP4 → Boltz2 score
│   └── pool_cache.py         # NEW. Disk-backed per-reaction pre-filtered pool
│
├── nanobody/
│   ├── generator.py          # NEW. Template + mutate, CDR graft, optional LM
│   ├── validator.py          # NEW. Mirrors utils.nanobodies filters locally
│   ├── scorer.py             # NEW. BoltzGen wrapper call
│   └── templates.py          # NEW. Curated framework + CDR template library
│
├── shared/
│   ├── uniqueness.py         # NEW. Cached HF Submission-Archive lookup
│   ├── competitor_tracker.py # NEW. Watch on-chain commitments / GitHub uploads
│   └── timing.py             # NEW. Epoch boundary detection, submission scheduler
│
└── tests/                    # (exists) extend with nanobody + surrogate tests
```

### 4.2 Main loop sketch

```python
async def run_miner(config, wallet, subtensor):
    # Continuous: refresh competitor data, retrain surrogate if needed
    competitor_task = asyncio.create_task(competitor_tracker.run(config, subtensor))

    while True:
        # 1. Block-hash → epoch challenge
        block_hash, epoch_block = await wait_for_epoch_boundary(subtensor, config)
        challenge = get_challenge_params_from_blockhash(
            block_hash,
            small_molecule_target=config.small_molecule_target,
            nanobody_target=config.nanobody_target,
            include_reaction=config.random_valid_reaction,
        )
        allowed_rxn = challenge["allowed_reaction"]
        target_mol = challenge["small_molecule_target"]
        target_nb = challenge["nanobody_target"]

        # 2. Refresh archive uniqueness sets
        mol_inchikeys = uniqueness.fetch_archive(target_mol, "molecules")
        nb_hashes = uniqueness.fetch_archive(target_nb, "nanobodies")

        # 3. Molecule search (parallel to nanobody)
        mol_task = asyncio.create_task(
            search_molecule(allowed_rxn, target_mol, mol_inchikeys, config)
        )
        nb_task = asyncio.create_task(
            search_nanobody(target_nb, nb_hashes, config)
        )

        # 4. Initial submission as soon as either has a candidate
        best_mol, best_nb = None, None
        async for kind, candidate in iter_completed_first({mol_task, nb_task}):
            if kind == "mol": best_mol = candidate
            if kind == "nb":  best_nb = candidate
            await submit_if_allowed(best_mol, best_nb, state)

        # 5. Refinement loop until epoch ends or rate-limit prevents resubmits
        await refine_until_epoch_end(state, target_mol, target_nb, ...)
```

### 4.3 Surrogate model spec

- **Features:** ECFP4 (radius 2, 2048 bits) + MACCS (167) + target-protein embedding (ESM2-T12 last-layer mean, cached per target). Total ~2.5k features.
- **Model:** LightGBM regressor (gbdt, 1000 trees, lr 0.05, num_leaves 64). Trains in ~5 min on 50k samples on CPU.
- **Label:** real Boltz2 combined score with heavy-atom normalization as the validator computes it.
- **Validation:** held-out 10% per target. Track Spearman ρ; target ρ > 0.5 before trusting; ρ > 0.7 before pruning aggressively.
- **Retraining trigger:** new target (weekly rotation), or +5000 new labels accumulated.

### 4.4 Uniqueness cache

- Pull `Metanova/Submission-Archive/{target}_molecules.csv` and `{target}_nanobodies.csv` via the HF metadata SHA cache (already implemented in `utils.challenge.entry_unique_for_protein_hf` — reuse it).
- Refresh on a 60s TTL during active mining; faster (every commit) at epoch boundary.
- Store InChIKey set in-memory; persist last snapshot to disk so cold-start is fast.

### 4.5 Competitor tracker

The validator publishes ranked logs (`add individual validator scores logging for auditing` — recent commit). Parse logs / on-chain commitments to learn:
- Who's submitting (UIDs, rough commit rate).
- Approximate top scores (from validator log scraping if available, otherwise from Submission-Archive growth rate).
- Diversity of submitted scaffolds (we can compute Murcko scaffolds from the archive ourselves).

Use this to:
- Detect uncovered scaffold regions in the SAVI space.
- Estimate whether our surrogate's top-K already overlaps with archive entries (= competitors found them first).

---

## 5. Hardware & deployment

**Minimum competitive (one A100 80GB):**
- Run Boltz2 + BoltzGen sequentially per epoch (~300 Boltz2 calls and ~300 BoltzGen calls per 72-min epoch).
- 64GB RAM, 16 cores, 1TB NVMe for SAVI pool + ESM cache.

**Strong setup (2× A100 80GB):**
- Parallel Boltz2 (GPU 0) and BoltzGen (GPU 1) per `utils/inference.py`'s multi-GPU dispatch. This is the deployment the validator itself uses.
- Enables real-time scoring of ~500–1000 candidates per track per epoch.

**Basilica provisioning:**
```bash
basilica up \
    --gpu-count 2 \
    --gpu-type a100-80gb \
    --name nova-elite-miner \
    --memory-mb 131072 \
    --disk-gb 1024 \
    -d
```

---

## 6. Implementation phases

### Phase 1 — Repair foundation (3 days)
- [ ] Audit existing `elite_miner/{searcher,filters,scorer,run}.py` against current `utils/`, `neurons/miner/miner.py`. Patch imports for moved modules (`utils.inference`, `utils.challenge`, `utils.molecules`).
- [ ] Replace PSICHIC references with Boltz2 wrapper calls (`external_tools.boltz.boltz_wrapper.BoltzWrapper`).
- [ ] Add `uniqueness.py` wrapper around `entry_unique_for_protein_hf`.
- [ ] Reorg into `molecule/` subpackage; preserve existing tests.

### Phase 2 — Surrogate + pool cache (4 days)
- [ ] `pool_cache.py`: per-reaction prefiltered SMILES on disk (parquet, partitioned by reaction id).
- [ ] `surrogate.py`: ECFP4+MACCS+ESM2 → LightGBM. Training script + checkpoint loader.
- [ ] Seed label set (5k–10k per current target) by running real Boltz2 offline.
- [ ] Validation harness: held-out Spearman ρ vs real Boltz2.

### Phase 3 — Nanobody track (5 days)
- [ ] `templates.py`: curate ~20 humanized VHH templates with strong nativeness + framework scores.
- [ ] `validator.py`: locally enforce all `utils/nanobodies.py` rules so we don't waste BoltzGen calls.
- [ ] `generator.py`: mutate (CDR-biased) + CDR-graft. Optional: IgLM/AbLang integration as a feature flag.
- [ ] `scorer.py`: wrap `external_tools.boltzgen.boltzgen_wrapper.BoltzgenWrapper.run_nanobody_inference`.

### Phase 4 — Orchestration + timing (3 days)
- [ ] `timing.py`: precise epoch boundary detection, submission rate-limiter.
- [ ] `run.py` rewrite: async parallel molecule + nanobody, early-submit-then-refine pattern.
- [ ] `competitor_tracker.py`: archive diff watcher (HF) + on-chain commitment scraper.

### Phase 5 — Production (3 days)
- [ ] Basilica deploy scripts (2× A100).
- [ ] Health checks: GPU memory, last successful submission age, archive sync.
- [ ] Surrogate auto-retrain cron when target rotates (weekly).
- [ ] Off-chain payout receipt monitor (verify our coldkey is actually receiving from compound-payout API).

---

## 7. Success metrics

| Metric | Target | Reference |
|---|---|---|
| Weekly molecule wins | ≥ 1 (out of ~7 days × ~20 epochs/day) | Hard to predict; depends on competitor strength |
| Weekly nanobody wins | ≥ 2 | Larger search space + more competitor variance |
| Submission success rate | ≥ 99% | Reference miner is ~90%; we should be ≥ 99% |
| Epoch coverage | 100% | Submit at least one candidate per epoch |
| Surrogate ρ vs Boltz2 | ≥ 0.6 within 2 weeks of new target | Internal validation |
| Cold-start to first submit | ≤ 60s of epoch boundary | Limits competitors' early-submit tiebreak advantage |

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Off-chain payout API stops paying (override remains) | Monitor coldkey receipts daily; have a one-knob switch to disable mining if payouts dry up |
| Weekly target rotation invalidates surrogate | Pre-warm: as soon as next target is announced (or we detect rotation), retrain in parallel with current week's mining |
| Boltz2 / BoltzGen package version drift in `external_tools/` | Pin to upstream's vendored versions; re-test on every upstream merge |
| Uniqueness archive growth makes surrogate top-K mostly stale | Track archive overlap rate; switch to scaffold-novelty-biased sampling when overlap > 50% |
| Submission rate-limit (`no_submission_blocks: 10`) wastes refinement budget | Submit best-known immediately; only re-submit when expected score gain × tiebreak risk > 0 |
| BoltzGen rank-sum scoring means absolute score is unstable across epochs | Treat BoltzGen as a relative ranker; track rolling top-percentile from our own submissions and submit only when above that |
| `combinatorial_db` SQLite open-mode quirks (read-only constraint in sandbox-like env) | Open with `mode=ro&immutable=1` — already idiomatic, just need to confirm |

---

## 9. Open questions to resolve before Phase 1 ends

1. **Is the emission override actually paying out?** Cross-check via the off-chain compound-payout API logs (`emission-transfer-api.metanova-labs.ai`) and a fresh coldkey receipt. If payouts have stalled, all of the above is moot.
2. **Who is UID 61?** That's the on-chain weight recipient. If MetaNova rotates this regularly, fine; if it's frozen, the bounty payout is the only path and we should treat that as the canonical incentive.
3. **What is the current top molecule and nanobody score for the live targets?** Establishes a floor we need to clear. Pull from validator audit logs (`add individual validator scores logging for auditing`) or scrape recent epochs from the archive.
4. **Is there a `combinatorial_db` schema change in this merge?** The merge stat shows `combinatorial_db/reactions.py` was touched; verify our `searcher.py` still parses it correctly.
5. **Multi-GPU vs single-GPU economics.** Confirm Boltz2 throughput on A100 80GB to set realistic per-epoch candidate budgets.

---

## 10. References

- `neurons/miner/miner.py` — reference (random-sampling) miner, current upstream.
- `neurons/validator/ranking.py` — winner-take-all ranking logic, tiebreak chain.
- `neurons/validator/weights.py` — burn rate, override, payout dispatch.
- `neurons/validator/molecule_validity.py` — full molecule validation pipeline.
- `utils/nanobodies.py` — nanobody validation primitives (homopolymer, di-repeat, cys-pair, signal-peptide, etc).
- `utils/inference.py` — Boltz / BoltzGen multi-GPU dispatch.
- `utils/challenge.py` — block-hash → reaction selection, HF archive uniqueness check.
- `utils/molecules.py` — `is_boltz_safe_smiles`, `get_heavy_atom_count`, `compute_maccs_entropy`.
- `config/config.yaml` — live thresholds, reactions, nanobody_weight, payout config.
- `external_tools/boltz/`, `external_tools/boltzgen/` — vendored scoring models.
- HF datasets: `Metanova/Submission-Archive`, `Metanova/Proteins`, `Metanova/SAVI-2020`.
