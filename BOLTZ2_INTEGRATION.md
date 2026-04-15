# Boltz-2 Miner Integration

## Context

The NOVA subnet (SN68) incentive mechanism splits rewards as follows:

| Model | Weight | Miner implementation |
|-------|--------|----------------------|
| Boltz-2 | **100%** (`boltz_weight: 1.0`) | Implemented — see below |
| PSICHIC | 0% (weight unused) | Used as cheap pre-filter only |

The validator already runs full Boltz-2 scoring (`boltz/wrapper.py`). The stock miner was
optimising exclusively for PSICHIC, which was never rewarded. This document describes what
was implemented and what remains to do.

---

## Scoring Formula

The validator scores each miner's submission with:

```
boltz_score = (affinity_probability_binary - affinity_pred_value) / heavy_atom_count
```

Where:
- `affinity_probability_binary` ∈ [0, 1] — predicted probability that the ligand binds
- `affinity_pred_value` ∈ (−∞, 0] — predicted binding energy (kcal/mol, more negative = stronger)
- `heavy_atom_count` — number of heavy atoms in the molecule

Winner is the UID with the **maximum** `boltz_score`.

Key insight: because `affinity_pred_value` is negative, subtracting it is additive.
A molecule with `affinity_probability_binary=0.9` and `affinity_pred_value=-9.0` over 25
heavy atoms gives `(0.9 − (−9.0)) / 25 = 0.396`. This is a **ligand-efficiency metric**:
smaller, more potent binders score higher per atom.

**Optimal molecule properties:**
- High binary binding probability (≥ 0.7)
- Strong predicted binding energy (≤ −7 kcal/mol)
- Moderate heavy atom count (15–30 atoms) — large molecules are penalised by the denominator
- Drug-like: MW 200–450 Da, logP 1–4, ≤ 5 H-bond donors, ≤ 10 H-bond acceptors

The validator picks the **first** molecule in a miner's submission
(`sample_selection: "first"`, `num_molecules_boltz: 1`), so molecule ordering matters.

---

## Implemented

### 1. Ligand-efficiency pre-filter (`neurons/miner.py`, `config/config.yaml`)

`max_heavy_atoms: 35` in `config.yaml` drops molecules larger than ~500 Da before PSICHIC
scoring. The Boltz-2 scoring formula divides by `heavy_atom_count`, so a 50-HA molecule
with the same binding as a 25-HA molecule scores half as well. 35 HA is a conservative
cutoff that eliminates clear non-starters while keeping typical drug-like candidates.

```python
max_ha = getattr(state['config'], 'max_heavy_atoms', None)
if max_ha:
    df = df[df['heavy_atoms'] <= max_ha]
```

### 2. Boltz-safe filter in the PSICHIC loop (`neurons/miner.py`)

```python
# Filter for Boltz-2 compatibility (validator scores with Boltz-2)
df = df[df['product_smiles'].apply(lambda x: is_boltz_safe_smiles(x)[0])]
```

Molecules that fail `is_boltz_safe_smiles` (e.g., atom names > 4 characters) would receive
`-inf` from the validator. Filtering them out early ensures we never waste a submission on
an unscorable molecule.

### 3. `run_boltz_prescoring` function (`neurons/miner.py`)

When the miner is ≤ 100 blocks from epoch end and has found a candidate, it:

> **Why 100 blocks?** The original 50-block window (~10 min) is tight for slow hardware.
> 5 candidates × 150 s/mol (RTX 3090) = 12.5 min > 10 min available.
> Triggering at 100 blocks (~20 min) ensures Boltz completes before the submission
> deadline on all hardware. The `boltz_prescored` flag resets automatically when a
> new best candidate is found, so a second pass still fires for any new molecules.

1. Takes the top-5 PSICHIC-ranked molecules
2. Checks the in-memory Boltz score cache for each (see §3)
3. Runs `BoltzWrapper.score_molecules_target()` only on uncached molecules
4. Stores new scores in the cache
5. Sorts by `boltz_score` descending
6. Puts the highest-scoring Boltz-2 molecule **first** in `candidate_product`

The blocking Boltz inference runs via `asyncio.to_thread()` so the event loop stays live.

### 4. Boltz score cache (`neurons/miner.py`)

```python
state['boltz_score_cache'] = {}  # {(canonical_smiles, protein_code): float}
```

Key properties:
- Keyed by `(get_canonical_smiles(smiles), protein_code)` — canonical SMILES ensures
  equivalent molecules match regardless of input representation.
- Persists across epoch boundaries within a single miner session. If the weekly target
  changes, the new protein code makes old entries inert (different key → cache miss).
- Molecules already scored in a previous Boltz call are returned instantly without any
  GPU work, saving 45–150 s per cache hit.
- Reset on process restart (purely in-memory). For persistent caching see §Future Work.

**Benefit:** When a new PSICHIC best is found mid-epoch and `boltz_prescored` resets,
the second Boltz call only re-runs on genuinely new molecules. The previously top-ranked
molecules are served from cache in microseconds.

### 5. `candidate_molecules` state tracking

The full PSICHIC top-10 DataFrame is kept in `state['candidate_molecules']` so
`run_boltz_prescoring` can access SMILES without re-querying the dataset.

### 6. `boltz_prescored` flag

Set to `True` once Boltz pre-scoring completes for a given candidate. Reset to `False`
whenever a new best PSICHIC candidate is found or the epoch rolls over. Combined with the
cache, this prevents redundant GPU calls without losing the ability to re-evaluate new
candidates.

### 7. `get_canonical_smiles` utility (`utils/molecules.py`)

```python
def get_canonical_smiles(smiles: str) -> str:
    """RDKit canonical form for consistent cache keying. Falls back to input on failure."""
```

---

## Architecture Diagram

```
Epoch start
    │
    ▼
stream SAVI-2020 chunks (128 mols each)
    │
    ├─ filter: min_heavy_atoms ≥ 10
    ├─ filter: max_heavy_atoms ≤ 35   ← new: drops ligand-efficiency non-starters
    ├─ filter: is_boltz_safe_smiles()
    │
    ▼
PSICHIC scoring (target - antitarget_weight × antitarget)
    │
    ▼
top-10 molecules → candidate_molecules (stored in state)
    │
    ▼  (when blocks_until_epoch ≤ 100)  ← was 50; 20-min window fits slow hardware
Boltz-2 pre-scoring on top-5 candidates
    │  ├─ cache hit  → score returned instantly
    │  └─ cache miss → asyncio.to_thread(BoltzWrapper) → score stored in cache
    ▼
reorder: best Boltz mol → position 0 in submission
    │
    ▼  (when blocks_until_epoch ≤ 20)
encrypt + submit to chain / GitHub
    │
    ▼
Validator receives submission
    ▼
Boltz-2 scores candidate_product[0]  (sample_selection="first")
    ▼
Winner = max boltz_score across all miners
```

---

## MSA File Requirement

`BoltzWrapper.create_yaml_content()` references a pre-computed MSA file:

```
boltz/msa_files/{weekly_target}.a3m
```

Current files available:
- `boltz/msa_files/P31645.a3m`
- `boltz/msa_files/P31652.a3m`

**Action required when the weekly target rotates:** generate and commit the new `.a3m` file
before the epoch begins. Without it, Boltz-2 will run without evolutionary context and
predictions will be weaker.

To generate an MSA for a new target (UniProt ID `P12345`):

```bash
# Using ColabFold's mmseqs2 API (fast, no local DB needed)
python -c "
from colabfold.batch import get_msa_and_templates
get_msa_and_templates('P12345_sequence_here', 'boltz/msa_files/', use_env=True)
"
# Rename output to P12345.a3m
```

Alternatively use the `colabfold_search` CLI or any standard MSA pipeline
(HHblits / Jackhmmer against UniRef90 + BFD).

---

## Performance Considerations

With the default `boltz_config.yaml` settings (`sampling_steps: 100`,
`diffusion_samples: 1`, `sampling_steps_affinity: 100`,
`diffusion_samples_affinity: 3`), one Boltz-2 prediction takes approximately:

| Hardware | Time per molecule |
|----------|-------------------|
| A100 80 GB | ~45 s |
| RTX 4090 | ~90 s |
| RTX 3090 | ~150 s |

Scoring 5 candidates ≈ 4–12 minutes on typical mining hardware. Epochs are ~72 minutes
(360 blocks × 12 s), so this fits comfortably within the 50-block window (~10 minutes).

**Tuning options** (edit `boltz/boltz_config.yaml`):

```yaml
# Faster but lower-quality — suitable for early-epoch filtering
sampling_steps: 50             # default: 100
diffusion_samples_affinity: 1  # default: 3
```

Reduce `max_candidates` in `run_boltz_prescoring(state, max_candidates=3)` if GPU memory
is tight.

---

## Implemented Optimisations (follow-on)

### F. Pharmacophore pre-filter (`neurons/miner.py`)

Lipinski-inspired drug-likeness check applied **before** PSICHIC scoring, eliminating
molecules that cannot plausibly bind (extreme logP, no H-bond capacity):

```python
def _pharma_ok(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return (
        1 <= Descriptors.NumHDonors(mol) <= 3
        and 2 <= Descriptors.NumHAcceptors(mol) <= 7
        and 0.0 <= Descriptors.MolLogP(mol) <= 4.5
    )
```

Typical reduction: ~25–35% of SAVI-2020 batch molecules, saving ~50–100 ms per 128-molecule
chunk before PSICHIC inference. RDKit descriptors run in microseconds per molecule.

### G. Adaptive Boltz timing (`boltz/wrapper.py` + `neurons/miner.py`)

`BoltzWrapper` now records `last_inference_duration` after each successful run. After the
first Boltz call, the trigger threshold is updated:

```
trigger_blocks = int(time_per_mol × max_candidates / 12) + 20   (minimum: 30)
```

Effect on different hardware:

| Hardware | Time/mol | 5 candidates | Old trigger | New trigger |
|----------|----------|--------------|-------------|-------------|
| A100 80 GB | ~45 s | ~4 min | 100 blocks (20 min) | ~39 blocks (~8 min) |
| RTX 4090 | ~90 s | ~8 min | 100 blocks (20 min) | ~58 blocks (~12 min) |
| RTX 3090 | ~150 s | ~13 min | 100 blocks (20 min) | ~83 blocks (~17 min) |

On A100, this gives **12 extra minutes of PSICHIC streaming** (to find a better seed)
before Boltz kicks in. On RTX 3090 it barely changes (hardware is the bottleneck anyway).
The first epoch always uses the conservative 100-block default.

### H. Anytime incremental Boltz scoring (`neurons/miner.py`)

**Problem with old one-shot batch:** `run_boltz_prescoring` previously gathered all N
uncached candidates into a single batch, ran one `BoltzWrapper.score_molecules_target()`
call, and only reordered the submission **after** all N molecules were scored. If the epoch
ended mid-run (e.g., 3 of 5 molecules complete), the submission still reflected the raw
PSICHIC ranking for position 0.

**New anytime approach:** candidates are now scored **one molecule at a time** in descending
PSICHIC-score order. After each molecule is scored (whether from cache or GPU inference),
`state['candidate_product']` is reordered immediately to put the best Boltz result first.

```
for each candidate in PSICHIC-rank order:
    score = cache_lookup(candidate)     # μs
    if cache miss:
        score = boltz_inference(candidate)  # 45–150 s
    update_submission_with_best_so_far()   # immediate reorder
```

**Guarantee:** Even if the epoch timer fires after molecule 1 of 5, the submission has
already been reordered to put the best Boltz score (of the 1 scored so far) at position 0.
With all-cached runs the full reorder is nearly instantaneous.

**No efficiency loss:** Each Boltz call takes the same GPU time regardless of batch size
(the protein–ligand forward pass is per-pair). The one extra Python overhead per molecule
is negligible vs. 45–150 s inference.

**Adaptive timing accuracy improves:** The trigger update now uses the actual time for one
molecule (not total batch time ÷ N), which is more accurate because Boltz's per-molecule
time is constant and independent of batch composition.

---

## Implemented Optimisations (continued)

### I. PSICHIC Ligand-Efficiency Scoring (`neurons/miner.py`)

The Boltz-2 scoring formula is:
```
boltz_score = (affinity_probability_binary - affinity_pred_value) / heavy_atom_count
```

Previously, PSICHIC pre-filtering ranked candidates by raw binding affinity:
```python
combined_score = target_affinity - antitarget_weight × antitarget_affinity
```

This could promote large molecules (30–35 HA) with good absolute affinity over small
molecules (15–20 HA) with equally strong per-atom binding — the opposite of what Boltz
actually rewards.

**New formula:**
```python
combined_score = (target_affinity - antitarget_weight × antitarget_affinity) / heavy_atoms
```

Effect: the top-N candidates passed to Boltz-2 are now the **most ligand-efficient**
molecules by PSICHIC score — directly aligned with the Boltz denominator.  On hardware
where only 3–5 Boltz calls fit in the trigger window, this maximises the chance that the
molecule at position 0 is the epoch winner.

### J. Global Candidate Pool (`neurons/miner.py`)

**Problem with batch-replacement approach:** When PSICHIC finds a new best batch (higher
entropy-weighted score-sum), `candidate_molecules` is replaced entirely with that batch's
top-10.  A molecule that ranked #1 in an early chunk could be lost if a later chunk's
*aggregate* score is higher, even if no individual molecule in the later chunk beats it.

**Fix:** `state['global_candidate_pool']` accumulates the top-20 molecules (by per-atom
PSICHIC score) across **all chunks streamed this epoch**.  After each chunk:

```python
combined_pool = pd.concat([global_candidate_pool, top_molecules])
combined_pool.drop_duplicates(subset=['product_name'])
combined_pool.sort_values(by=['combined_score'], ascending=False)
global_candidate_pool = combined_pool.head(20)
```

`run_boltz_prescoring` now draws candidates from `global_candidate_pool` (falling back to
`candidate_molecules` if the pool is empty).  This ensures Boltz always evaluates the 5
highest-efficiency molecules found across the whole epoch, regardless of when they appeared.

**Zero submission-format risk:** the global pool only contains `product_name` values from
SAVI-2020 streaming (rxn:3 / rxn:5 format), so all candidates remain valid for submission.

**Memory cost:** 20 rows × ~5 columns ≈ negligible.

---

## Implemented Optimisations (continued)

### N. SALSA — Stochastic Approximate Ligand Scoring and Optimisation ✅ Implemented

Iterative hill-climbing over chemical space using bioisosteric atom substitution.
Starting from the best PSICHIC-scored seed, SALSA perturbs its atoms (C↔N, O↔S,
Cl↔F, etc.) and maps each perturbation back to the nearest molecule in the
`savi_stream_pool` via Tanimoto similarity (Morgan r=2).  All returned hits are
valid SAVI-2020 product names from the current epoch's streamed pool — they can be
added directly to `global_candidate_pool` and validated with Boltz-2.

**Files changed:**
- `utils/salsa.py` — new module (`generate_perturbations`, `nearest_pool_molecules`,
  `run_salsa_search`)
- `neurons/miner.py` — imports `run_salsa_search`; adds `savi_stream_pool` and
  `salsa_run_this_epoch` state fields; SALSA trigger in PSICHIC loop; epoch reset

**State fields added:**

| Field | Type | Purpose |
|-------|------|---------|
| `savi_stream_pool` | DataFrame | All PSICHIC-scored molecules this epoch, up to 5000 rows |
| `salsa_run_this_epoch` | bool | Prevents duplicate SALSA runs per epoch |

**Trigger conditions:**
- `savi_stream_pool` ≥ 500 molecules
- `blocks_until_epoch` > `boltz_trigger_blocks × 1.5` (SALSA fires before Boltz window)
- `salsa_run_this_epoch == False`

**Algorithm (3 rounds, 60 perturbations/round):**

```
seed ← global_candidate_pool.iloc[0]['product_smiles']
for round in 1..3:
    perturbations ← generate_perturbations(seed, n_max=60)
    for each perturbation:
        hit ← nearest_pool_molecules(perturbation, savi_stream_pool, top_k=1)
        collect unique hits
    best_hit ← max(hits, key=combined_score)
    seed ← best_hit['product_smiles']
return top-5 hits by combined_score
```

**Typical timing:** 3 rounds × 60 variants × ~1 ms each = ~180 ms (CPU, RDKit + Morgan FP)
→ negligible compared to Boltz-2 inference.

**Expected benefit:** by directing the nearest-neighbour search toward bioisosterically
optimised scaffolds rather than random SAVI-2020 streaming, SALSA can surface molecules
that score higher per atom under the Boltz-2 formula — especially later in an epoch when
the pool has grown large (1000–5000 molecules).

## Implemented Optimisations (continued)

### O. GradientGA — Gradient-Guided Genetic Algorithm ✅ Implemented

Population-based search that complements SALSA's single-point hill-climbing.
Maintains a pool of ~50 SAVI-2020 molecules across 5 generations per epoch.

**Files changed:**
- `utils/genetic.py` — new module (`brics_crossover`, `tournament_select`,
  `run_gradient_ga`)
- `neurons/miner.py` — imports `run_gradient_ga`; adds `ga_run_this_epoch` state
  field; GA trigger fires after SALSA and before Boltz window

**Algorithm (5 generations, population=50):**

```
seed population ← global_candidate_pool + random sample from savi_stream_pool
for gen in 1..5:
    pairs ← tournament_select(population, n_pairs=25, k=3)
    for parent_a, parent_b in pairs:
        offspring ← brics_crossover(parent_a, parent_b, max_offspring=3)
        mutants   ← generate_perturbations(parent_a, n_max=5)   # from salsa.py
        for each candidate in offspring + mutants:
            hit ← nearest_pool_molecule(candidate, savi_pool_df)  # Tanimoto r=2
            if hit not already seen: add to new_hits
    population ← top-50(population ∪ new_hits)  # elitism via sort
return top-5 molecules from all generations
```

**State fields added:**

| Field | Type | Purpose |
|-------|------|---------|
| `ga_run_this_epoch` | bool | Prevents duplicate GA runs per epoch |

**Trigger conditions:**
- `savi_stream_pool` ≥ 500 molecules
- `salsa_run_this_epoch == True` (GA runs after SALSA so it seeds from SALSA hits)
- `blocks_until_epoch` > `boltz_trigger_blocks + 20`
- `ga_run_this_epoch == False`

**Expected benefit:** BRICS crossover recombines functional fragments from the
best PSICHIC-scored molecules found across all streaming chunks.  Unlike SALSA
(which hill-climbs from a single seed), the GA maintains diversity and can
escape local optima by mixing fragments from structurally different parents.
The nearest-SAVI-2020 mapping keeps every candidate valid for submission.

**Typical timing:** ~1–3 s (CPU) for 5 generations — negligible vs Boltz inference.

---

## Future Optimisation Opportunities

### A. (Implemented — see §O above)

### C. Multi-molecule entropy bonus

When the validator increases `num_molecules_boltz > 1`, the entropy bonus activates
(`ranking.py` lines 109–118). The miner should respond by submitting a set of molecules
with high MACCS fingerprint diversity, not just the single best binder.

Implementation: after Boltz pre-scoring, fill remaining submission slots greedily with
molecules that maximise `compute_maccs_entropy()` relative to the already-selected set.

### D. Binding-pocket guidance

`config.yaml` exposes `binding_pocket`, `max_distance`, and `force`. When the validator
sets a pocket constraint, Boltz-2 adds a soft or hard guidance term to steer the diffusion
towards specific residues. Miners could pre-filter for molecules whose docked pose (from
a fast docking tool like Vina or Gnina) is predicted to sit inside the specified pocket.

### E. Persistent Boltz score cache (disk) ✅ Implemented

~~The current in-memory cache resets on process restart.~~ A SQLite cache keyed by
`(canonical_smiles, protein_code)` now accumulates scores across restarts and across
miners sharing a machine.

**Implementation** (`neurons/miner.py`):

```
BOLTZ_CACHE_DB = <repo_root>/boltz_score_cache.db
```

- `_init_boltz_cache_db(db_path)` — creates the `boltz_cache` table on first run.
- `_disk_cache_get(db_path, smiles, protein)` — returns cached score or `None`.
- `_disk_cache_put(db_path, smiles, protein, score)` — upserts score (silently ignores errors).

Cache lookup order in `run_boltz_prescoring`:
1. In-memory dict `state['boltz_score_cache']` (microseconds)
2. Disk SQLite `boltz_score_cache.db` (microseconds, warms in-memory on hit)
3. GPU inference via `BoltzWrapper` (45–150 s) → stored in both layers

Result: a molecule scored in a previous session is never re-run on GPU.

### K. Dynamic Boltz candidate budget (`neurons/miner.py`)

**Problem with fixed `max_candidates=5`:** At the default 100-block trigger, an A100 can score
~21 molecules (960 s available ÷ 45 s/mol), but the hard-coded ceiling of 5 left 16 slots
unused. On the first epoch (before GPU speed is known) this was a significant missed opportunity.

**Fix:** the Boltz trigger now computes `_dyn_max` from available epoch time and a measured (or
estimated) per-molecule inference time:

```python
_t_per_mol = state.get('boltz_time_per_mol', 150.0)  # conservative RTX 3090 default
_avail_secs = max(0.0, blocks_until_epoch * 12 - 240)  # 4-min safety margin
_dyn_max = max(3, min(20, int(_avail_secs / _t_per_mol)))
await run_boltz_prescoring(state, max_candidates=_dyn_max)
```

Effect on first epoch (100-block trigger, `boltz_time_per_mol` not yet measured):

| Hardware | Estimated t/mol | _avail_secs | _dyn_max |
|----------|-----------------|-------------|----------|
| A100 (actual 45 s) | 150 s (default) | 960 s | **6** (+1 vs old 5) |
| RTX 4090 (actual 90 s) | 150 s | 960 s | **6** |
| RTX 3090 (actual 150 s) | 150 s | 960 s | **6** |

After the first real run (say, A100 → `boltz_time_per_mol = 45 s`) the trigger adapts:

| Hardware | Measured t/mol | Trigger (blocks) | _avail_secs | _dyn_max |
|----------|----------------|------------------|-------------|----------|
| A100 | 45 s | 39 | 228 s | **5** |
| RTX 4090 | 90 s | 58 | 456 s | **5** |
| RTX 3090 | 150 s | 83 | 756 s | **5** |

The dynamic formula converges to ~5 after calibration (the adaptive trigger is sized to hold
exactly N candidates). Its real benefit is **first-epoch conservative over-scoring** (+1 molecule
for free) and **correct handling of abnormally wide windows** (e.g., miner restarts when trigger
is already 100 but GPU is fast).

### L. Epoch-end guard inside the Boltz loop (`neurons/miner.py`)

**Problem:** `run_boltz_prescoring` had no check on remaining epoch time inside its per-molecule
loop. If the epoch ended mid-loop (e.g., the 4th of 6 molecules is scoring and the epoch fires),
the 5th and 6th Boltz calls would start pointlessly — burning 90–300 s of GPU time on a molecule
that would never affect this epoch's submission.

**Fix:** before each cache-miss GPU inference, a lightweight subtensor block query confirms
that at least 5 blocks remain:

```python
try:
    _curr_blk = await state['subtensor'].get_current_block()
    _next_ep = ((_curr_blk // state['epoch_length']) + 1) * state['epoch_length']
    if _next_ep - _curr_blk < 5:
        bt.logging.info("epoch ends in <5 blocks — stopping Boltz early")
        break
except Exception:
    pass  # subtensor unavailable; proceed anyway
```

Cache hits (microseconds) are never interrupted. Only cache-miss GPU inference calls are gated.
The `except` clause ensures a network hiccup on the block query doesn't abort Boltz incorrectly.

The anytime guarantee (§H) means the best score found so far is already at position 0 when the
loop exits early — zero regression in submission quality.

### M. `boltz_time_per_mol` state field (`neurons/miner.py`)

The adaptive trigger (§G) recalculated the trigger threshold in blocks each run but did not store
the raw per-molecule GPU time anywhere accessible to the trigger call site. Dynamic max_candidates
(§K) requires this value at the point where `_dyn_max` is computed — before the wrapper is even
instantiated.

**Fix:** after every successful single-molecule Boltz inference, the measured time is stored:

```python
state['boltz_time_per_mol'] = elapsed  # seconds per molecule, persists across epochs
```

This value survives epoch rollovers and process restarts are handled gracefully via the
`state.get('boltz_time_per_mol', 150.0)` default at the call site.

---

## Validator Config Reference

These parameters in `config/config.yaml` directly affect what the miner should optimise:

| Parameter | Current value | Effect on miner |
|-----------|---------------|-----------------|
| `boltz_weight` | 1.0 | 100% of rewards go to Boltz-2 winner |
| `num_molecules_boltz` | 1 | Only first molecule in submission is scored |
| `sample_selection` | "first" | Put best Boltz molecule first |
| `combination_strategy` | "heavy_atom_normalization" | Penalises large molecules |
| `boltz_metric` | `["affinity_probability_binary", "affinity_pred_value"]` | See scoring formula above |
| `boltz_mode` | "max" | Higher score wins |
| `min_heavy_atoms` | 10 | Lower bound; Boltz efficiency favours 15–30 |
| `max_rotatable_bonds` | 10 | Hard constraint |

---

## Files Changed

| File | Change |
|------|--------|
| `neurons/miner.py` | Added `is_boltz_safe_smiles` + `max_heavy_atoms` filters; `run_boltz_prescoring()` with two-tier cache; Boltz trigger at 100 blocks (was 50); `boltz_score_cache` + `boltz_cache_db` state fields; `_init_boltz_cache_db`, `_disk_cache_get`, `_disk_cache_put` helpers; fixed `entropy_weight` → `entropy_start_weight` AttributeError; pharmacophore pre-filter (§F); adaptive trigger using `boltz_trigger_blocks` state field (§G); anytime incremental scoring — one molecule at a time with immediate reorder (§H); PSICHIC ligand-efficiency scoring — divide combined_score by heavy_atoms (§I); global candidate pool — top-20 across all epoch chunks for Boltz (§J); dynamic max_candidates from available epoch time + `boltz_time_per_mol` state (§K); epoch-end guard before each cache-miss Boltz inference (§L); `boltz_time_per_mol` persisted in state (§M); SALSA stream pool + trigger + epoch reset (§N) |
| `boltz/wrapper.py` | Added `last_inference_duration` field populated after each `predict()` call (§G) |
| `config/config.yaml` | Added `max_heavy_atoms: 35` |
| `config/config_loader.py` | Loads and exposes `max_heavy_atoms` |
| `utils/molecules.py` | Added `get_canonical_smiles()` |
| `utils/__init__.py` | Exported `get_canonical_smiles`; exported SALSA functions (§N) |
| `utils/salsa.py` | New: SALSA algorithm — `generate_perturbations`, `nearest_pool_molecules`, `run_salsa_search` (§N) |
| `BOLTZ2_INTEGRATION.md` | This file |
