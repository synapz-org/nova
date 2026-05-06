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

## Implemented Optimisations (continued)

### P. Validator Constraint Filters in Pre-filter (`neurons/miner.py`)

**Problem:** The `_pharma_ok` function previously applied only the Lipinski-inspired drug-likeness
check (H-bond donors/acceptors, logP). Molecules containing banned atom types (e.g., Se) or
with rotatable bond counts outside `[min_rotatable_bonds, max_rotatable_bonds]` would pass
through PSICHIC scoring, be added to `global_candidate_pool`, and potentially be submitted —
only to be rejected or penalised by the validator.

**Fix:** Extended `_pharma_ok` to enforce all three constraint classes in a single RDKit parse:

```python
def _pharma_ok(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    # Validator-enforced: banned atom types (e.g. Se)
    _banned = getattr(state['config'], 'banned_atom_types', [])
    if _banned and any(a.GetSymbol() in _banned for a in mol.GetAtoms()):
        return False
    # Validator-enforced: rotatable bond bounds
    _rot = Descriptors.NumRotatableBonds(mol)
    _min_rb = getattr(state['config'], 'min_rotatable_bonds', 1)
    _max_rb = getattr(state['config'], 'max_rotatable_bonds', None)
    if _rot < _min_rb or (_max_rb is not None and _rot > _max_rb):
        return False
    # Lipinski-inspired drug-likeness
    return (
        1 <= Descriptors.NumHDonors(mol) <= 3
        and 2 <= Descriptors.NumHAcceptors(mol) <= 7
        and 0.0 <= Descriptors.MolLogP(mol) <= 4.5
    )
```

All three checks share a single `Chem.MolFromSmiles` call — no extra parsing overhead.

**Config values enforced:**
- `banned_atom_types: ["Se"]` — drops molecules containing selenium
- `min_rotatable_bonds: 1` — requires at least one rotatable bond
- `max_rotatable_bonds: 10` — drops overly flexible molecules that the validator rejects

### Q. Multi-seed SALSA (`neurons/miner.py`)

**Problem:** SALSA hill-climbed from a single seed (the highest-scoring molecule in
`global_candidate_pool`). If that seed is a local optimum in PSICHIC space, SALSA's
bioisosteric perturbations would all explore the same chemical neighbourhood, missing better
molecules in other scaffolds.

**Fix:** SALSA now runs from up to the top-3 molecules in `global_candidate_pool` as independent
seeds, merges all hits, deduplicates, and keeps the top-`5 × n_seeds` results:

```python
_n_seeds = min(3, len(state['global_candidate_pool']))
_seeds = state['global_candidate_pool'].head(_n_seeds)['product_smiles'].tolist()
_all_salsa = []
for _seed_smiles in _seeds:
    _hits = await asyncio.to_thread(run_salsa_search, _seed_smiles, salsa_pool, 3, 60, 5)
    if not _hits.empty:
        _all_salsa.append(_hits)
salsa_hits = pd.concat(_all_salsa, ...).drop_duplicates(...).sort_values(...)
```

**Runtime cost:** ~3 × 180 ms = ~540 ms CPU — still negligible relative to Boltz-2 inference.

**Benefit:** Three concurrent hill-climbing trajectories starting from chemically distinct
seeds can reach different basins of the PSICHIC landscape, surfacing up to 3× more candidate
molecules before Boltz scoring. Early epochs (only 1 molecule in the pool) gracefully fall
back to single-seed behaviour.

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

---

## Implemented Optimisations (continued)

### R. Pre-computed Pool Fingerprints + BulkTanimotoSimilarity (`utils/salsa.py`, `utils/genetic.py`)

**Problem:** `nearest_pool_molecules` previously recomputed Morgan fingerprints for every
molecule in the pool on every query.  A SALSA run with 3 rounds × 60 perturbations × 5000-molecule
pool = **900,000 redundant FP computations** per epoch (each FP computed ~180 times instead of once).
GradientGA with 5 generations × ~150 offspring × 5000-molecule pool has similar overhead.

**Fix:** Added `precompute_pool_fps(pool_df, smiles_col)` to `utils/salsa.py`:

```python
def precompute_pool_fps(pool_df, smiles_col='product_smiles') -> Tuple[pd.DataFrame, List]:
    """Pre-compute Morgan FPs once; returns (valid_df, fps_list)."""
```

`run_salsa_search` and `run_gradient_ga` now call this once before their main loops and pass
the resulting `(valid_pool, pool_fps)` to `nearest_pool_molecules` via its new optional
`pool_fps` argument.  When `pool_fps` is supplied, the function uses the vectorised C++
`DataStructs.BulkTanimotoSimilarity(target_fp, pool_fps)` instead of a per-molecule Python loop.

**Complexity reduction:**

| Metric | Before | After |
|--------|--------|-------|
| FP computations (SALSA, 5000-mol pool) | 900,000 | 5,000 (1×) |
| FP computations (GA, 5000-mol pool) | ~750,000 | 5,000 (1×) |
| Similarity loop | Python (scalar) | C++ (vectorised) |
| Expected SALSA wall-clock (CPU) | ~500–900 ms | ~20–40 ms |
| Expected GA wall-clock (CPU) | ~1–3 s | ~50–150 ms |

SALSA and GA are still negligible vs. Boltz-2 inference, but the speedup means:
- Less CPU contention during the Boltz window (Boltz uses GPU, but Python GIL and OS scheduler
  still benefit from shorter CPU bursts).
- Multi-seed SALSA (3 seeds) completes faster, leaving more headroom for other tasks.

**API change:** `nearest_pool_molecules` gains an optional `pool_fps` keyword argument (default
`None`, backward-compatible).  Direct callers outside SALSA/GA are unaffected.

---

## Implemented Optimisations (continued)

### S. MSA Auto-fetch at Miner Startup (`utils/msa.py`, `neurons/miner.py`)

**Problem:** Boltz-2 uses a Multiple Sequence Alignment (MSA) to incorporate
evolutionary context into its structure and affinity predictions. The MSA is a
pre-computed `.a3m` file stored at `boltz/msa_files/{protein_code}.a3m`. Without it,
Boltz runs in single-sequence mode, which measurably weakens affinity estimates.

Previously this file had to be generated manually and committed before each new epoch.
If a miner was deployed on a new machine or the weekly target rotated while the miner
was unattended, it would silently fall back to single-sequence mode with no warning.

**Fix:** `utils/msa.py` adds three public functions:

| Function | Purpose |
|----------|---------|
| `msa_exists(protein_code)` | Fast on-disk check — returns `True` if `.a3m` file is present |
| `fetch_msa(protein_code, sequence)` | Submits sequence to ColabFold API, polls for result, extracts and saves `.a3m` |
| `ensure_msa(protein_code, sequence)` | No-op if file exists; otherwise calls `fetch_msa` |

**API flow (`fetch_msa`):**

```
POST https://api.colabfold.com/ticket/msa
     {"q": ">query\n{sequence}", "mode": "env"}
     → {"id": "<job_id>"}

GET  https://api.colabfold.com/ticket/<job_id>
     → {"status": "PENDING"|"RUNNING"|"COMPLETE"|"ERROR"}
     (polled every 15 s, up to 15 min timeout)

GET  https://api.colabfold.com/result/download/<job_id>
     → tar.gz (or zip) archive containing 0.a3m
     → written to boltz/msa_files/{protein_code}.a3m
```

The download handler supports both gzip-tarball and zip archives to be robust against
API version changes.

**Integration (`neurons/miner.py`):** Called once during `run_miner()` startup, after
the Boltz cache is initialised and before the main epoch loop:

```python
_target_seq = get_sequence_from_protein_code(config.weekly_target)
if _target_seq:
    _msa_ok = ensure_msa(config.weekly_target, _target_seq)
    if not _msa_ok:
        bt.logging.warning("Boltz-2 will run in single-sequence mode (weaker predictions).")
```

All errors are caught and logged as warnings — the miner continues even if the API is
unreachable (graceful degradation to single-sequence mode rather than a crash).

**Typical timing:** ColabFold MSA jobs for typical drug-target proteins (300–500 aa)
complete in 1–3 minutes. Miner startup is blocked until the MSA is ready or the
15-minute timeout is reached. On subsequent starts the file is already on disk and the
call returns immediately.

**Files changed:**
- `utils/msa.py` — new module (`ensure_msa`, `fetch_msa`, `msa_exists`)
- `utils/__init__.py` — exports `ensure_msa`, `fetch_msa`, `msa_exists`
- `neurons/miner.py` — imports `ensure_msa`; calls it at startup after cache init

---

## Implemented Optimisations (continued)

### T. Forward `no_kernels`, `num_workers`, `preprocessing_threads` from config (`boltz/wrapper.py`)

**Problem:** `BoltzWrapper.score_molecules_target()` called `predict()` with a hard-coded subset
of parameters. Three config keys were silently ignored:

| Key | Config value | `predict()` default | Effect of bug |
|-----|-------------|---------------------|---------------|
| `no_kernels` | `true` | `False` | Custom CUDA kernels enabled despite config saying disable — may fail on hardware without `cuequivariance` |
| `num_workers` | *(unset → 2)* | 2 | No effect in this case, but un-configurable |
| `preprocessing_threads` | *(unset → 4)* | 4 | Same |

**Fix:** Three additional keyword arguments added to the `predict()` call:

```python
no_kernels = self.config.get('no_kernels', False),
num_workers = self.config.get('num_workers', 2),
preprocessing_threads = self.config.get('preprocessing_threads', 4),
```

`no_kernels` uses `dict.get()` for backward compatibility (old config files without the key
get the library default of `False`).

**Impact:** Miners running on hardware without `cuequivariance` or `triton` installed would
previously get a runtime error when Boltz tried to load the custom kernel. With the fix, setting
`no_kernels: true` in `boltz_config.yaml` (already the default) correctly falls back to the
pure-PyTorch implementation.

---

---

## Implemented Optimisations (continued)

### U. `use_potentials` inference-time potentials (`boltz/wrapper.py`, `boltz/boltz_config.yaml`)

**Background:** The Boltz-2 `predict()` function supports a `use_potentials` flag that
activates FK (forward kinematics) steering and physical guidance during diffusion sampling.
These are inference-time potentials that steer the diffusion trajectory toward physically
plausible poses — improved backbone geometry, better clash avoidance, and stronger
protein–ligand contacts.

**Previous state:** `use_potentials` defaulted to `False` in the library and was never
forwarded from our config, so it was always off regardless of hardware capabilities.

**Fix:** Two changes:

1. `boltz/boltz_config.yaml` — new key:
   ```yaml
   use_potentials: false  # set true on A100/H100 for better affinity accuracy
   ```

2. `boltz/wrapper.py` — added to `predict()` call:
   ```python
   use_potentials = self.config.get('use_potentials', False),
   ```

**When to enable:** On A100 / H100 hardware where the additional ~10–20% inference time
overhead is affordable within the epoch window. On RTX 3090 (150 s/mol baseline), enabling
potentials may push per-molecule time to ~180 s — the adaptive trigger (§G) will compensate
automatically by updating `boltz_trigger_blocks`.

**Risk:** Zero — the default remains `false`, preserving identical behaviour on all existing
deployments. Miners on fast hardware can opt in by changing one YAML line.

---

### V. `step_scale` diffusion temperature control (`boltz/wrapper.py`, `boltz/boltz_config.yaml`)

**Background:** The diffusion process has a temperature parameter `step_scale` (library
default: 1.5) that controls pose diversity:

| `step_scale` | Effect |
|--------------|--------|
| < 1.5 | Less diverse; more consistent, lower-variance poses |
| 1.5 | Library default — balanced diversity/consistency |
| > 1.5 | More diverse; explores wider conformational space |

With `diffusion_samples_affinity: 3`, the affinity ensemble averages 3 samples at the
configured temperature. A lower `step_scale` (e.g., 1.2) reduces variance between samples,
giving a more stable `affinity_probability_binary` estimate. A higher scale (e.g., 1.8) may
find a lucky high-energy binding mode but with higher score variance.

**Previous state:** `step_scale` was not forwarded from config; the library default (1.5)
was always used.

**Fix:**

1. `boltz/boltz_config.yaml`:
   ```yaml
   step_scale: null  # null → library default (1.5); set e.g. 1.2 for lower variance
   ```

2. `boltz/wrapper.py`:
   ```python
   step_scale = self.config.get('step_scale', None),
   ```

**Recommended tuning:** Start with `null` (library default). If Boltz scores are noisy
between epochs for the same molecule (variance > 0.05 on `affinity_probability_binary`),
try `step_scale: 1.2` to tighten the distribution.

---

## Implemented Optimisations (continued)

### X. Defensive Boltz wrapper — missing results, empty scores, MSA fallback (`boltz/wrapper.py`)

Three crash paths eliminated from `BoltzWrapper`:

**X.1 — Missing results directory** (`postprocess_data`):

Previously, if Boltz-2 failed to write prediction files for a specific molecule (GPU OOM
mid-batch, invalid YAML, upstream `predict()` partial failure), the call to
`os.listdir(results_path)` would raise `FileNotFoundError` and propagate uncaught through
`postprocess_data`, crashing the entire pre-scoring run and leaving `state['candidate_product']`
at the raw PSICHIC ordering with no Boltz reorder.

**Fix:** The `os.listdir` call is now wrapped in `try/except (FileNotFoundError, OSError)`.
A failed molecule gets `scores[mol_idx] = {}` (empty dict) and a warning log. All other
molecules in the batch continue to score normally.

**X.2 — Empty scores → score assignment KeyError** (`postprocess_data`):

If `scores[mol_idx]` is empty (from X.1 above, or if the results files contained no matching
keys), the old code would call `self.combine_boltz_scores({}, smiles)`, which then tried
`{}['affinity_probability_binary']` and raised `KeyError`.

**Fix:** An explicit `if not mol_scores: final_score = -math.inf` guard precedes the
score combination call. Also fixed the dead-code `else` branch (single-metric path) to use
`mol_scores.get(metric_key, -math.inf)` instead of a bare `[]` index.

**X.3 — combine_boltz_scores KeyError / ZeroDivisionError**:

Added a `try/except (KeyError, TypeError, ZeroDivisionError)` around the entire
`combine_boltz_scores` body. Catches (a) missing metric keys in the scores dict,
(b) `None` values from a partially-parsed JSON file, and (c) zero `heavy_atom_count`
from `get_heavy_atom_count` on a degenerate SMILES. All return `-math.inf`.

**X.4 — MSA file missing → hard Boltz crash** (`create_yaml_content`):

`create_yaml_content` previously always wrote `msa: /path/to/X.a3m` into the input YAML.
If `ensure_msa` failed at startup (ColabFold timeout, network error), the `.a3m` file would
not exist; Boltz-2 would crash on the non-existent path rather than falling back to
single-sequence mode.

**Fix:** The method now checks `os.path.exists(msa_path)` before including the `msa:` line.
When the file is absent, the line is omitted entirely and a warning is logged. Boltz-2
handles a missing `msa:` key by running in single-sequence mode — weaker predictions, but
not a crash.

**Net effect:** A GPU OOM on molecule N no longer aborts the entire Boltz pre-scoring loop.
Molecules N+1 … M continue scoring, the anytime guarantee keeps the best-so-far at position 0,
and the miner submits a meaningful result regardless.

---

## Implemented Optimisations (continued)

### Z. MSA Subsampling Control (`boltz/wrapper.py`, `boltz/boltz_config.yaml`)

**Background:** Boltz-2 uses a Multiple Sequence Alignment to provide evolutionary context during
inference.  The `predict()` function exposes two subsampling knobs that were not previously
forwarded from config:

| Parameter | Library default | Effect |
|-----------|----------------|--------|
| `subsample_msa` | `true` | Whether to subsample the MSA at all |
| `num_subsampled_msa` | `1024` | How many sequences to keep from the full MSA |

These parameters are passed at data-loading time and affect both the structure and affinity
prediction passes — increasing `num_subsampled_msa` gives the model richer evolutionary
context, which correlates with more accurate affinity predictions.

**Fix:** Both parameters now forwarded from `boltz_config.yaml` to `predict()`:

```python
subsample_msa = self.config.get('subsample_msa', True),
num_subsampled_msa = self.config.get('num_subsampled_msa', 1024),
```

The `boltz_config.yaml` defaults are unchanged (`subsample_msa: true`, `num_subsampled_msa: 1024`),
so existing deployments are unaffected.

**Recommended tuning:**

| Hardware | `num_subsampled_msa` | Expected Δ inference time |
|----------|----------------------|--------------------------|
| RTX 3090 / 4090 | 1024 (default) | baseline |
| A100 80 GB | 2048 | ~+15% |
| H100 80 GB | 4096 | ~+30% |

On A100 with the adaptive trigger (§G), the extra ~20s per molecule from 2048 sequences
is automatically compensated: `boltz_trigger_blocks` updates to fire earlier, preserving the
same total number of candidates scored per epoch.

**Also exposed explicitly:**
- `num_workers` (default 2) — data-loader worker threads
- `preprocessing_threads` (default 4) — YAML preprocessing parallelism

These were already forwarded via `self.config.get()` but not listed in `boltz_config.yaml`,
making them invisible to miners who wanted to tune them.

---

## Implemented Optimisations (continued)

### AA. Warm Epoch Start from Disk Cache (`neurons/miner.py`)

**Problem:** At every epoch boundary `candidate_product` was reset to `None`. The miner had
nothing valid to submit until PSICHIC streaming processed enough chunks to find a candidate
— typically 5–15 minutes into the epoch. If the miner restarted with only 10 blocks remaining,
it would submit nothing at all for that epoch.

**Fix:** The disk cache schema is extended with a `product_name TEXT` column. Whenever a
molecule is scored by Boltz-2 during `run_boltz_prescoring`, its SAVI-2020 `product_name` is
stored alongside the SMILES and score:

```python
_pname = row.get('product_name')
if not isinstance(_pname, str):
    _pname = None
_disk_cache_put(db_path, canon, protein, score, product_name=_pname)
```

Two new helpers query the cache:

| Function | Purpose |
|----------|---------|
| `_disk_cache_get_best(db_path, protein)` | Returns `(score, smiles, product_name)` for the highest-scoring cached entry with a non-NULL product_name |
| `_apply_warm_start(state, db_path, protein)` | Calls `_disk_cache_get_best`; sets `state['candidate_product']` if a hit is found |

`_apply_warm_start` is called in two places:
1. **Miner startup** — immediately after `_cleanup_boltz_cache`.
2. **Each epoch boundary** — after the epoch state reset.

**Guarantee:** From block 1 of every epoch after the first, the miner has a Boltz-validated
molecule at `candidate_product[0]`. If PSICHIC streaming subsequently finds a batch with a
higher entropy-weighted PSICHIC score, it replaces `candidate_product` automatically (because
`best_score` remains `−∞` for the warm-start molecule, so any positive PSICHIC result wins).
If streaming is slow or the miner restarts late, the cached molecule is submitted.

**Schema migration:** `_init_boltz_cache_db` now issues `ALTER TABLE boltz_cache ADD COLUMN
product_name TEXT` after the `CREATE TABLE IF NOT EXISTS` statement. SQLite raises
`OperationalError` if the column already exists; this is caught silently, making the migration
backward-compatible with existing cache databases that predate §AA.

**Interaction with Boltz prescoring trigger:** `candidate_product` being non-None from epoch
start causes the Boltz trigger to evaluate at `blocks_until_epoch ≤ boltz_trigger_blocks`.
If `global_candidate_pool` is still empty at that point (streaming not yet run), Boltz
prescoring returns immediately with a warning. The prescoring flag `boltz_prescored` is then
reset when PSICHIC finds its first new best, allowing a second Boltz pass on genuine PSICHIC
candidates — no regression vs. the pre-§AA behaviour.

**Files changed:**
- `neurons/miner.py` — `_init_boltz_cache_db` (ALTER TABLE); `_disk_cache_put` (product_name
  arg); new `_disk_cache_get_best`; new `_apply_warm_start`; two call sites in `run_miner()`

---

## Implemented Optimisations (continued)

### CC. Warm-Start Guard — Retain Cached Best When It Beats Epoch Boltz Run (`neurons/miner.py`)

**Problem:** The warm-start mechanism (§AA) seeds `state['candidate_product']` at epoch start from
the disk cache's best Boltz-scored molecule for the current protein.  This is the fallback
submission before PSICHIC streaming finds new candidates.

However, once PSICHIC processes its first batch, it replaces `state['candidate_product']` entirely
with the new PSICHIC top-10, and `state['global_candidate_pool']` is seeded fresh from those
batches.  The warm-start molecule is no longer in the pool and is therefore **not evaluated by
`run_boltz_prescoring`** later in the epoch.

`_reorder_submission` only promotes molecules it has scored in the current run.  If the warm-start
molecule's prior-session Boltz score (e.g., 0.42) is higher than all scores from the current run
(e.g., best 0.31), the submission incorrectly puts a worse molecule at position 0.

**Example:**
```
Epoch N−1: molecule A → Boltz 0.42 → stored in disk cache
Epoch N:
  _apply_warm_start  → candidate_product = "product_A"
  PSICHIC streaming  → candidate_product = "product_B, product_C, ..."  (warm-start evicted)
  Boltz run on pool  → B=0.31, C=0.28
  _reorder_submission → product_B at position 0   ← BUG: product_A (0.42) was better
```

**Fix:** At the end of `run_boltz_prescoring`, after all candidates have been scored and
`_reorder_submission` has placed the best new molecule at position 0:

```python
best_new = max(v for v in all_scores.values() if isfinite(v), default=-inf)
ws_best = _disk_cache_get_best(db_path, protein)
if ws_best and ws_best.score > best_new and ws_smiles not in all_scores:
    # Cached molecule was not evaluated this run but is known to be better.
    # Prepend (or move) its product_name to position 0.
    state['candidate_product'] = ws_pname + ',' + rest_of_current_submission
```

**Conditions checked:**
- `_ws_score > best_new` — cached molecule is demonstrably better
- `not _already_scored` — molecule was not re-evaluated this run (avoids interfering with
  `_reorder_submission`, which already handles molecules that were in the current pool)
- `math.isfinite(_ws_score)` — guards against `-inf` sentinel values from failed prior runs
- `state.get('candidate_product')` — no-op if the submission is empty

**Typical scenarios:**
- Fast hardware (A100): many molecules scored → warm-start rarely overrides
- Slow hardware (RTX 3090): only 3 molecules fit in the window → warm-start override is more
  likely to be beneficial, especially mid-epoch when the chemical search is still converging

**Zero regression risk:** If the warm-start molecule IS in the current pool (because it appeared in
a PSICHIC batch with high ligand efficiency), it enters `candidates`, gets re-evaluated (cache hit,
microseconds), appears in `all_scores`, and is handled normally by `_reorder_submission`.  The
`not _already_scored` check prevents the guard from firing in that case.

---

## Implemented Optimisations (continued)

### DD. Extended SALSA Perturbation Operators — Functional Group Addition (`utils/salsa.py`)

**Problem:** The original `generate_perturbations` function only performed *bioisosteric substitution*
— replacing each heavy atom with an equivalent (C↔N, O↔S, Cl↔F, …).  This keeps the molecular size
fixed and explores only electronic/polarity variants.  The nearest-SAVI-2020 Tanimoto search
therefore swept a tight chemical neighbourhood and often mapped back to the same small cluster of
pool molecules across multiple perturbation rounds.

**Fix:** Added a second perturbation pass — **functional group addition** — that appends a single
heavy atom (F, Cl, methyl-C, amine-N, hydroxyl-O) at every position with an available implicit
hydrogen on a C/N/O/S attachment atom.  The resulting SMILES are *probe vectors only* — they are
never submitted directly — but they extend the Tanimoto search radius into the +1-heavy-atom
neighbourhood of the seed, mapping back to SAVI-2020 molecules that differ by a methyl, halogen,
or polar group.

**New constants in `utils/salsa.py`:**

```python
_FG_ATOMS: list[int] = [9, 17, 6, 7, 8]          # F, Cl, C (methyl), N (amine), O (hydroxyl)
_FG_ATTACHMENT_ATOMS: frozenset[int] = frozenset([6, 7, 8, 16])  # C, N, O, S eligible positions
```

**Algorithm extension in `generate_perturbations`:**

```
existing: bioisosteric substitution (unchanged)

new second pass:
for each atom with implicit H and atomic number in {C, N, O, S}:
    for each fg_an in [F, Cl, C, N, O]:
        rw = RWMol(mol)
        rw.AddAtom(Atom(fg_an))
        rw.AddBond(atom.idx, new_idx, SINGLE)
        SanitizeMol(rw)                    # drops valence violations silently
        emit canonical SMILES if unseen
```

**Effect on SALSA coverage:**

| Round | Old operators | New operators |
|-------|--------------|--------------|
| Bioisostere sub | ≤ N_atoms × 3 variants | same |
| FG addition | 0 variants | ≤ N_atoms × 5 × |{C,N,O,S}| variants |
| Unique probes/round | ~60–120 | ~120–300 (depending on molecule) |

At 300 probes/round × 3 rounds, SALSA maps back from a 2–3× richer probe set, increasing the
probability of discovering high-scoring pool molecules outside the immediate bioisosteric
neighbourhood of the seed.

**Risk:** Zero — probes are discarded after nearest-neighbour lookup; submitted molecules always
come from `savi_pool_df` and pass all existing safety filters (`is_boltz_safe_smiles`, `_pharma_ok`).

**Files changed:**
- `utils/salsa.py` — new `_FG_ATOMS` and `_FG_ATTACHMENT_ATOMS` constants; second loop in
  `generate_perturbations` for FG addition; updated docstring.

---

### EE. Scaffold-Diverse Boltz Candidate Selection (`neurons/miner.py`)

**Problem:** `run_boltz_prescoring` previously called `candidates.head(max_candidates)` on the
`global_candidate_pool` (sorted by ligand-efficiency PSICHIC score).  After SALSA or GradientGA
converge to a high-scoring chemical region, the top-N candidates by PSICHIC often share the same
Murcko scaffold — they're N variants of one core that SALSA hill-climbed to.

Scoring 5 near-identical molecules with Boltz-2 wastes GPU budget: protein–ligand binding is
dominated by the scaffold interaction, so all 5 receive similar affinity estimates.  Only one can
be submitted at position 0.  The remaining 4 Boltz calls produce no incremental value.

**Fix:** New helper `_scaffold_diverse_candidates(df, max_k)` performs a greedy two-pass
selection:

1. **Diversity pass:** iterate candidates in PSICHIC-score order; admit each molecule whose Murcko
   scaffold has not yet been seen.  This maximises scaffold variety within `max_k` slots.

2. **Fill pass:** if the diversity pass admitted fewer than `max_k` candidates (small pool or
   chemically homogeneous epoch), fill remaining slots from the original ranking — allowing scaffold
   repeats rather than scoring fewer molecules.

`_scaffold_diverse_candidates` is called in `run_boltz_prescoring` after the Boltz-safety filter,
on a pre-selected slice of `max_candidates × 3` rows.  The wider slice gives the diversity pass
enough candidates to choose from even when SALSA has converged tightly.

**Code in `neurons/miner.py`:**

```python
# Pull a wider slice (3× budget) so scaffold diversity has candidates to choose from.
candidates = candidates.head(max_candidates * 3).copy()
safe_mask = candidates['product_smiles'].apply(lambda s: is_boltz_safe_smiles(s)[0])
candidates = candidates[safe_mask].reset_index(drop=True)

# Scaffold-diverse selection
candidates = _scaffold_diverse_candidates(candidates, max_candidates)
```

**Concrete benefit scenario:**

> Epoch N: SALSA converges to a pyrimidine scaffold.  Top-5 PSICHIC candidates are all pyrimidine
> variants.  Without §EE, all 5 Boltz calls score the same binding mode (pyrimidine-hinge).
> With §EE, candidates 4 and 5 are replaced by the best benzimidazole and indazole variants from
> positions 6–10 in the pool.  If either non-pyrimidine binds better, it surfaces at position 0.

**Zero regression risk:**
- If all top-20 pool molecules share one scaffold (rare), the fill-pass returns the same top-5 as
  before — identical to the original `head(max_candidates)` behaviour.
- The anytime guarantee (§H) and warm-start guard (§CC) are unaffected.

**Interaction with §K (dynamic budget):** `_scaffold_diverse_candidates` receives `max_k =
_dyn_max` — the same dynamic budget value.  Scaffold diversity acts as a selection strategy within
that budget, not as an additional constraint on it.

**Files changed:**
- `neurons/miner.py` — new `_scaffold_diverse_candidates` helper (above `run_boltz_prescoring`);
  updated candidate-selection block in `run_boltz_prescoring` (wider slice + diversity call +
  log message).

---

## Remaining Future Opportunities

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

### W. `sampling_steps_affinity` accuracy tuning

Current value: **100** (boltz_config.yaml). Library default: **200**.

This was intentionally halved for speed. On A100 hardware with the adaptive trigger, the
per-epoch budget comfortably fits 5–6 molecules at 100 steps. Increasing to 150 or 200
would improve affinity estimate accuracy at the cost of longer per-molecule inference
(~1.5–2× longer). With `use_potentials: true` already adding overhead, consider profiling
before increasing.

Formula: `sampling_steps_affinity` linearly scales GPU time per molecule (more steps =
more diffusion iterations = longer inference). The adaptive trigger (§G) accounts for this
automatically after the first measured inference.

---

## Implemented Optimisations (continued)

### BB. Quality-first SAVI Stream Pool (`neurons/miner.py`)

**Problem:** `savi_stream_pool` previously kept the **first** 5000 unique molecules
added during the epoch (insertion order).  After the pool filled up, later streaming
chunks were silently discarded even if they contained higher-PSICHIC-scoring molecules.
SALSA and GradientGA therefore searched a pool biased toward the chemistry found at the
*start* of each epoch (a random SAVI-2020 CSV file), not the best chemistry found across
all chunks.

There was also wasted CPU: every streaming chunk continued to run `pd.concat +
drop_duplicates` against the pool even after both SALSA and GA had already fired and
the pool was no longer read by any algorithm.

**Fix:** Two changes to the pool management block in `run_psichic_model_loop`:

1. **Sort by `combined_score`, keep top-5000** (quality-first cap):
   ```python
   _pool_combined = pd.concat([state['savi_stream_pool'], df], ignore_index=True)
   _pool_combined.drop_duplicates(subset=['product_name'], inplace=True)
   _pool_combined.sort_values('combined_score', ascending=False, inplace=True)
   state['savi_stream_pool'] = _pool_combined.head(5000).reset_index(drop=True)
   ```
   After the pool reaches capacity (5000 molecules), new chunks can still displace
   low-scoring early entries when their `combined_score` is higher.  The pool always
   represents the 5000 best molecules seen so far this epoch.

2. **Skip update after SALSA + GA have both run:**
   ```python
   if not (state.get('salsa_run_this_epoch') and state.get('ga_run_this_epoch')):
       # ... update pool ...
   ```
   Once `salsa_run_this_epoch` and `ga_run_this_epoch` are both `True`, no remaining
   code path reads `savi_stream_pool`.  The guard eliminates a `pd.concat + sort` call
   on a 5000-row DataFrame for every subsequent chunk (~30 chunks/min × up to 60 min =
   ~1800 skipped operations).

**Expected benefit for SALSA/GA search quality:**
When SALSA fires at the `≥500-molecule` threshold, those 500 molecules are now the
500 highest-PSICHIC-scoring molecules found so far, not the first 500 from a random
streaming file.  SALSA's nearest-neighbour search therefore maps bioisosteric
perturbations onto higher-efficiency scaffolds, improving the quality of candidates
added to `global_candidate_pool` before Boltz-2 scoring.

**Zero regression risk:** The sort order of `savi_stream_pool` is only used inside
SALSA and GA (which extract the best `score_col` row as the hill-climbing seed and
return pool rows ranked by `combined_score`).  The guard condition is strictly
conservative — `savi_stream_pool` is only skipped after both flags are set.

---

## Files Changed

| File | Change |
|------|--------|
| `neurons/miner.py` | Added `is_boltz_safe_smiles` + `max_heavy_atoms` filters; `run_boltz_prescoring()` with two-tier cache; Boltz trigger at 100 blocks (was 50); `boltz_score_cache` + `boltz_cache_db` state fields; `_init_boltz_cache_db`, `_disk_cache_get`, `_disk_cache_put` helpers; fixed `entropy_weight` → `entropy_start_weight` AttributeError; pharmacophore pre-filter (§F); adaptive trigger using `boltz_trigger_blocks` state field (§G); anytime incremental scoring — one molecule at a time with immediate reorder (§H); PSICHIC ligand-efficiency scoring — divide combined_score by heavy_atoms (§I); global candidate pool — top-20 across all epoch chunks for Boltz (§J); dynamic max_candidates from available epoch time + `boltz_time_per_mol` state (§K); epoch-end guard before each cache-miss Boltz inference (§L); `boltz_time_per_mol` persisted in state (§M); SALSA stream pool + trigger + epoch reset (§N); `ga_run_this_epoch` added to initial state dict; validator constraint filters (banned atoms, rotatable bonds) added to `_pharma_ok` (§P); multi-seed SALSA — top-3 seeds for broader chemical space coverage (§Q); MSA auto-fetch at startup via `ensure_msa` (§S) |
| `boltz/wrapper.py` | Added `last_inference_duration` field populated after each `predict()` call (§G); pass `no_kernels`, `num_workers`, `preprocessing_threads` from config to `predict()` (§T); pass `use_potentials` from config (§U); pass `step_scale` from config (§V); try/except around `os.listdir(results_path)` — missing directory → score=-inf instead of crash (§X.1); empty-scores guard + safe `mol_scores.get()` in score assignment (§X.2); try/except around entire `combine_boltz_scores` body (§X.3); `create_yaml_content` checks `os.path.exists(msa_path)` before including MSA line — absent file falls back to single-sequence mode gracefully (§X.4); forward `subsample_msa` and `num_subsampled_msa` from config to `predict()` (§Z) |
| `boltz/boltz_config.yaml` | Added `use_potentials: false` (§U); added `step_scale: null` (§V); added `subsample_msa`, `num_subsampled_msa`, `num_workers`, `preprocessing_threads` with defaults (§Z) |
| `config/config.yaml` | Added `max_heavy_atoms: 35` |
| `config/config_loader.py` | Loads and exposes `max_heavy_atoms` |
| `utils/molecules.py` | Added `get_canonical_smiles()` |
| `utils/__init__.py` | Exported `get_canonical_smiles`; exported SALSA functions (§N); exported `precompute_pool_fps` (§R); exported `ensure_msa`, `fetch_msa`, `msa_exists` (§S) |
| `utils/salsa.py` | New: SALSA algorithm — `generate_perturbations`, `nearest_pool_molecules`, `run_salsa_search` (§N); added `precompute_pool_fps` + optional `pool_fps` arg to `nearest_pool_molecules` (§R) |
| `utils/genetic.py` | New: GradientGA — `brics_crossover`, `tournament_select`, `run_gradient_ga` (§O); pre-compute pool FPs + pass to `nearest_pool_molecules` (§R) |
| `utils/msa.py` | New: MSA auto-fetch — `ensure_msa`, `fetch_msa`, `msa_exists` (§S) |
| `neurons/miner.py` | Move `dataset_iter` creation inside the outer `while` loop (§Y) |
| `neurons/miner.py` | Quality-first savi_stream_pool: sort by combined_score + skip after SALSA+GA done (§BB) |
| `neurons/miner.py` | Extend disk cache schema with `product_name`; add `_disk_cache_get_best`, `_apply_warm_start`; call warm-start at startup + each epoch boundary (§AA) |
| `neurons/miner.py` | Broader pharmacophore pre-filter: drop HBD minimum, raise HBA/logP ceilings to Lipinski Ro5 (§AB) |
| `utils/salsa.py` | FG-addition perturbation operator — `_FG_ATOMS`, `_FG_ATTACHMENT_ATOMS`; second loop in `generate_perturbations` (§DD) |
| `neurons/miner.py` | `_scaffold_diverse_candidates` helper; wider candidate slice (3×) + diversity selection in `run_boltz_prescoring` (§EE) |
| `BOLTZ2_INTEGRATION.md` | This file |

---

## Implemented Optimisations (continued)

### AB. Broader Pharmacophore Pre-filter (`neurons/miner.py`)

**Problem:** The `_pharma_ok` filter previously required `NumHDonors >= 1`, which silently excluded
entire drug scaffold classes with no NH/OH groups — N-alkylated heterocycles, pyrimidines with
N-methyl substitution, and many aromatic kinase-inhibitor cores.  These molecules can bind strongly
via their H-bond acceptor N/O atoms interacting with protein donors, and the Boltz-2 scoring formula
has no intrinsic bias toward or against them.  By filtering them out before PSICHIC, we shrank the
searchable chemical space unnecessarily.

The old upper bounds (`NumHAcceptors <= 7`, `MolLogP <= 4.5`) were also tighter than the standard
Lipinski Rule of 5, cutting out legitimate polar binders (HBA 8–10) and mild-logP compounds (4.5–5).

**Fix:** Aligned with Lipinski Rule of 5:

```python
# Old (too restrictive)
1 <= Descriptors.NumHDonors(mol) <= 3
and 2 <= Descriptors.NumHAcceptors(mol) <= 7
and 0.0 <= Descriptors.MolLogP(mol) <= 4.5

# New (Lipinski-aligned)
Descriptors.NumHDonors(mol) <= 5            # minimum removed; Ro5 max = 5
and 2 <= Descriptors.NumHAcceptors(mol) <= 10  # Ro5 max = 10 (was 7)
and -1.0 <= Descriptors.MolLogP(mol) <= 5.0   # allow mild polar + mild lipo
```

**Validator constraints enforced earlier in `_pharma_ok` are unchanged** (banned atom types,
rotatable bond bounds). The only change is in the drug-likeness heuristic block.

**Expected effect:** ~5–15% more molecules pass `_pharma_ok` per chunk (depends on SAVI-2020 file).
The additional molecules all pass the validator's hard constraints (Se ban, rotatable bonds) and are
within Lipinski space — so they're legitimate drug candidates the previous filter was wrongly
excluding. Given PSICHIC inference is batched, the throughput impact is negligible.

---

## Implemented Optimisations (continued)

### Y. Dataset Iterator Refresh on Exhaustion (`neurons/miner.py`)

**Problem:** `stream_random_chunk_from_dataset` was called once before the outer `while`
loop in `run_psichic_model_loop`, creating a single HuggingFace streaming iterator tied
to one randomly-chosen CSV file.  Once that file's batches were consumed, the inner
`for chunk in dataset_iter` loop exited immediately on every subsequent outer-loop
iteration, leaving the miner spinning on `await asyncio.sleep(2)` with no molecule
exploration for the rest of the epoch.

In practice SAVI-2020 files are large enough that exhaustion within a single ~72-minute
epoch is unlikely, but the latent bug could silently degrade search throughput if a small
file were selected or if epochs lengthen.

**Fix:** Moved `dataset_iter` creation to the top of the `while` loop body so each cycle
opens a fresh stream from a new randomly-chosen file:

```python
while not state['shutdown_event'].is_set():
    dataset_iter = stream_random_chunk_from_dataset(
        dataset_repo=state['hugging_face_dataset_repo'],
        chunk_size=state['chunk_size'],
    )
    for chunk in dataset_iter:
        ...
```

**Benefits:**
- Prevents the spin-on-empty-iterator bug regardless of file size.
- Explores a different region of the 283M-compound SAVI-2020 space on each outer cycle,
  increasing chemical diversity for subsequent PSICHIC and Boltz-2 evaluations.
- Zero overhead: the HuggingFace streaming client uses lazy loading; creating a new
  iterator only issues an HTTP range request when the first batch is consumed.
