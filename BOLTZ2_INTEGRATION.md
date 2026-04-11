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

## Future Optimisation Opportunities

### A. SALSA (Stochastic Approximate Ligand Scoring and Optimisation)

Iterative perturbation from a seed molecule, using PSICHIC as a cheap filter and Boltz-2
as the validation oracle. Unlike dataset streaming, SALSA can find molecules not in
SAVI-2020 for the specific weekly target.

**Constraints:** The miner submits `product_name` values that must be resolvable to SMILES
by the validator (via the AWS API or `rxn:` format). Raw novel SMILES cannot be submitted
directly. SALSA therefore works as a **pre-filter** to identify promising chemical space,
with the final submitted molecule still being a SAVI-2020 or rxn: product.

Implementation sketch:

```python
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# Substituents for atom-level perturbation
_SUBSTITUENTS = ['F', 'Cl', 'Br', 'OH', 'NH2', 'CH3', 'CF3', 'OCH3', 'CN']

def perturb_molecule(mol: Chem.Mol) -> list[Chem.Mol]:
    """Generate atom-substitution and functional-group perturbations."""
    variants = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue  # skip explicit H
        for sub in _SUBSTITUENTS:
            edit = Chem.RWMol(mol)
            try:
                edit.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(
                    Chem.GetPeriodicTable().GetAtomicNumber(sub.rstrip('H'))
                )
                candidate = edit.GetMol()
                Chem.SanitizeMol(candidate)
                variants.append(candidate)
            except Exception:
                pass
    return variants

async def run_salsa_loop(seed_smiles: str, state, rounds: int = 3) -> str:
    """
    SALSA: 3 rounds of perturbation → PSICHIC filter → Boltz validation.
    Returns the SMILES of the best molecule found (for lookup in SAVI-2020 or rxn: DB).
    """
    best_smiles = seed_smiles
    for _ in range(rounds):
        seed_mol = Chem.MolFromSmiles(best_smiles)
        if seed_mol is None:
            break
        variants = perturb_molecule(seed_mol)
        variant_smiles = [Chem.MolToSmiles(v) for v in variants]
        # PSICHIC filter (fast) — score with existing models
        # ... score variant_smiles with state['psichic_models'] ...
        # Boltz validation of top-3 (uses cache)
        # ... call run_boltz_prescoring on top-3 variants ...
        # Update best_smiles from Boltz winner
    return best_smiles
```

**Roadblock to resolve:** map the SALSA winner back to a SAVI-2020 `product_name` or
generate a valid `rxn:` product string so it can be submitted.

### B. GradientGA (Gradient-Guided Genetic Algorithm)

Maintain a population of ~50 molecules per epoch. Use PSICHIC as the cheap fitness
function for selection/crossover/mutation, and promote only the top-N to Boltz-2 evaluation.

Population operations:
- **Crossover**: SMILES substring exchange (fragmentation at rotatable bonds)
- **Mutation**: atom substitution, functional-group addition/removal
- **Selection**: tournament selection on PSICHIC combined score
- **Elitism**: always keep the top-1 Boltz-2 scored molecule

Same submission constraint as SALSA: winners must map to a submittable product name.

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
| `neurons/miner.py` | Added `is_boltz_safe_smiles` + `max_heavy_atoms` filters; `run_boltz_prescoring()` with two-tier cache; Boltz trigger at 100 blocks (was 50); `boltz_score_cache` + `boltz_cache_db` state fields; `_init_boltz_cache_db`, `_disk_cache_get`, `_disk_cache_put` helpers; fixed `entropy_weight` → `entropy_start_weight` AttributeError; pharmacophore pre-filter (§F); adaptive trigger using `boltz_trigger_blocks` state field (§G); anytime incremental scoring — one molecule at a time with immediate reorder (§H) |
| `boltz/wrapper.py` | Added `last_inference_duration` field populated after each `predict()` call (§G) |
| `config/config.yaml` | Added `max_heavy_atoms: 35` |
| `config/config_loader.py` | Loads and exposes `max_heavy_atoms` |
| `utils/molecules.py` | Added `get_canonical_smiles()` |
| `utils/__init__.py` | Exported `get_canonical_smiles` |
| `BOLTZ2_INTEGRATION.md` | This file |
