# Boltz-2 Miner Integration

## Context

The NOVA subnet (SN68) incentive mechanism splits rewards as follows:

| Model | Weight | Miner implementation (stock) |
|-------|--------|-------------------------------|
| Boltz-2 | **100%** (`boltz_weight: 1.0`) | Previously 0% — now partially implemented |
| PSICHIC | 0% (weight unused) | 100% |

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
heavy atoms gives `(0.9 − (−9.0)) / 25 = 0.396`.  This is a **ligand-efficiency metric**:
smaller, more potent binders score higher per atom.

**Optimal molecule properties:**
- High binary binding probability (≥ 0.7)
- Strong predicted binding energy (≤ −7 kcal/mol)
- Moderate heavy atom count (15–30 atoms) — large molecules are penalised by the denominator
- Drug-like: MW 200–450 Da, logP 1–4, ≤ 5 H-bond donors, ≤ 10 H-bond acceptors

The validator picks the **first** molecule in a miner's submission
(`sample_selection: "first"`, `num_molecules_boltz: 1`), so molecule ordering matters.

---

## What Was Implemented (this PR)

### 1. Boltz-safe filter in the PSICHIC loop (`neurons/miner.py`)

```python
# Filter for Boltz-2 compatibility (validator scores with Boltz-2)
df = df[df['product_smiles'].apply(lambda x: is_boltz_safe_smiles(x)[0])]
```

Molecules that fail `is_boltz_safe_smiles` (e.g., atom names > 4 characters) would receive
`-inf` from the validator. Filtering them out early ensures we never waste a submission on
an unscorable molecule.

### 2. `run_boltz_prescoring` function (`neurons/miner.py`)

When the miner is ≤ 50 blocks from epoch end and has found a candidate, it:

1. Takes the top-5 PSICHIC-ranked molecules
2. Runs `BoltzWrapper.score_molecules_target()` on them (same code the validator uses)
3. Sorts by `boltz_score` descending
4. Puts the highest-scoring Boltz-2 molecule **first** in `candidate_product`

The blocking Boltz inference runs via `asyncio.to_thread()` so the event loop stays live.

### 3. `candidate_molecules` state tracking

The full PSICHIC top-10 DataFrame is kept in `state['candidate_molecules']` so
`run_boltz_prescoring` can access SMILES without re-querying the dataset.

### 4. `boltz_prescored` flag

Set to `True` once Boltz pre-scoring completes for a given candidate.  Reset to `False`
whenever a new best PSICHIC candidate is found or the epoch rolls over.  This prevents
repeated expensive Boltz calls within the same epoch.

---

## Architecture Diagram

```
Epoch start
    │
    ▼
stream SAVI-2020 chunks (128 mols each)
    │
    ├─ filter: min_heavy_atoms ≥ 10
    ├─ filter: is_boltz_safe_smiles()          ← NEW
    │
    ▼
PSICHIC scoring (target - antitarget_weight × antitarget)
    │
    ▼
top-10 molecules → candidate_molecules         ← NEW (stored)
    │
    ▼  (when blocks_until_epoch ≤ 50)
Boltz-2 pre-scoring on top-5 candidates        ← NEW
    │  (asyncio.to_thread → BoltzWrapper)
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
before the epoch begins.  Without it, Boltz-2 will run without evolutionary context and
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

Scoring 5 candidates ≈ 4–12 minutes on typical mining hardware.  Epochs are ~72 minutes
(360 blocks × 12 s), so this fits comfortably within the 50-block window (~10 minutes).

**Tuning options** (edit `boltz/boltz_config.yaml`):

```yaml
# Faster but lower-quality — suitable for early-epoch filtering
sampling_steps: 50           # default: 100
diffusion_samples_affinity: 1  # default: 3
```

Reduce `max_candidates` in `run_boltz_prescoring(state, max_candidates=3)` if GPU memory
is tight.

---

## Future Optimisation Opportunities

### A. SALSA (Stochastic Approximate Ligand Scoring and Optimisation)

SALSA is an iterative perturbation approach: start from a good seed molecule, apply small
SMILES perturbations (atom substitutions, ring variations, functional-group swaps), score
each variant with a fast surrogate, keep winners.

Relevance for this miner:
- Run a SALSA loop using the Boltz-2 affinity score as the objective
- Initialise from the top PSICHIC molecule each epoch
- 5–10 perturbation rounds × 20 variants = 100–200 fast PSICHIC evaluations + final Boltz
  validation of the top-5

This could generate novel molecules that score better than anything in SAVI-2020 for the
specific weekly target.

Implementation sketch:
```python
from rdkit import Chem
from rdkit.Chem import AllChem

def perturb_smiles(smiles: str) -> list[str]:
    """Atom-substitution and ring-opening perturbations."""
    ...

async def run_salsa_loop(seed_smiles: str, state, rounds=5, variants=20) -> str:
    best = seed_smiles
    for _ in range(rounds):
        candidates = [perturb_smiles(best) for _ in range(variants)]
        # quick PSICHIC filter
        # Boltz-2 validation of top-3
        best = top_boltz_candidate(candidates)
    return best
```

### B. GradientGA (Gradient-Guided Genetic Algorithm)

Maintain a population of ~50 molecules per epoch. Use PSICHIC as the cheap fitness
function for selection/crossover/mutation, and promote only the top-N to Boltz-2 evaluation.

Key advantage: explores chemical space much more broadly than streaming random chunks from
SAVI-2020.

Population operations:
- **Crossover**: SMILES substring exchange (fragmentation at rotatable bonds)
- **Mutation**: atom substitution, functional-group addition/removal
- **Selection**: tournament selection on PSICHIC score
- **Elitism**: always keep the top-1 Boltz-2 scored molecule

### C. Multi-molecule entropy bonus

When the validator increases `num_molecules_boltz > 1`, the entropy bonus activates
(`ranking.py` lines 109–118).  The miner should respond by submitting a set of molecules
with high MACCS fingerprint diversity, not just the single best binder.

Implementation: after Boltz pre-scoring, fill remaining submission slots greedily with
molecules that maximise `compute_maccs_entropy()` relative to the already-selected set.

### D. Binding-pocket guidance

`config.yaml` exposes `binding_pocket`, `max_distance`, and `force`.  When the validator
sets a pocket constraint, Boltz-2 adds a soft or hard guidance term to steer the diffusion
towards specific residues.  Miners could pre-filter for molecules whose docked pose (from
a fast docking tool like Vina or Gnina) is predicted to sit inside the specified pocket,
avoiding molecules that Boltz-2 will penalise for wrong pose.

### E. Caching Boltz scores across epochs

If the weekly target doesn't change between epochs, previously scored molecules can be
cached (keyed by `SMILES + protein_sequence`).  This allows the miner to accumulate a
lookup table of Boltz scores and skip re-scoring known molecules.

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
| `neurons/miner.py` | Added `is_boltz_safe_smiles` filter, `run_boltz_prescoring()`, Boltz trigger logic, state fields |
| `BOLTZ2_INTEGRATION.md` | This file |
