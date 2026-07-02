# Molecular Optimisation Survey — SN68 Mining Context

Survey of algorithmic approaches for ligand discovery relevant to the NOVA subnet (Bittensor SN68).
The subnet scores with Boltz-2 using the ligand-efficiency formula:

```
boltz_score = (affinity_probability_binary − affinity_pred_value) / heavy_atom_count
```

All approaches below are evaluated against **this specific objective**: maximise that score for
the weekly target protein within one ~72-minute epoch, subject to the constraint that submitted
molecules must be resolvable by the validator (SAVI-2020 product name or `rxn:` format).

---

## 1. SALSA — Stochastic Approximate Ligand Scoring and Optimisation

### Concept

SALSA is an iterative hill-climbing strategy over chemical space. Starting from a seed molecule,
it generates a neighbourhood of structurally similar variants, scores them with a fast surrogate
(PSICHIC), and selects the best variant as the new seed for the next round. After N rounds, the
top candidates are validated with the expensive oracle (Boltz-2). Because PSICHIC is ~1000× faster
than Boltz-2, SALSA can explore thousands of molecules per epoch while spending GPU time only on
the most promising leads.

### Algorithm

```
seed ← best PSICHIC molecule from SAVI-2020 streaming
for round in 1..N:
    neighbourhood ← perturb(seed)          # atom substitution + FG addition/removal
    scored ← psichic_score(neighbourhood)  # ~ms per molecule
    boltz_safe ← filter(scored, is_boltz_safe_smiles)
    boltz_safe ← filter(boltz_safe, heavy_atoms ≤ 35)
    seed ← argmax(boltz_safe, combined_score)
boltz_top ← boltz_prescore(top_k(scored, k=5))
submit(boltz_top[0])                       # best Boltz-2 molecule first
```

### Perturbation operators

| Operator | Description | Example |
|----------|-------------|---------|
| Atom substitution | Replace C/N/O with bioisostere | C→N, OH→F, NH₂→CF₃ |
| Functional group addition | Append FG to an available position | +CH₃, +CN, +OCH₃ |
| Functional group removal | Strip a peripheral group | remove OH, remove halogen |
| Ring walk | Expand/contract ring size by ±1 | 5→6-membered ring |
| Scaffold hop | Replace core ring with bioisostere | benzene→pyridine |

RDKit implementation sketch:

```python
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

_BIOISOSTERES = {
    6:  [7, 8, 16],        # C → N, O, S
    7:  [6, 8],            # N → C, O
    8:  [6, 7, 16],        # O → C, N, S
    17: [9, 35],           # Cl → F, Br
    35: [17, 9],           # Br → Cl, F
}

def atom_substitution_variants(mol: Chem.Mol) -> list[Chem.Mol]:
    variants = []
    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        for target_an in _BIOISOSTERES.get(an, []):
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(target_an)
            try:
                Chem.SanitizeMol(rw)
                variants.append(rw.GetMol())
            except Exception:
                pass
    return variants
```

### Submission constraint

The validator resolves `product_name` values via the SAVI-2020 AWS API or the combinatorial DB.
SALSA-generated molecules cannot be submitted directly unless they happen to be SAVI-2020 products.

**Resolution strategy:** After SALSA identifies an optimal SMILES, perform a nearest-neighbour
lookup in the local SAVI-2020 dataset using Tanimoto similarity (Morgan fingerprints, radius=2).
Submit the nearest SAVI-2020 molecule. This preserves the chemistry insight while staying within
the submission envelope.

```python
from rdkit.Chem import DataStructs, AllChem

def nearest_savi_molecule(target_smiles: str, savi_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, radius=2, nBits=2048)
    savi_fps = savi_df['product_smiles'].apply(
        lambda s: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
    )
    sims = savi_fps.apply(lambda fp: DataStructs.TanimotoSimilarity(target_fp, fp))
    return savi_df.iloc[sims.nlargest(top_k).index]
```

### Expected benefit

3 SALSA rounds × 100 variants = 300 PSICHIC calls (~30 s) → 5 Boltz-2 validations (~4–12 min).
Compared to random SAVI-2020 streaming, SALSA should find higher-scoring molecules in the same
wall-clock budget because it concentrates search around promising chemical space.

### Estimated implementation effort

~200 lines of Python in a new `utils/salsa.py` module + integration in `neurons/miner.py`.

---

## 2. GradientGA — Gradient-Guided Genetic Algorithm

### Concept

A population-based approach that maintains a pool of ~50 molecules per epoch. It uses PSICHIC as
the cheap fitness function for selection, crossover, and mutation, and promotes only the top-N
survivors to Boltz-2 evaluation. The "gradient" refers to using PSICHIC score gradients (or rank
differences) to bias crossover towards chemically complementary parents.

### Algorithm

```
population ← top-50 PSICHIC molecules from SAVI-2020 streaming (first 30 min)
for generation in 1..G:
    parents ← tournament_select(population, k=2, n=25)   # 25 parent pairs
    offspring ← crossover(parents)                        # fragment exchange
    mutants ← mutate(offspring, rate=0.2)                 # atom sub / FG add-remove
    scored ← psichic_score(mutants + population)
    population ← elitism(top_49(scored)) + boltz_elite    # keep best Boltz mol
boltz_top ← boltz_prescore(top_5(population))
submit(boltz_top[0])
```

### Crossover

Fragment-based crossover at rotatable bonds:

```python
from rdkit.Chem import BRICS

def brics_crossover(mol_a: Chem.Mol, mol_b: Chem.Mol) -> list[Chem.Mol]:
    """Exchange BRICS fragments between two molecules."""
    frags_a = list(BRICS.BRICSDecompose(mol_a))
    frags_b = list(BRICS.BRICSDecompose(mol_b))
    offspring = []
    for fa in frags_a:
        for fb in frags_b:
            try:
                combined = BRICS.BRICSBuild([Chem.MolFromSmiles(fa), Chem.MolFromSmiles(fb)])
                for mol in combined:
                    if mol is not None:
                        offspring.append(mol)
            except Exception:
                pass
    return offspring[:10]  # cap to avoid combinatorial explosion
```

### Elitism and Boltz integration

Always preserve the single best Boltz-2-scored molecule in the population across generations.
This prevents the GA from discarding a molecule that looked mediocre under PSICHIC but scored
well under Boltz-2 (correlation between the two models is imperfect).

### Submission constraint

Same as SALSA: offspring molecules must map to a SAVI-2020 product name. Resolution strategies:
1. Restrict crossover to SAVI-2020 BRICS fragments only (stay inside the space by construction).
2. Post-hoc nearest-neighbour lookup as in SALSA.

Option 1 is cleaner: pre-index SAVI-2020 fragments and only recombine within that index.

### Population seeding

Seed the initial population from the first 30 minutes of SAVI-2020 streaming (while Boltz timing
window is not yet active). Continue streaming in parallel to discover new seeds.

### Expected benefit

GAs converge faster than random search when the fitness landscape has exploitable structure.
For protein–ligand binding, the landscape is rugged but has local basins. GA with PSICHIC
fitness should converge to a basin, and Boltz-2 validates whether the basin is real.

### Estimated implementation effort

~350 lines of Python in `utils/genetic.py` + integration in `neurons/miner.py`. More complex
than SALSA due to population management and fragment indexing.

---

## 3. Pharmacophore-Guided Pre-filtering

### Concept

Before PSICHIC scoring, apply a cheap pharmacophore filter to drop molecules that cannot
geometrically satisfy the target's binding site requirements. This is faster than PSICHIC and
reduces the search space for more expensive methods.

### Tools

- **RDKit pharmacophore**: `rdkit.Chem.Pharm2D` — 2D fingerprint-based pharmacophore matching
- **OpenBabel**: Fast 3D conformer generation + pharmacophore matching

### SN68 applicability

Without knowing the binding site geometry (only the protein sequence is given), full 3D
pharmacophore matching requires a docked structure. However, 2D pharmacophore features
(H-bond donor/acceptor counts, hydrophobic groups, charge) can be used as lightweight filters.

Example filter for typical kinase binding pockets:

```python
from rdkit.Chem import Descriptors

def pharmacophore_prefilter(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    logp = Descriptors.MolLogP(mol)
    # Lipinski-inspired filter tuned for the scoring formula (smaller = better)
    return (1 <= hbd <= 3) and (2 <= hba <= 7) and (0 <= logp <= 4)
```

### Estimated benefit

Reduces PSICHIC batch size by ~30% with minimal false-negative rate for drug-like targets.
Trivial to implement (10 lines).

---

## 4. Fragment-Based Lead Discovery (FBLD)

### Concept

Score individual molecular fragments (MW 150–300 Da, 8–15 heavy atoms) with Boltz-2 to find
high-efficiency binders, then elaborate them by fragment growing or linking.

### Why it suits the scoring formula

The scoring formula divides by `heavy_atom_count`. A fragment with 10 heavy atoms and
`affinity_probability_binary=0.6`, `affinity_pred_value=-5` scores:
`(0.6 + 5) / 10 = 0.56`

A drug-like molecule with 25 heavy atoms and better raw affinity:
`affinity_probability_binary=0.8`, `affinity_pred_value=-8`:
`(0.8 + 8) / 25 = 0.352`

The fragment wins. **If Boltz-2's affinity predictions are reliable at fragment size**, this is a
significant opportunity. Fragment efficiency is already implicit in the scoring formula.

### Caveat

Boltz-2 was trained primarily on drug-like molecules (200–500 Da). Fragment affinity predictions
may be less calibrated. Empirical validation on the weekly target is needed.

### Submission constraint

SAVI-2020 is a reaction product database; its smallest molecules are typically 200+ Da. Fragment
SMILES cannot be submitted directly. This approach works best as a Boltz oracle call to guide
SALSA/GA (use fragments as candidate perturbation targets, not as submissions).

---

## 5. Binding-Pocket-Aware Scoring

### Concept

When `config.yaml` exposes `binding_pocket` coordinates, Boltz-2's pocket guidance term steers
diffusion toward specific residues. A miner can exploit this by pre-filtering for molecules whose
predicted docked pose is likely within the pocket.

### Fast docking tools

| Tool | Speed | Notes |
|------|-------|-------|
| AutoDock Vina | ~5 s/mol (CPU) | Open-source, reliable |
| GNINA | ~10 s/mol (GPU) | CNN-scoring, better for flexible ligands |
| DiffDock | ~30 s/mol (GPU) | Diffusion-based, best pose quality |

### Integration sketch

When `binding_pocket` is non-null in `config`:
1. Generate a 3D conformer for each PSICHIC top-10 candidate (RDKit ETKDGv3).
2. Dock to the specified pocket coordinates with Vina (CPU, parallel).
3. Score the docked pose: only retain molecules where the docking box centroid is within
   `max_distance` Å of the pocket centre.
4. Run Boltz-2 only on the filtered subset.

This adds ~50 s of CPU work but eliminates Boltz-2 inference on molecules that would not satisfy
the pocket constraint anyway.

### Estimated implementation effort

~150 lines in `utils/docking.py` + Vina installation. Conditional on `binding_pocket` being set.

---

## 6. Adaptive Boltz Timing

### Current behaviour

Boltz-2 pre-scoring is triggered at `blocks_until_epoch ≤ 100` (~20 min). This is hardware-safe
but conservative: on A100 hardware, 5 molecules take ~4 min, leaving 16 min of GPU idle time.

### Improvement

Profile the actual Boltz inference time on first run, then adapt the trigger:

```python
# After first successful Boltz run, update the trigger threshold
actual_time_s = wrapper.last_inference_duration  # add this attribute to BoltzWrapper
blocks_per_second = 1 / 12  # 1 block every 12 seconds
safety_margin_blocks = 20   # always leave 20 blocks before epoch end
trigger_blocks = int(actual_time_s / 12 * max_candidates) + safety_margin_blocks
state['boltz_trigger_blocks'] = max(trigger_blocks, 30)  # never less than 30 blocks
```

This would allow an A100 to trigger Boltz much later (~40 blocks instead of 100), giving more
time for PSICHIC streaming to find a better seed molecule before Boltz validation begins.

### Estimated implementation effort

~30 lines in `neurons/miner.py` + a `last_inference_duration` field in `BoltzWrapper`.

---

## 7. Priority Queue with Continuous Boltz Re-scoring

### Current behaviour

`boltz_prescored` is a boolean flag. Once set to True, Boltz does not run again unless a new
PSICHIC best is found (which resets the flag). If a new best is found with 10 blocks remaining,
there is no time to run Boltz again.

### Improvement

Maintain a priority queue of molecules sorted by PSICHIC score. When the Boltz window opens,
continuously score the queue head:

1. Pop the top PSICHIC molecule.
2. Cache miss → run Boltz (async, non-blocking).
3. If the Boltz score beats the current best, update submission immediately.
4. Continue until the epoch ends or the queue is exhausted.

This turns Boltz pre-scoring from a one-shot batch into a continuous anytime algorithm.
Even a partial result (1 of 5 molecules scored) improves on the PSICHIC-only baseline.

### Estimated implementation effort

~80 lines replacing the `boltz_prescored` flag + `asyncio.Queue` for candidates.

---

## Implementation Roadmap (priority order)

All items 1–10 are **implemented**. Items 5 and 7 remain as conditional/research directions.

| Priority | Approach | Effort | Status | Expected gain |
|----------|----------|--------|--------|---------------|
| 1 | Adaptive Boltz timing (§G) | 30 lines | ✅ Done | More PSICHIC streaming time on fast hardware |
| 2 | Pharmacophore pre-filter (§F/§AB) | 10 lines | ✅ Done | ~30% faster PSICHIC batches |
| 3 | Continuous Boltz re-scoring (§H) | 80 lines | ✅ Done | Anytime improvement vs. one-shot |
| 4 | SALSA (§N + §DD/§GG/§HH/§II/§Q/§FF/§MM) | 200+ lines | ✅ Done | Directed search + hill-climbing |
| 5 | Binding-pocket docking filter (§D) | 150 lines | ⏳ Conditional | Only needed when `binding_pocket` set |
| 6 | GradientGA (§O) | 350 lines | ✅ Done | Population-level optimisation |
| 7 | FBLD (fragments) | Research | ⏳ Research | Needs empirical Boltz calibration study |
| 8 | §NN: Reduced-sample §MM/§FF screening | 60 lines | ✅ Done | ~2× more §MM rounds per epoch budget |
| 9 | §PP: Full-coverage SALSA perturbations (n_perturb 60→200) + SAVI pool 5k→10k | 4 lines | ✅ Done | Ring walk + terminal removal now contribute to every SALSA call |
| 10 | §QQQQ: RF surrogate above 100 training points | 30 lines | ✅ Done | 5–20% NDCG improvement on week-2+ runs |
| 11 | §WWWWW: Cross-target protein-similarity seeding | 80 lines | ✅ Done | Better SALSA seeds on week 1 of new family-member target |
| 12 | §XXXXX: H100 ultra-high VRAM tier (num_subsampled_msa=4096, sampling_steps_affinity=200) | 15 lines | ✅ Done | Better affinity predictions on H100 80 GB hardware |
| 13 | §ZZZZZ: HA-adaptive SALSA operator budget allocation | 35 lines | ✅ Done | 5–15% more SALSA hits scoring above seed when seed >25 HA |
| 14 | §AAAAAA: Dual surrogate UCB — per-component tree-variance exploration on APB+APV RF models | 70 lines | ✅ Done | 5–10% more novel Boltz-confirmed binders/epoch at ≥100 cache points |
| 15 | §BBBBB: Persist adaptive timing (`boltz_time_per_mol`, `boltz_trigger_blocks`) across restarts | 35 lines | ✅ Done | 12–15 min extra PSICHIC streaming recovered per restart on A100/H100 |
| 16 | §CCCCCC: Persist §YY winning reaction class (`best_boltz_rxn_class`) across restarts | 40 lines | ✅ Done | 2× SAVI streaming bias toward best reaction template active immediately after restart |
| 17 | §DDDDDD: Cache `ligand_iptm` + confidence-weighted surrogate training | 30 lines | ✅ Done | 3–8% NDCG improvement via down-weighting uncertain-pose training examples |
| 18 | §EEEEEE: Top-K reaction class score weighting (4×/2×/1.5×/1× vs binary 2×/1×) | 60 lines | ✅ Done | Multi-modal SAVI sampling; fluke-resistant via running mean; active from epoch 2+ |
| 19 | §HHHHHH: Surrogate-blended SALSA pool score for §FF/§MM hill-climbing | 90 lines | ✅ Done | SALSA converges 1-2 rounds faster to Boltz-optimal region from epoch 3+ (RF tier) |
| 20 | §IIIIII: Online surrogate refresh after each §MM full-score | 30 lines | ✅ Done | Later §MM rounds use freshest surrogate; epoch winners tighten signal around hill-climbing region (RF tier, epoch 3+) |

All approaches share the same submission constraint: molecules must map to valid SAVI-2020
product names. SALSA and GradientGA both solve this via nearest-neighbour SAVI-2020 lookup.

---

## References

- Boltz-2: `jwohlwend/boltz` (MIT licence) — affinity prediction via diffusion + MSA
- PSICHIC: `Metanova/PSICHIC` — fast protein–ligand binding prediction (pre-filter)
- SAVI-2020: Synthetically Accessible Virtual Inventory, ~283M compounds
- RDKit: Open-source cheminformatics (BRICS, pharmacophore, Morgan FP)
- AutoDock Vina: Trott & Olson, J. Comput. Chem. 2010
- BRICS fragmentation: Degen et al., ChemMedChem 2008
