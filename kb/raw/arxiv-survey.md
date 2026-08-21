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

Items 1–59 are **implemented** (§VVVVVVVVVV added 2026-08-20). Item 5 remains as a conditional direction (binding-pocket docking filter, only relevant when `binding_pocket` is set in config). §UUUUUUUUUU (interface PDE surrogate weight) added 2026-08-19. §VVVVVVVVVV (overall interface iPTM surrogate weight) added 2026-08-20.

| Priority | Approach | Effort | Status | Expected gain |
|----------|----------|--------|--------|---------------|
| 1 | Adaptive Boltz timing (§G) | 30 lines | ✅ Done | More PSICHIC streaming time on fast hardware |
| 2 | Pharmacophore pre-filter (§F/§AB) | 10 lines | ✅ Done | ~30% faster PSICHIC batches |
| 3 | Continuous Boltz re-scoring (§H) | 80 lines | ✅ Done | Anytime improvement vs. one-shot |
| 4 | SALSA (§N + §DD/§GG/§HH/§II/§Q/§FF/§MM) | 200+ lines | ✅ Done | Directed search + hill-climbing |
| 5 | Binding-pocket docking filter (§D) | 150 lines | ⏳ Conditional | Only needed when `binding_pocket` set |
| 6 | GradientGA (§O) | 350 lines | ✅ Done | Population-level optimisation |
| 7 | FBLD (fragments, §OOOOOOOOOOO) | ~80 lines | ✅ Done (2026-08-13) | Fragment cold-start probe; empirical calibration now live |
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
| 21 | §JJJJJJ: Reduced MSA subsampling depth in fast-screen mode (full_msa//4, floor 256) | 12 lines | ✅ Done | ~8–12 s saved per fast-screen molecule on A100; ~0.5–0.8 extra §MM rounds per epoch |
| 22 | §LLLLLL: Parallel affinity diffusion samples on H100 (max_parallel_samples=3) | 8 lines | ✅ Done | ~2–3× throughput for diffusion_samples_affinity=3 on H100 ≥70 GiB; also fixed bug where `max_parallel_samples` was hardcoded to 1 in `boltz/src/boltz/main.py` regardless of config |
| 23 | §MMMMMM: Cross-call SALSA pool FP cache — eliminate redundant `precompute_pool_fps` across §MM rounds | 15 lines | ✅ Done | ~57 s saved on H100 (20 rounds, Ridge tier); ~27 s on A100 (10 rounds); ~2–3 extra §MM rounds on H100 at no GPU cost |
| 24 | §NNNNNN: §NN two-phase screening + FP cache for §XX tautomer search | 60 lines | ✅ Done | §XX reduced from up to 6 full Boltz calls to 1 fast-batch + 1 full call; ~5× GPU time reduction per epoch |
| 25 | §OOOOOO: Cache-evidence adaptive §TTTT fragment quota (500/1000/2500 based on per-bucket Boltz LE) | 45 lines | ✅ Done | Protein-adaptive fragment slot sizing; more small-molecule diversity when ≤18-HA molecules historically outperform |
| 26 | §XX: Tautomer enumeration after §MM convergence | 80 lines | ✅ Done | Explores H-bond donor/acceptor neighbourhood of epoch best without PSICHIC cost |
| 27 | §WW: Multi-seed Boltz stability check for position-0 ordering | 50 lines | ✅ Done | Reduces risk from stochastic seed-68 outlier at validation; uses mean(seeds 42,68,123) |
| 28 | §PPPPPP: Remote Boltz cache persistence via GitHub JSON export | 150 lines | ✅ Done | Warm surrogate + timing + rxn bias from epoch 1 on any container restart; 15–25% gain on restart sessions |
| 29 | §RRRRRR: Cross-target history in GitHub export — §WWWWW seeds on protein rotation + fresh container | 60 lines | ✅ Done | +3–8% Boltz score on epoch 1 after protein rotation when homologous prior target exists |
| 30 | §SSSSSS: Diversity-aware §UU cache seed selection — max-min Tanimoto from top-20 instead of top-3 by score | 50 lines | ✅ Done | +3–8% probability of finding new scaffold-family winner per epoch on week-3+ converged runs |
| 31 | §TTTTTT: Extended §XX tautomer search for 2nd/3rd epoch-best molecules | 200 lines | ✅ Done | +2–5% chance of new epoch best from rank-2/3 scaffold tautomers; free when all hits cached |
| 32 | §UUUUUU: Surrogate-guided GradientGA fitness — replace PSICHIC score with dual RF surrogate blend in GA tournament selection | 30 lines | ✅ Done | +3–6% Boltz score on GA-active epochs (epoch 3+, ≥100 cache points) by evolving toward Boltz objective instead of PSICHIC |
| 33 | §VVVVVV: Submission-Archive InChIKey pre-filter — drop candidates already submitted by any miner before Boltz-2 scoring | 20 lines | ✅ Done | Prevents wasted GPU time on non-unique submissions; ensures submitted molecule is always validator-accepted |
| 34 | §WWWWWWW: Boltz-2 ensemble variance cache storage — store std of 3-sample LE in boltz_le_std; use as combined down-weight with ligand_iptm in RF surrogate training | 70 lines | ✅ Done | +3–5% NDCG improvement in surrogate quality from epoch 3+; stacks with §DDDDDD |
| 35 | §XXXXXXXX: numpy bugfix (§WWWWWWW was silently inactive) + §WW inter-seed std cache — store std of [s_68, s_42, s_123] as boltz_ww_std; use as additional surrogate confidence weight | 55 lines | ✅ Done | Bugfix restores full §WWWWWWW benefit; §XXXXXXXX adds +1–2% NDCG on top via cross-seed variance weighting |
| 36 | §YYYYYY: Startup surrogate initialisation from GitHub cache — fit dual RF at startup (≥40 pts) and apply 0.4×PSICHIC + 0.6×surrogate blend inline in PSICHIC streaming loop so global_candidate_pool is Boltz-calibrated from chunk 1 | ~50 lines | ✅ Done | +3–8% on warm-cache restart sessions; SALSA seeds improve when streaming time is short |
| 37 | §ZZZZZZZZ: Cache-aware adaptive §WW seed budget — skip extra seeds (42/123) when cached boltz_ww_std < 0.003; save 1–2 Boltz call budgets for §MM/§TTTTTT on stable cached molecules | ~25 lines | ✅ Done | +2–4% on epoch 3+ when top-2 candidates already cached with low inter-seed variance |
| 38 | §AAAAAAAAA: SALSA convergence-based early stopping — detect when the best seed SMILES is unchanged between rounds and break early; frees CPU budget for §MM's next molecule or another SALSA instance | ~12 lines | ✅ Done | Saves 1–N redundant rounds on converged §MM hill-climbing calls (up to 20 rounds/epoch on H100); zero regression |
| 39 | §BBBBBBBBBB: Warm-start molecule inclusion in §MM seed pool — inject top SQLite-cached entry into `_mm_all_scored` when it was evicted from the initial Boltz pass; lets §MM explore prior-epoch best neighbourhood immediately | ~12 lines | ✅ Done | +3–8% expected Boltz score on epoch 2+ when warm-start molecule was filtered out of scaffold-diversity selection |
| 40 | §CCCCCCCCCC: Surrogate-pool basin-hop fallback for §MM seed exhaustion — when all Boltz-scored seeds are exhausted (`_mm_next_seed is None`), scan top-50 surrogate-pool entries for an untried safe SMILES and continue §MM from that basin | ~30 lines | ✅ Done | +3–8% on epoch 1 (low cache, small Boltz-seed pool) by enabling 1–4 additional §MM rounds from surrogate-nominated chemical basins; zero regression on warm-cache epochs |
| 41 | §DDDDDDDDDD: MSA GitHub cache — after ColabFold fetches an MSA, gzip-compress and upload to `msa_cache/{protein}.a3m.gz` in the GitHub repo; on container restart try GitHub download before ColabFold (saves 5–15 min per restart for known proteins) | ~90 lines | ✅ Done | +2–5% Boltz LE on restart sessions by enabling full-MSA Boltz from epoch 1 instead of single-sequence mode; ColabFold wait eliminated for all proteins seen in prior week |
| 42 | §EEEEEEEEEE: Gzip-compress Boltz cache export before base64 — apply same gzip technique as §DDDDDDDDDD MSA cache to §PPPPPP JSON export; expand entries from 500→1000 within GitHub 1 MB limit; backward-compatible downloader via magic-byte detection | ~20 lines | ✅ Done | 2× more surrogate training data from GitHub on restart (+3–8% NDCG on epoch 3+ with 1000 vs 500 cache points); also prevents export failures when cache grows large |
| 43 | §FFFFFFFFFF: Persist Boltz-2 `confidence_score` (overall complex structural confidence, 0–1) to SQLite cache and GitHub export; extend dual RF surrogate weight to `lig_iptm * conf_score / ((1+10*le_std)*(1+10*ww_std))` so structurally uncertain molecules are down-weighted in surrogate training | ~50 lines | ✅ Done | ~1–3% NDCG improvement on structurally mixed cache; reduces false-positive UCB candidates when training set contains noisier low-confidence Boltz runs |
| 44 | §GGGGGGGGGG: Fast-mode structure recycling 3→1 + disable FK potentials — two Boltz-2 call parameters never previously adapted for fast mode; saves 30–40% fast-call wall time on A100/H100 | ~15 lines | ✅ Done | +1–4 §MM rounds/epoch on A100/H100; equivalent to 2–6% higher expected Boltz LE per epoch |
| 45 | §HHHHHHHHHH: Boltz-2 embedding surrogate — use `write_embeddings=True` to extract 384D ligand embeddings from Boltz-2, PCA-reduce to 32D, concatenate with Morgan FP+physchem descriptor for protein-conditioned surrogate features | ~200 lines | ✅ Done | +3–8% surrogate NDCG on epoch 3+ (estimated); especially impactful on week-1 new targets with few cache points |
| 46 | §IIIIIIIIII: PSICHIC LE as surrogate training feature — store PSICHIC `combined_score` at Boltz call time; use as feature 85 (or 118 with embeddings) in surrogate training to learn PSICHIC→Boltz correction | ~75 lines | ✅ Done | +3–8% NDCG; especially impacts week-1 runs where embeddings are sparse but PSICHIC scores are universal |
| 47 | §JJJJJJJJJJ: Mid-epoch exploratory Boltz probe — fire one ultra-fast Boltz call (≈15–50 s, fast=True) when pool ≥1000 molecules and cache is empty; seeds surrogate 20–40 min earlier on cold-start epoch 1 | ~65 lines | ✅ Done | +3–6% Boltz LE on epoch 1 of new weekly target with no GitHub cache |
| 48 | §KKKKKKKKKK: Boltz-2 embedding centroid diversity bonus — add cosine-distance bonus (γ=0.05) from centroid of scored molecules' PCA embeddings to UCB score; diversifies candidate selection in binding-pose space beyond Morgan FP Tanimoto | ~30 lines | ✅ Done | +2–5% novel scaffold winner probability per epoch on week-4+ converged runs |
| 49 | §LLLLLLLLLL: Multi-molecule cold-start probe (3-scaffold batch) — rewrite §JJJJJJJJJJ to score 3 scaffold-diverse molecules instead of 1; Ridge surrogate requires ≥3 training points to fit at all | ~40 lines | ✅ Done (2026-08-11) | 3× surrogate training data from probe; Ridge tier activates on next chunk vs. degenerate with 1 pt |
| 50 | §MMMMMMMMMM: `complex_iplddt` surrogate confidence weight — persist Boltz-2 interface-weighted pLDDT to SQLite and add `max(0.1, iplddt)` down-weight to surrogate training alongside ligand_iptm and confidence_score | ~80 lines | ✅ Done (2026-08-11) | Cleaner surrogate signal; poses with disordered binding interface ≤5× lower training weight |
| 51 | §NNNNNNNNNNN: Cross-target seeds in cold-start probe — extend §LLLLLLLLLL probe with up to 2 homolog-protein validated molecules; scoring them on the new target creates known-binder calibration points for the surrogate | ~20 lines | ✅ Done (2026-08-12) | +2–4% surrogate NDCG on epoch 1 of family-member protein rotations; anchors high end of score distribution from first probe |
| 52 | §OOOOOOOOOOO: FBLD fragment probe in cold-start — after main §LLLLLLLLLL probe, score top-3 PSICHIC-scoring molecules with 10–15 HA from savi_stream_pool using a second fast Boltz pass; provides fragment-scale LE calibration data so §OOOOOO correctly sizes the fragment slot quota from epoch 1 | ~80 lines | ✅ Done (2026-08-13) | +1–4% surrogate NDCG from correctly-calibrated fragment quota; +2–6% Boltz LE on epoch 2+ when fragments prove to be the dominant chemical class for the weekly target |
| 53 | §PPPPPPPPPP: Boltz-2 embedding export/import in GitHub cache — base64-encode top-20 384D embeddings in the gzip-compressed export JSON; import on container restart via UPDATE boltz_cache SET boltz_embedding=? WHERE boltz_embedding IS NULL; §HHHHHHHHHH warm-starts from epoch 1 | ~50 lines | ✅ Done (2026-08-14) | Embedding surrogate active immediately on container restart; +3–8% NDCG on restart sessions vs. cold-embedding start |
| 54 | §QQQQQQQQQQ: TF32 Tensor Core matmul enable — `torch.set_float32_matmul_precision('high')` replaces `'highest'` in `boltz/src/boltz/main.py`; routes float32 matmul through Tensor Cores on Ampere+ GPUs | 1 line | ✅ Done (2026-08-15) | 20–50% overall inference speedup on A100/H100/RTX 3090+; 1–4 extra §MM rounds per epoch |
| 55 | §RRRRRRRRRR: BF16 mixed precision on A100/H100 — `precision="bf16-mixed"` in Lightning Trainer (hardware-gated ≥38 GiB); additional ~1.5× forward-pass speedup via AMP | ~20 lines | ✅ Done (2026-08-16) | Additional 1.3–1.8× speedup on top of §QQQQQQQQQQ; combined 2–3× total vs baseline |
| 56 | §SSSSSSSSSS: `torch.compile()` graph fusion — JIT-compile model after checkpoint load for 10–20% more speedup via kernel fusion and Python dispatch elimination | ~15 lines | ✅ Done (2026-08-16) | Best when scoring multiple molecules per predict() call; stacks with §QQQQQQQQQQ + §RRRRRRRRRR |
| 57 | §TTTTTTTTTT: Streaming saturation redirect — when surrogate (≥40 pts) detects no improvement in top predicted score for 3 consecutive chunks, yield streaming thread to §MM and reduce batch rate to 1 chunk/5 min | ~40 lines | ✅ Done (2026-08-18) | +1–3 §MM rounds on warm-cache epochs where the PSICHIC pool has converged; zero regression on epoch 1 |
| 58 | §UUUUUUUUUU: `complex_ipde` surrogate weight — store Boltz-2 interface predicted distance error (Å) in SQLite cache; add `/ (1 + 0.3 × ipde)` down-weight to all 3 surrogate training functions so uncertain-geometry runs contribute less to RF/Ridge training | ~50 lines | ✅ Done (2026-08-19) | +1–3% surrogate NDCG by filtering noise from geometrically uncertain Boltz runs; stacks with §MMMMMMMMMM (iplddt) and §FFFFFFFFFF (confidence_score) |
| 59 | §VVVVVVVVVV: `iptm` surrogate weight — store Boltz-2 overall interface iPTM (cross-chain A-B confidence for protein+ligand) in SQLite cache; add `* max(0.1, iptm)` to all 3 surrogate training weight formulas; filters false-positive training examples where ligand structure is confident but binding orientation is uncertain | ~40 lines | ✅ Done (2026-08-20) | +1–2% surrogate NDCG by filtering ligand_iptm-high but iptm-low outliers; orthogonal to all existing confidence signals |
| 60 | §WWWWWWWWWW: Auto-pocket derivation from Boltz-2 predicted pose — when `binding_pocket` is null in config, parse the mmCIF/PDB structure written by Boltz-2 for the epoch's best molecule, identify protein residues within 4.5 Å of any ligand heavy atom, and use those residue numbers as a soft pocket constraint (`force: false`) for all subsequent Boltz calls in that epoch; no external tools required (numpy coordinate arithmetic on Boltz output) | ~100 lines | ⏳ Proposed (2026-08-21) | +3–8% affinity accuracy on subsequent §MM rounds by focusing Boltz diffusion on the confirmed binding site; self-consistent with Boltz-2's own prediction — no assumption about binding site geometry |
| 61 | §XXXXXXXXXX: RF prediction-interval §MM pre-screen gate — before each §MM fast-mode Boltz call (§NN), compute the RF surrogate's per-tree 95th-percentile score for the candidate; if upper bound < `current_best_score × 0.85`, skip fast-screen and advance to next §MM candidate immediately; saves 15–50 s GPU time per skipped molecule | ~25 lines | ⏳ Proposed (2026-08-21) | +1–3 §MM rounds per epoch on well-converged runs (epoch 4+) where the surrogate reliably identifies dead-end chemical regions; zero regression when surrogate is Ridge-only (RF tier not yet reached) |
| 62 | §YYYYYYYYYY: Dynamic PSICHIC/surrogate blend ratio in §YYYYYY streaming — replace the fixed 0.4/0.6 PSICHIC/surrogate split with a cache-size-adaptive ratio: `alpha = min(0.90, n_cache_pts / 200.0)`, so blend starts PSICHIC-dominant at 20 pts (alpha=0.10 surrogate) and saturates surrogate-dominant at 200+ pts (alpha=0.90); prevents premature surrogate over-trust in early cold-start epochs | ~15 lines | ⏳ Proposed (2026-08-21) | +1–3% global_candidate_pool quality on epoch 2 runs where the fixed 0.6 surrogate weight over-trusts a Ridge model with only 20–40 training points; no change on epoch 4+ (≥200 pts, already in RF regime) |

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
