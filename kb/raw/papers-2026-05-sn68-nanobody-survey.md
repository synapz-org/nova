# SN68 Nanobody Track — Arxiv Survey (May 2026)

Curated for the BoltzGen 10-metric rank-sum objective on Q9NZQ7 (5-HT2C).
Scope spans five subgoals: (1) sequence-only surrogates for AF/BoltzGen-style
interface metrics, (2) nanobody/VHH generation and re-ranking, (3) active
learning / BO under expensive oracles, (4) multi-objective / rank-sum
optimization, (5) anti-saturation and diversity.

---

## Rēs ipSAE loquuntur: What's wrong with AlphaFold's ipTM score and how to fix it (bioRxiv 2025.02.10.637595, 2025)

**Summary:** Dunbrack shows the standard ipTM is depressed by disordered or
non-interacting domains because it normalizes by full chain length. ipSAE
recomputes the score over only residue pairs that pass a PAE+distance cutoff
and adjusts d0 accordingly, separating true vs false complexes much better
than ipTM. Works from existing AF2/AF3 JSON outputs — no retraining.

**Steal:** Add ipSAE as a cheap post-hoc re-ranker over our BoltzGen outputs;
it almost certainly correlates with `interaction_pae` and
`design_to_target_iptm` (two of our top-4 metrics) better than raw ipTM does.

---

## BindEnergyCraft: Casting Protein Structure Predictors as Energy-Based Models for Binder Design (arxiv:2505.21241, 2025)

**Summary:** Reinterprets AF/Boltz confidence outputs as an energy-based model
via Joint Energy-based Modeling, yielding pTMEnergy — a dense, well-behaved
signal derived from predicted inter-residue error distributions. Outperforms
BindCraft / RFdiffusion / ESM3 on Rosetta filters across 7/8 targets, with up
to 16-point gains in success rate. Directly addresses the sparse-gradient
problem of optimizing raw ipTM.

**Steal:** Use pTMEnergy as the surrogate target instead of raw `design_iiptm`
— it captures the same physics but trains better and correlates with multiple
rank-sum metrics simultaneously.

---

## pLDDT-Predictor: High-Speed Protein Screening Using Transformer and ESM2 (arxiv:2410.21283, 2024; withdrawn 2025-06 but architecture still useful)

**Summary:** ESM2 embeddings → transformer encoder → per-residue pLDDT
prediction, Pearson 0.79 against AF2 ground truth at ~250,000× speedup
(7 ms/protein). Demonstrates that a small head on frozen ESM2 features is
enough to recover a key structural confidence metric.

**Steal:** Reuse the architectural recipe (frozen ESM2 + shallow transformer
head) for our own surrogate, but train multi-head against the 10 BoltzGen
metrics with our 8k labels — don't waste capacity on iiptm-only.

---

## Twin Peaks: Dual-Head Architecture for Structure-Free Prediction of Protein-Protein Binding Affinity and Mutation Effects (arxiv:2509.22950, 2025)

**Summary:** ESM3 embeddings of paired sequences fused via cross-attention,
with two heads predicting ΔG and ΔΔG simultaneously from sequence alone — no
structure required. The shared backbone exploits the fact that absolute
affinity and mutation effects share representation structure.

**Steal:** Adopt the dual/multi-head + shared-backbone pattern, but expand
to 10 heads (one per BoltzGen metric). Cross-attention between target
(Q9NZQ7, fixed) and nanobody candidate is exactly our setup.

---

## ABAG-Rank: Improving Model Selection of AlphaFold Antibody-Antigen Complexes by Learning to Rank (bioRxiv 2026.03.17.712376, 2026)

**Summary:** DeepSets-based ranker over AF3 decoy ensembles that uses
inter-chain residue-pair features (confidence, geometry, optional ESM
embeddings) and is permutation-invariant over variable-size pools. Beats
AF3's own internal ranking on antibody-antigen complexes — exactly the
regime where AF3 fails to distinguish good from bad designs.

**Steal:** Run BoltzGen with multiple seeds per candidate and use an ABAG-Rank-style
listwise head to pick the best decoy before computing our 10 metrics. Free win.

---

## AbRank: A Benchmark Dataset and Metric-Learning Framework for Antibody-Antigen Affinity Ranking (arxiv:2506.17857, 2025)

**Summary:** Reframes affinity prediction as pairwise ranking on 380k binding
assays with an "m-confident" filter that discards comparisons below an m-fold
gap — focusing the loss on signal, not noise. WALLE-Affinity baseline
combines PLM embeddings with structure for pairwise predictions.

**Steal:** Train our surrogate with a pairwise ranking loss (not MSE) using
our 8k archive pairs. Since the validator literally dense-ranks every metric,
a ranking loss is the matched training objective — and m-confident filtering
will skip near-ties that just add noise.

---

## AbBiBench: A Benchmark for Antibody Binding Affinity Maturation and Design (arxiv:2506.04235, 2025)

**Summary:** Benchmark of 184k experimental mutants across 14 antibodies and
9 antigens, evaluating models on whether they score the *full Ab-Ag complex*
rather than the antibody in isolation. Headline finding: structure-conditioned
inverse-folding models (AntiFold, AbMPNN) outperform PLMs on both correlation
with affinity and de novo generation.

**Steal:** Use AbMPNN / AntiFold log-likelihoods as features into our
surrogate — they correlate with affinity better than generic PLM scores.
Don't waste effort scoring sequences without antigen context.

---

## NbBench: Benchmarking Language Models for Comprehensive Nanobody Tasks (arxiv:2505.02022, 2025)

**Summary:** First unified nanobody benchmark — 8 tasks, 10 datasets across
structure annotation, binding, developability. Evaluates 11 models (general
PLMs vs antibody-specific vs nanobody-specific). Key finding: antibody LMs win
on antigen-related tasks but *all* models struggle on regression
(thermostability, affinity).

**Steal:** Initialize our surrogate from a nanobody-specific PLM (NanoBERT
or similar) — they outperform generic ESM2 on VHH tasks. Don't expect a
regressor to beat ranker for our problem (matches the AbRank finding).

---

## Nativeness-constrained diffusion framework for nanobody design (Briefings in Bioinformatics, 2025; AbNatiV2-related)

**Summary:** Scaffold-constrained diffusion that integrates an explicit
"nativeness" prior so designs stay in evolutionarily plausible sequence space
while still hitting structural targets. Demonstrates higher native-like CDR
generation than unconstrained diffusion.

**Steal:** Penalize/post-filter our designs by AbNatiV2 score before
submitting — sequences too far from natural nanobody distribution are
likely to fail BoltzGen liability/developability metrics (`liability_score`,
`liability_num_violations`).

---

## Fast and Accurate Antibody Sequence Design via Structure Retrieval (IgSeek) (arxiv:2502.19395, 2025)

**Summary:** Retrieval-based CDR design: encode a query CDR backbone with an
equivariant GNN, retrieve nearest neighbors from a natural antibody database,
copy/blend their sequences. Achieves 65% CDR-H3 recovery (typical Ab) and
63% (nanobody) without any generative training.

**Steal:** As a cheap seed generator: retrieve top-K natural VHHs near
high-scoring archive members, mutate their CDRs locally. Faster than diffusion
sampling for warm-starting our search.

---

## Protein Sequence Design with Batch Bayesian Optimisation (arxiv:2303.10429, 2023)

**Summary:** Classical batch-BO over protein fitness using CNN surrogate +
GP on top, with batch acquisition that balances exploration and exploitation
within a single round of expensive oracle calls. Outperforms naive directed
evolution on fixed budget.

**Steal:** This is the playbook for our compute budget: every BoltzGen
oracle call is ~$1, so use batch-BO to pick the next 50 sequences to score,
not 50 independent EI picks (which collapse to near-duplicates).

---

## Batched Bayesian Optimization with Correlated Candidate Uncertainties (arxiv:2410.06333, 2024)

**Summary:** Selects batches by maximizing the probability that the *true
optimum is contained in the batch*, explicitly modeling correlation between
candidate uncertainties. Outperforms standard batch EI/UCB on molecular
optimization benchmarks where candidates are similar.

**Steal:** Our archive top-K is highly correlated (many submitters cluster
on the same hot region). Use this acquisition to ensure each batch *covers
modes* rather than dog-piling one ridge.

---

## Directed Evolution of Proteins via Bayesian Optimization in Embedding Space (arxiv:2509.04998, 2025)

**Summary:** BO over PLM embedding space rather than raw sequence space —
GP on ESM features, EI acquisition, decode by nearest-neighbor or local
mutation. Smoother landscape than discrete sequence, better sample efficiency.

**Steal:** Do our BO in ESM2 embedding space; project candidates back via
nearest archive sequence + local CDR mutation. Avoids the combinatorial blowup
of full discrete sequence BO.

---

## Steering Generative Models with Experimental Data for Protein Fitness Optimization (arxiv:2505.15093, 2025)

**Summary:** Thompson-sampling-style guidance of a generative protein model
using an ensemble of neural-network regressors as value functions. Drawing
different value functions per generation step gives more exploration than a
single fixed reward — higher max fitness reached on same budget.

**Steal:** Train an ensemble of 5–10 surrogate heads on bootstrap resamples
of our 8k archive and Thompson-sample the head used to score each generated
batch. Cheap, parallelizable, well-matched to our finite-budget setup.

---

## Why risk matters for protein binder design (arxiv:2504.00146, 2025)

**Summary:** Benchmarks 72 BO configurations (encoding × surrogate ×
acquisition) on 11 binder fitness landscapes and ranks them by CVaR (worst-
case) in addition to mean performance. Finding: stochasticity of protein BO
swamps the CVaR signal — but the *budget-to-threshold* metric (how much
compute to reach fitness X) is a meaningful selection criterion.

**Steal:** When picking our BO config, measure "GPU-$ to reach top-K
rank-sum" rather than mean improvement. Bake risk awareness in: a single
flop on a $1 oracle call matters when budget is tight.

---

## Pareto-optimal sampling for multi-objective protein sequence design (MosPro) (iScience, 2025; code on GitHub)

**Summary:** Discrete sampling algorithm that adaptively weights multiple
property predictors to push designs toward the Pareto front, rather than
pre-mixing objectives with fixed weights. Adaptive weighting is the key:
it follows the gradient that *jointly* improves the most-lagging objectives.

**Steal:** Adaptive weighting is exactly what we want for rank-sum — at
each step, up-weight the metric where our candidate is ranked worst. Direct
match for "lower rank-sum wins."

---

## Improving Protein Sequence Design through Designability Preference Optimization (arxiv:2506.00297, 2025)

**Summary:** DPO-style preference fine-tuning of a protein sequence
generator using pairs of (more designable, less designable) sequences as
preference data. Improves the structural-realism axis without sacrificing
diversity, by training on rank-ordered pairs rather than scalar rewards.

**Steal:** Fine-tune our generator (BoltzGen-conditional or a small VHH LM)
on (winning archive sequence, losing archive sequence) preference pairs.
Matches our ranking-loss surrogate philosophy at the generator level.

---

## Biological Sequence Design with GFlowNets (arxiv:2203.04115, 2022; foundational)

**Summary:** GFlowNets sample sequences proportional to a reward, naturally
giving *diverse* high-reward candidates — not just the single argmax. Active
learning loop: train surrogate, sample diverse batch from GFlowNet, label,
update surrogate.

**Steal:** GFlowNet sampling beats argmax/beam-search for anti-saturation —
when the archive top is crowded, you want diverse high-scoring designs, not
1000 copies of the same ridge.

---

## Why ranking matters: the rank-sum target (synthesis note)

The validator's scoring is *dense rank-sum across 10 metrics*. Three
implications directly informed by the papers above:

1. **Pairwise/listwise ranking losses beat regression** (AbRank, MosPro).
   Train surrogates with margin-ranking on archive triples
   (i, j, sign(rank_i − rank_j)) — not MSE on raw metric values.

2. **Adaptive metric weighting** (MosPro): at each round, pump the metric
   where the candidate is *currently worst*, since one bad rank dominates
   the sum.

3. **Top-4 focus is correct but watch the long tail.** Optimizing only
   interaction_pae / design_to_target_iptm / delta_sasa_refolded / design_ptm
   leaves liability_num_violations and liability_score to randomly tank
   submissions. Penalize liabilities as a *hard filter* (AbNatiV2 humanness
   prior, BindEnergyCraft Rosetta filters) before the rank-sum stage.

---

## What we're NOT going to use

Skipped papers that surfaced but don't fit our budget / problem shape:

- Full RFdiffusion antibody pipelines (Nature 2025) — too compute-heavy and
  not VHH-optimized; BoltzGen already covers nanobody generation.
- Generic PPI-affinity benchmarks (TopoBind, ESM2_AMP) — they predict a
  scalar ΔG, not the 10 BoltzGen interface metrics we actually need.
- Chimera-Bench / ABBibench training sets — useful for pretraining but our
  8k in-distribution archive labels are more directly aligned with the
  validator's scoring distribution.
