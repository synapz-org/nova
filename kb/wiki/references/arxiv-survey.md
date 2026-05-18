# arXiv Research Survey: MetaNova NOVA (SN68) Miner Optimization

**Date:** 2026-04-02
**Scope:** Papers from 2024-2026 relevant to competitive mining on Bittensor SN68

---

## Subnet Context

NOVA SN68 is a molecular search competition. Miners search the SAVI-2020 database (1.75B synthesizable molecules) to find the highest-scoring candidate for a given protein target. Scoring uses two oracles:

- **PSICHIC (50% weight):** Sequence-based GNN that predicts binding affinity from protein sequence + SMILES
- **Boltz-2 (50% weight):** Structure prediction model that estimates binding probability and affinity

Miners submit one molecule at a time over 360-block epochs. Winner-takes-all per epoch.

---

## Research Landscape

The literature clusters into five areas directly relevant to our competitive strategy:

### 1. Understanding the Oracles

| Paper | Year | Actionability |
|-------|------|--------------|
| **PSICHIC** (Koh et al., Nature Machine Intelligence) | 2024 | Foundational -- this IS the oracle. Study interaction fingerprints to understand what drives high scores. |
| **Boltz-2** (Wohlwend et al., bioRxiv) | 2025 | 50% of incentive, MIT-licensed, 1000x faster than FEP. Must integrate. |
| **Boltz-2 Reliability Evaluation** (Wan et al., arXiv) | 2026 | Calibration: weak correlation on large datasets, no correlation for top compounds. Use as coarse filter. |
| **IPBind** (Li et al., arXiv) | 2025 | Matches PSICHIC on benchmarks. Useful for understanding what structural features matter. |

**Key takeaway:** PSICHIC should remain our primary optimization target for fine-grained ranking. Boltz-2 is essential for the 50% incentive weight but its predictions are noisy -- optimize for consistently good Boltz-2 scores rather than chasing the maximum.

### 2. Billion-Scale Screening (Surrogate Models)

| Paper | Year | Actionability |
|-------|------|--------------|
| **SPRINT** (McNutt et al., arXiv) | 2024 | Screened 6.7B molecules in 16 minutes via co-embeddings. Could pre-filter SAVI-2020 down to manageable candidate sets. |
| **MolLIBRA** (Okada et al., arXiv) | 2026 | Multi-fingerprint GP ensemble for pre-ranking before oracle calls. Directly applicable to our PSICHIC evaluation bottleneck. |

**Key takeaway:** We cannot score 1.75B molecules with PSICHIC. Two complementary approaches: (a) SPRINT-style co-embedding for coarse pre-filtering at full scale, (b) MolLIBRA-style GP ensemble for fine-grained pre-ranking of the filtered set. The multi-fingerprint ensemble (ECFP + MACCS + Topological Torsion) is better than single-fingerprint MLP.

### 3. Active Learning for Combinatorial Libraries

| Paper | Year | Actionability |
|-------|------|--------------|
| **SALSA** (Grigg et al., arXiv) | 2025 | Factors active learning over synthon/fragment choices. Maps directly to SAVI's building-block structure. Highest-priority implementation target. |
| **MolPAL** (Graff et al.) | 2021 | Established framework for pool-based active learning in virtual screening. Bayesian optimization with surrogate model. |

**Key takeaway:** SALSA is the most actionable paper for our miner. SAVI-2020 is built from ~150K building blocks via reaction transforms. Instead of scoring random products, learn which building blocks produce high-scoring products and focus search there. This is fundamentally more sample-efficient than whole-molecule screening.

### 4. Genetic Algorithm Enhancement

| Paper | Year | Actionability |
|-------|------|--------------|
| **Gradient GA** (Zhuang et al., arXiv) | 2025 | 25% improvement over vanilla GA via gradient-guided mutations in discrete molecular space. Code available. |
| **MolLIBRA** (Okada et al., arXiv) | 2026 | Multi-fingerprint surrogates for GA candidate pre-ranking. |

**Key takeaway:** Our planned GA (Edge 4 in ELITE_MINER_DESIGN.md) should use Gradient GA's Discrete Langevin Proposal instead of random mutations. Requires a differentiable surrogate model, which we are already planning to build. The 25% improvement in top-10 score could be decisive in winner-takes-all competition.

### 5. Multi-Objective Optimization

| Paper | Year | Actionability |
|-------|------|--------------|
| **CheapVS** (Dang et al., arXiv) | 2025 | Preferential multi-objective Bayesian optimization. Recovered known drugs while screening only 6% of library. |
| **Boltz-2 Reliability** (Wan et al., arXiv) | 2026 | Confirms need for multi-objective approach -- no single oracle is reliable alone. |

**Key takeaway:** With PSICHIC and Boltz-2 as competing objectives (plus entropy), Pareto optimization is essential. CheapVS's approach of incorporating preferences into the optimization could help us weight objectives dynamically based on epoch phase (early = affinity focus, late = entropy focus).

---

## Priority Implementation Roadmap

Based on this research survey, the highest-impact improvements to our miner, ordered by expected competitive edge:

### Tier 1: Must-Have (Week 1-2)
1. **Boltz-2 integration** -- Currently 0% implemented, worth 50% of incentive. Use the MIT-licensed code at github.com/jwohlwend/boltz. Apply as second-stage scorer on top-10 PSICHIC candidates.
2. **SALSA-style building-block active learning** -- Factor surrogate model over SAVI building blocks (scaffolds + boronic acids) instead of random product sampling. Most sample-efficient approach for combinatorial search.

### Tier 2: High-Value (Week 2-3)
3. **Multi-fingerprint surrogate ensemble** -- Replace single MLP with GP ensemble over ECFP, MACCS, and Topological Torsion fingerprints (MolLIBRA approach). Better uncertainty estimates enable smarter acquisition.
4. **Gradient GA upgrade** -- Replace random mutations with Discrete Langevin Proposal from Gradient GA paper. Requires differentiable surrogate (pairs with item 3). Code available at github.com/debadyuti23/GradientGA.

### Tier 3: Experimental Edge (Week 3-4)
5. **SPRINT-style co-embedding pre-filter** -- If we can train a fast co-embedding model on PSICHIC scores, use it to pre-filter the full 1.75B SAVI database before detailed scoring. High upside but significant engineering effort.
6. **Boltz-2 uncertainty averaging** -- Run multiple Boltz-2 predictions per candidate, average scores for stability. Exploits the known prediction variance to get more reliable rankings.

---

## Papers by File

All raw paper summaries are in `~/Projects/nova/kb/raw/`:

| File | Paper |
|------|-------|
| `paper-2024-psichic-physicochemical-gnn-interaction-fingerprints.md` | PSICHIC (oracle model) |
| `paper-2025-boltz2-binding-affinity-prediction.md` | Boltz-2 (structure oracle) |
| `paper-2026-boltz2-reliability-evaluation.md` | Boltz-2 reliability analysis |
| `paper-2025-sprint-billion-scale-virtual-screening.md` | SPRINT billion-scale screening |
| `paper-2025-salsa-active-learning-synthons.md` | SALSA synthon active learning |
| `paper-2025-gradient-ga-molecular-optimization.md` | Gradient GA |
| `paper-2025-ipbind-geometric-deep-learning-binding-affinity.md` | IPBind (PSICHIC comparison) |
| `paper-2025-mollibra-multi-fingerprint-genetic-optimization.md` | MolLIBRA multi-fingerprint surrogates |
