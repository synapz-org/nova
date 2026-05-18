# Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction

**Authors:** Jeremy Wohlwend et al. (MIT / Recursion Pharmaceuticals)

**Published:** bioRxiv, June 2025

**Link:** https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1

**GitHub:** https://github.com/jwohlwend/boltz

## Key Insight

Boltz-2 is the first AI model to approach free-energy perturbation (FEP) accuracy for small molecule-protein binding affinity, while being 1000x more computationally efficient. It jointly predicts biomolecular complex structure and binding affinity, trained on ~1.2M binders from ChEMBL/BindingDB plus ~2M decoys. On the FEP+ benchmark, it achieves Pearson r=0.66 on a 4-target subset. For hit discovery, it achieves enrichment factor of 18.4 at 0.5% on MF-PCBA.

## Relevance to Our Miner

Boltz-2 scoring accounts for 50% of the NOVA SN68 incentive. This is currently 0% implemented in the stock miner -- the single biggest competitive gap.

- **Hit discovery mode:** Binary binding probability (`affinity_probability_binary`) -- useful for initial filtering of SAVI candidates.
- **Affinity regression:** Continuous log10(IC50) prediction (`affinity_pred_value`) -- useful for ranking top PSICHIC candidates.
- **Dual optimization:** Select molecules that rank well on both PSICHIC and Boltz-2 via Pareto optimization. Even a simple top-10 PSICHIC -> Boltz-2 reranking would be a major edge.
- **Speed:** Fast enough to score thousands of candidates per epoch, but not billions. Use as a second-stage validator after PSICHIC pre-filtering.
- **Controllability:** Supports pocket conditioning and distance constraints -- could guide structure-aware search.

## Reliability Caveat

A March 2026 evaluation (Wan et al., arXiv:2603.05532) found Boltz-2 shows "only weak to moderate correlations" on large datasets (16K-21K compounds) and "no significant correlation" for top compounds. It is useful for initial screening but lacks resolution for fine-grained lead ranking. This means we should use it as a coarse filter, not the sole ranker.

## Code/Repo

- GitHub: https://github.com/jwohlwend/boltz (MIT license, weights included)
