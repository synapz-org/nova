# On the Reliability of AI Methods in Drug Discovery: Evaluation of Boltz-2 for Structure and Binding Affinity Prediction

**Authors:** Shunzhou Wan, Xibei Zhang, Xiao Xue, Peter V. Coveney

**arXiv:** https://arxiv.org/abs/2603.05532

**Submitted:** March 2026

## Key Insight

Independent evaluation of Boltz-2 on large datasets (16,780 compounds for 3CLPro, 21,702 for TNKS2) found significant limitations: (1) large structural variability in predicted conformations (high RMSD variance), (2) only weak-to-moderate correlation with experimental binding energies across full datasets, and (3) no significant correlation for top-scoring compounds specifically. The authors conclude Boltz-2 is useful for initial broad screening but "lacks the energetic resolution required for lead identification."

## Relevance to Our Miner

Critical calibration for our Boltz-2 integration strategy:

- **Use as coarse filter, not fine ranker:** Boltz-2 can help separate binders from non-binders at scale, but should not be trusted for fine-grained ranking among top candidates. In our dual-optimization pipeline, PSICHIC should remain the primary ranker for final candidate selection.
- **Structural variability warning:** Multiple predicted conformations per molecule means Boltz-2 scores may be noisy. Consider running multiple predictions per candidate and averaging/ensembling for more stable scores.
- **Complementary value:** Even with weak-moderate correlation, Boltz-2 provides an orthogonal signal to PSICHIC. Molecules that score well on both are more robust candidates than those excelling on only one metric.
- **Practical implication for NOVA:** Since Boltz-2 is 50% of the incentive weight, we need to optimize for it. But this paper suggests the variance in Boltz-2 predictions creates an opportunity -- miners that account for prediction uncertainty (e.g., submit molecules with consistently high Boltz-2 scores across multiple runs) may have an edge.

## Code/Repo

- No code; evaluation paper only.
