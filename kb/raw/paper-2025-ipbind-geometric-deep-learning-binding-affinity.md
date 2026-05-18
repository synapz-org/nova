# IPBind: Accurate and Generalizable Protein-Ligand Binding Affinity Prediction with Geometric Deep Learning

**Authors:** Krinos Li, Xianglu Xiao, Zijun Zhong, Guang Yang

**arXiv:** https://arxiv.org/abs/2504.16261

**Submitted:** April 2025

## Key Insight

IPBind uses geometric deep learning and machine learning interatomic potentials to predict binding affinity by comparing a complex's bound and unbound states. It reportedly matches or exceeds PSICHIC's performance across multiple benchmarks (1.561, 1.290, 0.478, 0.763 on standard test sets). The key innovation is leveraging the energy difference between bound and unbound conformations as a physically grounded signal.

## Relevance to Our Miner

IPBind's relevance is as a potential alternative or supplementary scoring model:

- **PSICHIC comparison:** If IPBind correlates with PSICHIC scores, it could serve as an additional fast pre-filter or cross-validation signal. Molecules that score highly on both PSICHIC and IPBind are more likely to be genuine binders.
- **Geometric features:** IPBind's use of interatomic potential features could inform what molecular properties PSICHIC implicitly values. Understanding the overlap between structure-based and sequence-based scoring could help design better surrogate models.
- **Requires 3D structures:** Unlike PSICHIC (sequence-only), IPBind needs structural data, making it less practical for billion-scale screening. More useful as a late-stage validator.
- **Benchmark context:** Knowing that IPBind matches PSICHIC on standard benchmarks tells us PSICHIC is competitive with the latest methods, confirming that beating it requires genuine molecular optimization, not just model gaming.

## Code/Repo

- No public repo mentioned as of search date.
