# Gradient GA: Gradient Genetic Algorithm for Drug Molecular Design

**Authors:** Chris Zhuang, Debadyuti Mukherjee, Yingzhou Lu, Tianfan Fu, Ruqi Zhang

**arXiv:** https://arxiv.org/abs/2502.09860

**Submitted:** February 2025

**GitHub:** https://github.com/debadyuti23/GradientGA

## Key Insight

Gradient GA incorporates gradient information from a differentiable objective function into genetic algorithms for molecular optimization. Instead of relying purely on random mutation/crossover, each candidate molecule iteratively moves toward optimal solutions by following the gradient direction in discrete molecular space, using a technique called Discrete Langevin Proposal. This achieves up to 25% improvement in top-10 score over vanilla GA with faster convergence.

## Relevance to Our Miner

The ELITE_MINER_DESIGN.md already includes a genetic algorithm component (Edge 4). Gradient GA offers a direct upgrade:

- **25% improvement over vanilla GA:** Our current GA design uses random mutation/crossover operators. Replacing these with gradient-guided proposals could significantly improve the quality of evolved molecules.
- **Discrete Langevin Proposal:** Enables gradient-based optimization in discrete SMILES/molecular graph space, which is the representation our miner works with.
- **Differentiable surrogate required:** Gradient GA needs a differentiable objective function. Our planned surrogate model (MLP/GCN trained to approximate PSICHIC) would serve this role. Train surrogate -> use its gradients to guide GA mutations.
- **Convergence speed:** Faster convergence means finding better molecules within the 360-block epoch window, which is the key competitive constraint.

The combination of SALSA-style building-block active learning for SAVI search + Gradient GA for de novo molecule evolution gives us two complementary search strategies.

## Code/Repo

- GitHub: https://github.com/debadyuti23/GradientGA
