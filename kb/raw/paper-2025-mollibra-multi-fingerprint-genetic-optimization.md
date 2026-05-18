# MolLIBRA: Genetic Molecular Optimization with Multi-Fingerprint Surrogates and Text-Molecule Aligned Critic

**Authors:** Masahi Okada, Kazuki Sakai, Hiroaki Yoshida, Masaki Okoshi, Tadahiro Taniguchi

**arXiv:** https://arxiv.org/abs/2602.07002

**Submitted:** February 2026

## Key Insight

MolLIBRA addresses the oracle evaluation bottleneck in molecular optimization by pre-ranking GA candidates using multiple critics before expensive scoring. It uses an ensemble of Gaussian process surrogates trained on different molecular fingerprint types (ECFP, MACCS, etc.), combined with CLAMP (a pretrained text-molecule alignment model). On PMO-1K benchmark, the best variant outperformed baselines on 14/22 tasks.

## Relevance to Our Miner

This paper directly addresses our core challenge -- limited PSICHIC evaluations per epoch:

- **Multi-fingerprint ensemble:** Rather than relying on a single fingerprint (our current plan: ECFP4), use an ensemble of GPs trained on ECFP, MACCS, Topological Torsion, and RDKit fingerprints. The ensemble adaptively selects which representation best captures the scoring landscape for each target protein.
- **Pre-ranking before oracle calls:** In our pipeline, PSICHIC is the expensive oracle. Using MolLIBRA-style surrogate ranking to select which candidates get PSICHIC evaluation could dramatically improve hit rates within our evaluation budget.
- **GP surrogates over MLP:** Gaussian processes provide uncertainty estimates, which enable acquisition functions (expected improvement, UCB) for more principled active learning. Our ELITE_MINER_DESIGN currently plans a simple MLP surrogate -- GPs might be more sample-efficient given the limited training data from PSICHIC evaluations.
- **Fingerprint diversity = entropy:** Using multiple fingerprint types naturally captures different aspects of molecular structure, which could also help with the entropy maximization component of NOVA scoring.

## Code/Repo

- No public repo mentioned as of search date.
