# SPRINT: Scaling Structure Aware Virtual Screening to Billions of Molecules

**Authors:** Andrew T. McNutt, Abhinav K. Adduri, Caleb N. Ellington, Monica T. Dayao, Eric P. Xing, Hosein Mohimani, David R. Koes

**arXiv:** https://arxiv.org/abs/2411.15418

**Submitted:** November 2024

**Tool:** ColabScreen (web interface)

## Key Insight

SPRINT enables screening entire chemical libraries against whole proteomes using a vector-based co-embedding approach. It uses protein language models and self-attention to learn a drug-target co-embedding space, then performs fast vector similarity search. It screened the entire ENAMINE Real Database (6.7B molecules) against the whole human proteome in 16 minutes. Achieves state-of-the-art enrichment on LIT-PCBA and DTI benchmarks.

## Relevance to Our Miner

SPRINT's approach is directly relevant to building a fast surrogate for SAVI-2020 screening:

- **Co-embedding architecture:** Instead of scoring each molecule-protein pair individually, embed both into a shared vector space. This enables nearest-neighbor search at billion scale, which is exactly our problem (1.75B SAVI molecules).
- **Protein language model backbone:** Structure-aware PLMs capture target-specific features that could correlate with PSICHIC scores, making SPRINT-style embeddings a candidate surrogate representation.
- **16 minutes for 6.7B:** This throughput is game-changing. Even if the correlation with PSICHIC is imperfect, using SPRINT to pre-filter 1.75B SAVI molecules down to 100K candidates for PSICHIC scoring would be a massive efficiency gain.
- **Interpretability:** Residue-level attention maps could identify binding hotspots, complementing PSICHIC's interaction fingerprints.

The key question is whether SPRINT's embedding similarity correlates well enough with PSICHIC scores to serve as a useful pre-filter. Worth testing.

## Code/Repo

- ColabScreen web interface (referenced in paper)
- arXiv: https://arxiv.org/abs/2411.15418
