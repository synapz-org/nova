# Active Learning on Synthons for Molecular Design (SALSA)

**Authors:** Tom George Grigg, Mason Burlage, Oliver Brook Scott, Adam Taouil, Dominique Sydow, Liam Wilbraham

**arXiv:** https://arxiv.org/abs/2505.12913

**Submitted:** May 2025

## Key Insight

SALSA (Scalable Active Learning via Synthon Acquisition) extends pool-based active learning to non-enumerable combinatorial chemical spaces by factoring modeling and acquisition over synthon/fragment choices rather than whole molecules. This lets it scale to spaces of trillions of compounds while maintaining sample efficiency. Tested on ligand- and structure-based objectives across three protein targets, SALSA-generated molecules showed comparable chemical profiles to known bioactives with greater diversity and higher scores than leading generative methods.

## Relevance to Our Miner

This is highly actionable for our combinatorial search strategy:

- **Combinatorial factorization:** SAVI-2020 is built from ~150K building blocks via 53 reaction transforms. SALSA's approach of factoring over synthon/fragment choices maps directly to SAVI's building-block structure. Instead of enumerating and scoring all 1.75B products, we can learn which building blocks (scaffolds, boronic acids in rxn:5) lead to high-scoring products and focus combinatorial search there.
- **Active learning loop:** Sample a batch of building-block combinations -> score with PSICHIC -> update surrogate model over building blocks -> select next batch. This is more sample-efficient than scoring random molecules.
- **Non-enumerable spaces:** SALSA handles spaces too large to enumerate, which is exactly our situation with SAVI's combinatorial explosion.
- **Direct implementation path:** Our `elite_miner/searcher.py` already loads rxn:5 building blocks (scaffolds + boronic acids). We could implement SALSA-style acquisition over these building blocks rather than random combinatorial sampling.

## Code/Repo

- No public repo mentioned in paper (as of search date).
