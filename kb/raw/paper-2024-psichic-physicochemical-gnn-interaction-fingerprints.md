# PSICHIC: Physicochemical Graph Neural Network for Learning Protein-Ligand Interaction Fingerprints from Sequence Data

**Authors:** Huan Yee Koh, Anh T.N. Nguyen, Shirui Pan, Lauren T. May, Geoffrey I. Webb

**Published:** Nature Machine Intelligence, 2024

**arXiv/bioRxiv:** https://www.biorxiv.org/content/10.1101/2023.09.17.558145v1

**GitHub:** https://github.com/huankoh/PSICHIC

## Key Insight

PSICHIC is the deterministic oracle used by NOVA SN68. It predicts protein-ligand binding affinity using only sequence data (protein amino acid sequence + ligand SMILES) -- no 3D structures required. It incorporates physicochemical constraints into a graph neural network, enabling it to match or surpass leading structure-based methods in binding affinity prediction while also generating interpretable interaction fingerprints that identify binding-site residues and involved ligand atoms.

## Relevance to Our Miner

This is the scoring function we must optimize against. Understanding PSICHIC's architecture is critical:

- **Input format:** Protein sequence + SMILES pair. Our surrogate model should learn to approximate PSICHIC's scoring behavior on these inputs.
- **Throughput:** ~100K compounds/hour on the web server. This constrains how many candidates we can validate per epoch -- reinforcing the need for fast surrogate pre-filtering.
- **Variants:** PSICHIC_XL (large multitask) and task-specific fine-tuned variants exist. The NOVA subnet uses a fixed version for deterministic evaluation.
- **Interpretability:** PSICHIC's interaction fingerprints reveal which substructures drive high scores. Mining these patterns from top-scoring molecules could guide our search heuristics (e.g., prioritize scaffolds with favorable pharmacophore patterns).

## Code/Repo

- GitHub: https://github.com/huankoh/PSICHIC
- Web server (beta): https://www.psichicserver.com
- Google Colab notebook available in repo
