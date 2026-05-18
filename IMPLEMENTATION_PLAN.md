# NOVA Elite Miner - Implementation Plan

## Overview

This plan outlines the step-by-step implementation of the elite NOVA miner. Each phase builds on the previous, allowing iterative testing and refinement.

---

## Phase 1: Foundation & Boltz2 Integration (Critical Path)

**Goal**: Implement the missing 50% of incentive mechanism

### Tasks

#### 1.1 Create Modular Project Structure
```bash
elite_miner/
├── __init__.py
├── miner.py              # Main orchestrator
├── config.py             # Configuration management
├── data/
│   ├── __init__.py
│   ├── savi_loader.py
│   ├── chembl_client.py
│   └── submission_history.py
├── search/
│   ├── __init__.py
│   └── similarity_search.py
├── scoring/
│   ├── __init__.py
│   ├── psichic_scorer.py
│   ├── boltz2_scorer.py
│   └── composite_scorer.py
├── optimization/
│   ├── __init__.py
│   └── validation.py
└── submission/
    ├── __init__.py
    └── github_manager.py
```

#### 1.2 Boltz2 Integration
```python
# elite_miner/scoring/boltz2_scorer.py

import subprocess
import tempfile
import yaml
from pathlib import Path

class Boltz2Scorer:
    """Integrate Boltz2 for structural binding affinity prediction."""

    def __init__(self, device='cuda'):
        self.device = device
        self._verify_installation()

    def _verify_installation(self):
        """Verify Boltz2 is installed."""
        try:
            result = subprocess.run(
                ['boltz', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError("Boltz2 not installed. Run: pip install boltz[cuda]")
        except FileNotFoundError:
            raise RuntimeError("Boltz2 not found. Run: pip install boltz[cuda]")

    def create_input_yaml(self, smiles: str, protein_sequence: str,
                          output_dir: Path) -> Path:
        """Create Boltz2 input YAML for small molecule-protein complex."""
        input_data = {
            'version': 1,
            'sequences': [
                {
                    'protein': {
                        'id': 'target',
                        'sequence': protein_sequence
                    }
                },
                {
                    'ligand': {
                        'id': 'molecule',
                        'smiles': smiles
                    }
                }
            ]
        }

        yaml_path = output_dir / 'input.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(input_data, f)

        return yaml_path

    def predict_single(self, smiles: str, protein_sequence: str) -> dict:
        """Predict binding affinity for single molecule."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            yaml_path = self.create_input_yaml(smiles, protein_sequence, tmpdir)

            # Run Boltz2
            result = subprocess.run([
                'boltz', 'predict',
                str(yaml_path),
                '--out_dir', str(tmpdir / 'output'),
                '--use_msa_server'
            ], capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                return {
                    'affinity_probability_binary': 0.0,
                    'affinity_pred_value': float('-inf'),
                    'error': result.stderr
                }

            # Parse output
            return self._parse_output(tmpdir / 'output')

    def predict_batch(self, smiles_list: list, protein_sequence: str) -> list:
        """Predict binding affinity for batch of molecules."""
        results = []
        for smiles in smiles_list:
            try:
                result = self.predict_single(smiles, protein_sequence)
                results.append(result)
            except Exception as e:
                results.append({
                    'affinity_probability_binary': 0.0,
                    'affinity_pred_value': float('-inf'),
                    'error': str(e)
                })
        return results

    def _parse_output(self, output_dir: Path) -> dict:
        """Parse Boltz2 output for affinity metrics."""
        # Boltz2 outputs JSON with predictions
        import json
        predictions_file = output_dir / 'predictions.json'

        if not predictions_file.exists():
            return {
                'affinity_probability_binary': 0.0,
                'affinity_pred_value': float('-inf'),
                'error': 'No predictions file found'
            }

        with open(predictions_file) as f:
            data = json.load(f)

        return {
            'affinity_probability_binary': data.get('affinity_probability_binary', 0.0),
            'affinity_pred_value': data.get('affinity_pred_value', float('-inf')),
            'confidence': data.get('confidence_score', 0.0)
        }
```

#### 1.3 ChEMBL Known Binders Client
```python
# elite_miner/data/chembl_client.py

from chembl_webresource_client.new_client import new_client
from functools import lru_cache
import bittensor as bt

class ChEMBLClient:
    """Query ChEMBL for known protein binders."""

    def __init__(self):
        self.target_api = new_client.target
        self.activity_api = new_client.activity
        self.molecule_api = new_client.molecule

    @lru_cache(maxsize=100)
    def get_chembl_target_id(self, uniprot_id: str) -> str:
        """Convert UniProt ID to ChEMBL target ID."""
        try:
            targets = self.target_api.filter(
                target_components__accession=uniprot_id
            ).only('target_chembl_id')

            if targets:
                return targets[0]['target_chembl_id']
            return None
        except Exception as e:
            bt.logging.warning(f"ChEMBL target lookup failed: {e}")
            return None

    def get_known_binders(self, uniprot_id: str,
                          max_ic50_nm: float = 10000,
                          limit: int = 500) -> list:
        """Get known active compounds for a protein target."""
        target_id = self.get_chembl_target_id(uniprot_id)
        if not target_id:
            return []

        try:
            activities = self.activity_api.filter(
                target_chembl_id=target_id,
                standard_type__in=['IC50', 'Ki', 'Kd'],
                standard_value__lte=max_ic50_nm,
                standard_units='nM'
            ).only(
                'canonical_smiles',
                'standard_value',
                'standard_type'
            )

            # Collect unique SMILES
            seen_smiles = set()
            binders = []
            for act in activities[:limit * 2]:  # Fetch extra to account for duplicates
                smiles = act.get('canonical_smiles')
                if smiles and smiles not in seen_smiles:
                    seen_smiles.add(smiles)
                    binders.append({
                        'smiles': smiles,
                        'activity': act.get('standard_value'),
                        'type': act.get('standard_type')
                    })
                    if len(binders) >= limit:
                        break

            bt.logging.info(f"Found {len(binders)} known binders for {uniprot_id}")
            return binders

        except Exception as e:
            bt.logging.warning(f"ChEMBL activity lookup failed: {e}")
            return []
```

#### 1.4 Similarity Search Implementation
```python
# elite_miner/search/similarity_search.py

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
import numpy as np
from datasets import load_dataset
from huggingface_hub import list_repo_files
import random
import bittensor as bt

RDLogger.DisableLog('rdApp.*')

class SimilaritySearch:
    """Find similar molecules in SAVI-2020 based on known binders."""

    def __init__(self, dataset_repo='Metanova/SAVI-2020'):
        self.dataset_repo = dataset_repo
        self.fingerprint_cache = {}

    def compute_fingerprint(self, smiles: str, radius: int = 2,
                           n_bits: int = 2048):
        """Compute Morgan fingerprint for molecule."""
        if smiles in self.fingerprint_cache:
            return self.fingerprint_cache[smiles]

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            self.fingerprint_cache[smiles] = fp
            return fp
        except:
            return None

    def tanimoto_similarity(self, fp1, fp2) -> float:
        """Compute Tanimoto similarity between fingerprints."""
        if fp1 is None or fp2 is None:
            return 0.0
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def search_similar(self, seed_smiles: list, min_similarity: float = 0.6,
                      max_candidates: int = 5000, chunk_size: int = 1000) -> list:
        """Search SAVI-2020 for molecules similar to seeds."""

        # Compute seed fingerprints
        seed_fps = []
        for smiles in seed_smiles:
            fp = self.compute_fingerprint(smiles)
            if fp is not None:
                seed_fps.append(fp)

        if not seed_fps:
            bt.logging.warning("No valid seed fingerprints")
            return []

        # Stream SAVI-2020 and find similar molecules
        candidates = []
        files = list_repo_files(self.dataset_repo, repo_type='dataset')
        csv_files = [f for f in files if f.endswith('.csv')]

        # Sample random files to search
        random.shuffle(csv_files)

        for csv_file in csv_files[:10]:  # Search up to 10 files
            try:
                dataset = load_dataset(
                    self.dataset_repo,
                    data_files={'train': csv_file},
                    streaming=True
                )['train']

                for chunk in dataset.batch(chunk_size):
                    for i, smiles in enumerate(chunk['product_smiles']):
                        mol_fp = self.compute_fingerprint(smiles)
                        if mol_fp is None:
                            continue

                        # Check similarity to any seed
                        max_sim = max(
                            self.tanimoto_similarity(mol_fp, seed_fp)
                            for seed_fp in seed_fps
                        )

                        if max_sim >= min_similarity:
                            candidates.append({
                                'smiles': smiles,
                                'name': chunk['product_name'][i],
                                'similarity': max_sim
                            })

                        if len(candidates) >= max_candidates:
                            break

                    if len(candidates) >= max_candidates:
                        break

            except Exception as e:
                bt.logging.warning(f"Error processing {csv_file}: {e}")
                continue

            if len(candidates) >= max_candidates:
                break

        # Sort by similarity
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        bt.logging.info(f"Found {len(candidates)} similar candidates")
        return candidates
```

#### 1.5 Integration Test
```python
# tests/test_phase1.py

import pytest
from elite_miner.scoring.boltz2_scorer import Boltz2Scorer
from elite_miner.data.chembl_client import ChEMBLClient
from elite_miner.search.similarity_search import SimilaritySearch

def test_boltz2_scorer():
    scorer = Boltz2Scorer()
    result = scorer.predict_single(
        smiles='CCO',
        protein_sequence='MKTVRQERLKSIVRIL...'  # Truncated
    )
    assert 'affinity_probability_binary' in result
    assert 0 <= result['affinity_probability_binary'] <= 1

def test_chembl_client():
    client = ChEMBLClient()
    # Test with well-known target (EGFR)
    binders = client.get_known_binders('P00533', limit=10)
    assert len(binders) > 0
    assert all('smiles' in b for b in binders)

def test_similarity_search():
    search = SimilaritySearch()
    # Use known drug as seed
    candidates = search.search_similar(
        seed_smiles=['CC(=O)OC1=CC=CC=C1C(=O)O'],  # Aspirin
        max_candidates=100
    )
    assert len(candidates) > 0
```

---

## Phase 2: ML-Guided Search (Performance Enhancement)

**Goal**: Screen billions of molecules efficiently with surrogate model

### Tasks

#### 2.1 Surrogate Model Training
```python
# elite_miner/search/surrogate_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import bittensor as bt

class SurrogateModel(nn.Module):
    """Neural network surrogate for PSICHIC scoring."""

    def __init__(self, input_dim=2048, hidden_dims=[512, 256, 128]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class SurrogateTrainer:
    """Train and use surrogate model for fast screening."""

    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.target_protein = None

    def featurize(self, smiles_list: list, n_bits: int = 2048) -> np.ndarray:
        """Convert SMILES to Morgan fingerprints."""
        features = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
                    features.append(np.array(fp))
                else:
                    features.append(np.zeros(n_bits))
            except:
                features.append(np.zeros(n_bits))

        return np.array(features)

    def train(self, smiles_list: list, psichic_scores: list,
              epochs: int = 50, batch_size: int = 256,
              learning_rate: float = 1e-3):
        """Train surrogate model on PSICHIC scores."""

        # Featurize
        X = self.featurize(smiles_list)
        y = np.array(psichic_scores)

        # Create model
        self.model = SurrogateModel(input_dim=X.shape[1]).to(self.device)

        # Create dataset
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                bt.logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

        self.model.eval()

    def predict(self, smiles_list: list) -> np.ndarray:
        """Predict scores using surrogate model."""
        if self.model is None:
            raise ValueError("Model not trained")

        X = self.featurize(smiles_list)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            scores = self.model(X_tensor).cpu().numpy()

        return scores

    def screen_database(self, data_generator, top_k: int = 1000,
                       batch_size: int = 10000) -> list:
        """Screen large database and return top candidates."""
        all_candidates = []

        for batch in data_generator:
            smiles_list = [m['smiles'] for m in batch]
            scores = self.predict(smiles_list)

            for i, (mol, score) in enumerate(zip(batch, scores)):
                all_candidates.append({
                    **mol,
                    'surrogate_score': float(score)
                })

            # Keep only top candidates in memory
            all_candidates.sort(key=lambda x: x['surrogate_score'], reverse=True)
            all_candidates = all_candidates[:top_k * 2]

        return all_candidates[:top_k]
```

#### 2.2 Batch Scoring Optimization
```python
# elite_miner/scoring/psichic_scorer.py

import torch
from PSICHIC.wrapper import PsichicWrapper
import bittensor as bt

class OptimizedPsichicScorer:
    """Optimized PSICHIC scoring with batching and caching."""

    def __init__(self, batch_size: int = 128):
        self.model = PsichicWrapper()
        self.batch_size = batch_size
        self.cache = {}
        self.current_protein = None

    def set_target(self, protein_sequence: str):
        """Initialize model for target protein."""
        if protein_sequence != self.current_protein:
            torch.cuda.empty_cache()
            self.model.run_challenge_start(protein_sequence)
            self.current_protein = protein_sequence
            self.cache = {}  # Clear cache for new protein
            bt.logging.info(f"PSICHIC initialized for new protein")

    def score_batch(self, smiles_list: list) -> list:
        """Score batch of molecules with caching."""
        # Check cache
        uncached = []
        uncached_idx = []
        results = [None] * len(smiles_list)

        for i, smiles in enumerate(smiles_list):
            if smiles in self.cache:
                results[i] = self.cache[smiles]
            else:
                uncached.append(smiles)
                uncached_idx.append(i)

        if uncached:
            # Score uncached molecules
            df = self.model.run_validation(uncached)

            for j, idx in enumerate(uncached_idx):
                score = df.iloc[j]['predicted_binding_affinity']
                self.cache[uncached[j]] = score
                results[idx] = score

        return results

    def score_with_progress(self, smiles_list: list) -> list:
        """Score with progress logging."""
        all_scores = []
        total = len(smiles_list)

        for i in range(0, total, self.batch_size):
            batch = smiles_list[i:i + self.batch_size]
            scores = self.score_batch(batch)
            all_scores.extend(scores)

            if (i + self.batch_size) % (self.batch_size * 10) == 0:
                bt.logging.info(f"Scored {min(i + self.batch_size, total)}/{total} molecules")

        return all_scores
```

---

## Phase 3: Genetic Algorithm & Optimization

**Goal**: Discover novel molecules and multi-objective optimization

### Tasks

#### 3.1 Genetic Algorithm for Molecule Evolution
```python
# elite_miner/search/genetic_algorithm.py

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import random
import numpy as np
import bittensor as bt

class MoleculeGA:
    """Genetic algorithm for molecular optimization."""

    def __init__(self, population_size: int = 100,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _mutate_atom(self, mol):
        """Randomly change an atom type."""
        try:
            mol = Chem.RWMol(mol)
            atoms = [a for a in mol.GetAtoms() if a.GetSymbol() not in ['H']]
            if not atoms:
                return mol.GetMol()

            atom = random.choice(atoms)
            new_type = random.choice(['C', 'N', 'O', 'S', 'F', 'Cl'])
            atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_type))

            Chem.SanitizeMol(mol)
            return mol.GetMol()
        except:
            return None

    def _add_atom(self, mol):
        """Add an atom to the molecule."""
        try:
            mol = Chem.RWMol(mol)
            atoms = list(mol.GetAtoms())
            if not atoms:
                return mol.GetMol()

            parent = random.choice(atoms)
            new_type = random.choice(['C', 'N', 'O'])
            new_idx = mol.AddAtom(Chem.Atom(new_type))
            mol.AddBond(parent.GetIdx(), new_idx, Chem.BondType.SINGLE)

            Chem.SanitizeMol(mol)
            return mol.GetMol()
        except:
            return None

    def _remove_atom(self, mol):
        """Remove a terminal atom."""
        try:
            mol = Chem.RWMol(mol)
            terminal = [a for a in mol.GetAtoms()
                       if a.GetDegree() == 1 and a.GetSymbol() != 'H']
            if not terminal:
                return mol.GetMol()

            atom = random.choice(terminal)
            mol.RemoveAtom(atom.GetIdx())

            Chem.SanitizeMol(mol)
            return mol.GetMol()
        except:
            return None

    def mutate(self, smiles: str) -> str:
        """Apply random mutation to molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mutation_ops = [self._mutate_atom, self._add_atom, self._remove_atom]
        op = random.choice(mutation_ops)
        mutant = op(mol)

        if mutant is None:
            return None

        return Chem.MolToSmiles(mutant)

    def crossover(self, smiles1: str, smiles2: str) -> str:
        """Combine fragments from two molecules."""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return None

            # Fragment both molecules
            frags1 = Chem.GetMolFrags(
                AllChem.FragmentOnBRICSBonds(mol1),
                asMols=True
            )
            frags2 = Chem.GetMolFrags(
                AllChem.FragmentOnBRICSBonds(mol2),
                asMols=True
            )

            if not frags1 or not frags2:
                return smiles1

            # Combine random fragments
            frag1 = random.choice(frags1)
            frag2 = random.choice(frags2)

            # Try to join fragments
            combined = Chem.CombineMols(frag1, frag2)
            Chem.SanitizeMol(combined)

            return Chem.MolToSmiles(combined)
        except:
            return None

    def is_valid(self, smiles: str, min_heavy: int = 20,
                min_rot: int = 1, max_rot: int = 10) -> bool:
        """Check if molecule meets constraints."""
        if smiles is None:
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # Heavy atoms
            heavy = Descriptors.HeavyAtomCount(mol)
            if heavy < min_heavy:
                return False

            # Rotatable bonds
            rot = Descriptors.NumRotatableBonds(mol)
            if rot < min_rot or rot > max_rot:
                return False

            return True
        except:
            return False

    def evolve(self, seed_smiles: list, scorer, generations: int = 50) -> list:
        """Evolve population to maximize score."""

        # Initialize population
        population = seed_smiles[:self.population_size]
        if len(population) < self.population_size:
            population = population * (self.population_size // len(population) + 1)
            population = population[:self.population_size]

        best_ever = None
        best_score = float('-inf')

        for gen in range(generations):
            # Score population
            scores = scorer.score_batch(population)

            # Track best
            max_idx = np.argmax(scores)
            if scores[max_idx] > best_score:
                best_score = scores[max_idx]
                best_ever = population[max_idx]

            # Select parents (tournament selection)
            parents = []
            for _ in range(self.population_size // 2):
                idx1, idx2 = random.sample(range(len(population)), 2)
                if scores[idx1] > scores[idx2]:
                    parents.append(population[idx1])
                else:
                    parents.append(population[idx2])

            # Generate offspring
            children = []
            while len(children) < self.population_size // 2:
                p1, p2 = random.sample(parents, 2)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.crossover(p1, p2)
                else:
                    child = random.choice([p1, p2])

                # Mutation
                if child and random.random() < self.mutation_rate:
                    child = self.mutate(child)

                # Validate
                if self.is_valid(child):
                    children.append(child)

            population = parents + children

            if (gen + 1) % 10 == 0:
                bt.logging.info(
                    f"Generation {gen+1}/{generations}, "
                    f"Best: {best_score:.4f}"
                )

        return population, best_ever, best_score
```

#### 3.2 Multi-Objective Pareto Optimization
```python
# elite_miner/optimization/pareto.py

import numpy as np

class ParetoOptimizer:
    """Multi-objective Pareto optimization."""

    def __init__(self):
        self.objectives = []

    def is_dominated(self, scores_a: dict, scores_b: dict,
                    objectives: list) -> bool:
        """Check if a is dominated by b (b is better in all objectives)."""
        better_or_equal = all(
            scores_b[obj] >= scores_a[obj]
            for obj in objectives
        )
        strictly_better = any(
            scores_b[obj] > scores_a[obj]
            for obj in objectives
        )
        return better_or_equal and strictly_better

    def find_pareto_front(self, candidates: list,
                         scores_dict: dict) -> list:
        """
        Find Pareto-optimal solutions.

        Args:
            candidates: List of candidate molecules
            scores_dict: {objective_name: [scores for each candidate]}

        Returns:
            List of Pareto-optimal candidates
        """
        n = len(candidates)
        objectives = list(scores_dict.keys())

        # Build per-candidate score dicts
        candidate_scores = []
        for i in range(n):
            scores = {obj: scores_dict[obj][i] for obj in objectives}
            candidate_scores.append(scores)

        # Find dominated candidates
        dominated = [False] * n
        for i in range(n):
            for j in range(n):
                if i != j and not dominated[i]:
                    if self.is_dominated(candidate_scores[i],
                                        candidate_scores[j],
                                        objectives):
                        dominated[i] = True
                        break

        # Return non-dominated
        pareto_front = [
            candidates[i] for i in range(n) if not dominated[i]
        ]

        return pareto_front

    def select_from_pareto(self, pareto_front: list,
                          scores_dict: dict,
                          weights: dict = None) -> object:
        """
        Select single candidate from Pareto front.

        Args:
            pareto_front: Pareto-optimal candidates
            scores_dict: Objective scores
            weights: Objective weights (default: equal)

        Returns:
            Selected candidate
        """
        if len(pareto_front) == 1:
            return pareto_front[0]

        objectives = list(scores_dict.keys())
        if weights is None:
            weights = {obj: 1.0 for obj in objectives}

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        # Compute weighted scores
        weighted_scores = []
        for candidate in pareto_front:
            idx = scores_dict['_candidates'].index(candidate)
            score = sum(
                weights[obj] * scores_dict[obj][idx]
                for obj in objectives
            )
            weighted_scores.append(score)

        best_idx = np.argmax(weighted_scores)
        return pareto_front[best_idx]
```

---

## Phase 4: Production Deployment

**Goal**: Reliable, monitored production miner

### Tasks

#### 4.1 Main Orchestrator
```python
# elite_miner/miner.py

import asyncio
import bittensor as bt
from elite_miner.config import Config
from elite_miner.data.chembl_client import ChEMBLClient
from elite_miner.data.submission_history import SubmissionHistory
from elite_miner.search.similarity_search import SimilaritySearch
from elite_miner.search.surrogate_model import SurrogateTrainer
from elite_miner.search.genetic_algorithm import MoleculeGA
from elite_miner.scoring.psichic_scorer import OptimizedPsichicScorer
from elite_miner.scoring.boltz2_scorer import Boltz2Scorer
from elite_miner.optimization.pareto import ParetoOptimizer
from elite_miner.submission.github_manager import GitHubManager

class EliteMiner:
    """Main orchestrator for elite NOVA mining."""

    def __init__(self, config: Config):
        self.config = config

        # Initialize components
        self.chembl = ChEMBLClient()
        self.similarity = SimilaritySearch()
        self.surrogate = SurrogateTrainer()
        self.genetic = MoleculeGA()
        self.psichic = OptimizedPsichicScorer()
        self.boltz2 = Boltz2Scorer()
        self.pareto = ParetoOptimizer()
        self.history = SubmissionHistory()
        self.github = GitHubManager(config)

        self.current_epoch = None
        self.current_target = None

    async def run_epoch(self, target_protein: str, antitargets: list,
                       epoch: int, protein_sequence: str):
        """Execute complete mining epoch."""

        bt.logging.info(f"Starting epoch {epoch} for target {target_protein}")
        self.current_epoch = epoch
        self.current_target = target_protein

        # Phase 1: Bootstrap with known binders
        bt.logging.info("Phase 1: Getting known binders from ChEMBL...")
        known_binders = self.chembl.get_known_binders(target_protein)
        seed_smiles = [b['smiles'] for b in known_binders]

        if not seed_smiles:
            bt.logging.warning("No known binders found, using random seeds")
            seed_smiles = self._get_random_seeds()

        # Phase 2: Similarity search
        bt.logging.info("Phase 2: Running similarity search...")
        similar_candidates = self.similarity.search_similar(
            seed_smiles,
            min_similarity=0.6,
            max_candidates=self.config.similarity_candidates
        )

        # Phase 3: Train/update surrogate
        if self._should_train_surrogate():
            bt.logging.info("Phase 3: Training surrogate model...")
            training_smiles = [c['smiles'] for c in similar_candidates[:5000]]
            self.psichic.set_target(protein_sequence)
            training_scores = self.psichic.score_batch(training_smiles)
            self.surrogate.train(training_smiles, training_scores)

        # Phase 4: ML-guided search
        bt.logging.info("Phase 4: Running ML-guided search...")
        ml_candidates = self._run_surrogate_search()

        # Phase 5: Genetic algorithm
        bt.logging.info("Phase 5: Running genetic algorithm...")
        ga_population, ga_best, ga_score = self.genetic.evolve(
            seed_smiles[:50],
            self.psichic,
            generations=30
        )

        # Phase 6: Combine all candidates
        all_smiles = list(set(
            [c['smiles'] for c in similar_candidates] +
            [c['smiles'] for c in ml_candidates] +
            ga_population
        ))
        bt.logging.info(f"Combined {len(all_smiles)} unique candidates")

        # Phase 7: Filter by properties
        valid_smiles = self._filter_properties(all_smiles)

        # Phase 8: PSICHIC scoring
        bt.logging.info("Phase 8: PSICHIC scoring...")
        self.psichic.set_target(protein_sequence)
        psichic_scores = self.psichic.score_with_progress(valid_smiles)

        # Phase 9: Select top for Boltz2
        top_k = min(100, len(valid_smiles))
        top_indices = np.argsort(psichic_scores)[-top_k:]
        top_smiles = [valid_smiles[i] for i in top_indices]
        top_psichic = [psichic_scores[i] for i in top_indices]

        # Phase 10: Boltz2 scoring (KEY DIFFERENTIATOR)
        bt.logging.info("Phase 10: Boltz2 scoring...")
        boltz2_results = self.boltz2.predict_batch(top_smiles, protein_sequence)
        boltz2_scores = [r['affinity_probability_binary'] for r in boltz2_results]

        # Phase 11: Entropy scoring
        entropy_scores = self._compute_entropy_scores(top_smiles)

        # Phase 12: Pareto optimization
        bt.logging.info("Phase 11: Pareto optimization...")
        scores_dict = {
            'psichic': top_psichic,
            'boltz2': boltz2_scores,
            'entropy': entropy_scores,
            '_candidates': top_smiles
        }

        pareto_front = self.pareto.find_pareto_front(top_smiles, scores_dict)
        final_selection = self.pareto.select_from_pareto(
            pareto_front,
            scores_dict,
            weights=self._get_epoch_weights()
        )

        # Phase 13: Submit
        bt.logging.info(f"Final selection: {final_selection}")
        await self._submit(final_selection, top_smiles)

        # Phase 14: Record history
        self.history.record(final_selection, target_protein, epoch)

    def _get_epoch_weights(self):
        """Get objective weights based on current epoch."""
        entropy_weight = self.config.entropy_start + \
            (self.current_epoch - self.config.entropy_start_epoch) * \
            self.config.entropy_step

        if entropy_weight < 0.5:
            return {'psichic': 0.5, 'boltz2': 0.4, 'entropy': 0.1}
        elif entropy_weight < 1.0:
            return {'psichic': 0.4, 'boltz2': 0.3, 'entropy': 0.3}
        else:
            return {'psichic': 0.3, 'boltz2': 0.2, 'entropy': 0.5}

    # ... additional helper methods
```

#### 4.2 Basilica Deployment
```bash
#!/bin/bash
# deploy_basilica.sh

# Provision GPU instance
basilica up \
    --gpu-count 1 \
    --gpu-type a100-80gb \
    --name nova-elite-miner \
    --image nvidia/cuda:12.6.0-runtime-ubuntu24.04 \
    --memory-mb 131072 \
    -d

# Wait for instance
sleep 30

# Transfer code
basilica cp -r . nova-elite-miner:/workspace/nova/
basilica cp .env nova-elite-miner:/workspace/nova/

# Install dependencies
basilica exec nova-elite-miner "cd /workspace/nova && pip install -e . && pip install boltz[cuda]"

# Run miner
basilica exec nova-elite-miner "cd /workspace/nova && python -m elite_miner.miner \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $HOTKEY \
    --logging.info"
```

---

## Testing Strategy

### Unit Tests
- Each module has isolated unit tests
- Mock external services (ChEMBL, PSICHIC, Boltz2)

### Integration Tests
- Test complete pipeline with small molecule set
- Verify scoring consistency

### Performance Tests
- Benchmark surrogate model speed
- Measure PSICHIC/Boltz2 throughput
- Track memory usage

### Production Validation
- Run on testnet first
- Compare scores with stock miner
- Monitor submission success rate

---

## Success Criteria

| Phase | Criteria |
|-------|----------|
| 1 | Boltz2 integration working, similarity search finding candidates |
| 2 | Surrogate model achieving >0.8 correlation with PSICHIC |
| 3 | GA discovering valid novel molecules, Pareto selection working |
| 4 | Stable production deployment, >99% submission success |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Boltz2 slow/timeout | Batch smaller, add timeout handling |
| Surrogate inaccurate | Continuous retraining, ensemble models |
| GA invalid molecules | Strict validation, SMILES sanitization |
| ChEMBL rate limits | Cache results, local mirror |
| GPU OOM | Sequential scoring, memory management |
