# Competitive NOVA SN68 Miner Design

## Problem

The default miner streams random SAVI-2020 molecules and optimizes for PSICHIC only. But 100% of emissions now go to the Boltz2 winner (`boltz_weight: 1.0`). Only 1 miner (UID 52) currently wins all epochs (~665 TAO/day). We need a miner that optimizes for the Boltz2 scoring formula.

## Scoring Formula

Validators rank miners by:
```
boltz_score = (affinity_probability_binary - affinity_pred_value) / heavy_atom_count
```
Higher is better. This rewards small molecules with strong binding probability.

## Winning Strategy (from competition analysis)

- 100% rxn:5 (double Suzuki coupling, 3-component)
- Top scaffolds: aminomethyl-bromochlorobenzenes (mol_ids: 192490, 192488, 192710, 192713)
- Small aromatic boronic acids as coupling partners
- Products: 21-30 heavy atoms, 4-5 rotatable bonds
- Search space: ~33M with top 4 scaffolds, 99.99% unexplored

## Architecture

Standalone miner that replaces the default search logic. Plugs into existing NOVA infrastructure (wallet, subtensor, GitHub submission, DRAND encryption).

### Components

1. **CombinatorialSearcher** - Enumerates rxn:5 combinations from SQLite DB. Prioritizes top scaffolds, expands to ~50 best scaffolds. Generates SMILES via reaction logic.

2. **ValidityFilter** - Pre-filters: heavy atoms >= 10, rotatable bonds 1-10, no Se atoms, Boltz-safe (atom names <= 4 chars), unique for target protein (HF archive check).

3. **ScoreOptimizer** - PSICHIC batch scoring, ranks by `predicted_binding_affinity / heavy_atom_count` as Boltz2 proxy. Also scores against antitarget and applies penalty.

4. **EpochManager** - Main loop: generate batch, filter, score, submit best. Tracks submitted molecules across restarts.

### Data Flow

```
SQLite DB (225K building blocks)
  -> Enumerate rxn:5 combos (scaffold x boronic1 x boronic2)
  -> React molecules (get SMILES)
  -> Validity filter (HA, RB, banned atoms, Boltz-safe)
  -> Uniqueness check (HF archive)
  -> PSICHIC batch scoring (target + antitarget)
  -> Rank by proxy: (target_affinity - 0.9 * antitarget_affinity) / heavy_atoms
  -> Submit top molecule via encrypted GitHub commit
```

### Key Decisions

- rxn:5 only (all recent winners use it exclusively)
- Expand to top ~50 scaffolds for unexplored territory
- Proxy scoring: PSICHIC_affinity / HA correlates with Boltz2 formula
- Batch processing: score thousands per epoch
- Persistent state: track scored/submitted molecules across restarts

### Not Building

- No local Boltz2 inference (validators run it)
- No genetic algorithm or molecule generation
- No ML surrogate model beyond PSICHIC
- No web UI

## Validation Requirements

- min_heavy_atoms: 10
- rotatable_bonds: 1-10
- banned_atom_types: ["Se"]
- Boltz-safe: atom names <= 4 chars
- Unique for target protein (checked via HF Submission-Archive)
- Only rxn:3 or rxn:5 allowed (we use rxn:5)

## Success Criteria

- Win at least some epochs against UID 52
- Consistently submit valid, high-scoring molecules
- Survive validation checks (uniqueness, reaction type, properties)
