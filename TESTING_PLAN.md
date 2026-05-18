# NOVA Mining - Comprehensive Testing Plan

## Overview

This document outlines the testing strategy for both NOVA miners:
- **Phase 1 (elite_miner)**: For the original `nova` repo with GitHub submissions
- **Phase 2 (blueprint_miner)**: For the `nova-blueprint` sandboxed execution

---

## Phase 1: Elite Miner Testing

### Location
`/Users/dwbarnes/Projects/nova/elite_miner/`

### Dependencies
```
rdkit
torch
numpy
pandas
bittensor
requests (for ChEMBL API)
```

### Test Stages

#### Stage 1: Unit Tests
- [ ] Test `MoleculeValidator` with known valid/invalid molecules
- [ ] Test `SimilaritySearch` with sample SMILES
- [ ] Test `SurrogateModel` training and prediction
- [ ] Test `MolecularGA` mutation and crossover operations
- [ ] Test `CompositeScorer` score calculations
- [ ] Test `ParetoOptimizer` selection

#### Stage 2: Integration Tests
- [ ] Test ChEMBL client with real UniProt IDs
- [ ] Test PSICHIC wrapper with sample proteins
- [ ] Test full scoring pipeline (PSICHIC only, no Boltz2)
- [ ] Test submission history tracking

#### Stage 3: End-to-End Tests
- [ ] Run `run_epoch` with mock data
- [ ] Run `run_epoch_hybrid` with mock data
- [ ] Verify output format matches expected

### Test Command
```bash
cd /Users/dwbarnes/Projects/nova
python -m elite_miner.tests.test_all
```

---

## Phase 2: Blueprint Miner Testing

### Location
`/Users/dwbarnes/Projects/nova-blueprint/blueprint_miner/`

### Dependencies
```
rdkit
torch
numpy
sqlite3 (built-in)
```

### Test Stages

#### Stage 1: Unit Tests
- [ ] Test `MoleculeDB` database queries
- [ ] Test `FingerprintCache` computation and caching
- [ ] Test `MoleculeGenerator` random and guided generation
- [ ] Test `SurrogatePredictor` training and prediction
- [ ] Test score calculation formula

#### Stage 2: Integration Tests
- [ ] Test `ActiveLearningSearch` with mock scorer
- [ ] Test SMILES resolution from product names
- [ ] Test output writing to result.json

#### Stage 3: End-to-End Tests
- [ ] Run full miner with local database
- [ ] Verify output format: `{"molecules": ["rxn:4:123:456", ...]}`
- [ ] Test with simulated time budget

### Test Command
```bash
cd /Users/dwbarnes/Projects/nova-blueprint
python -m blueprint_miner.tests.test_all
```

---

## Local Testing Setup

### For Blueprint Miner (Phase 2)

1. **Create mock input.json**:
```json
{
  "targets": ["P00533"],
  "antitargets": ["P12345"],
  "allowed_reaction": 4
}
```

2. **Set environment variables**:
```bash
export WORKDIR=/Users/dwbarnes/Projects/nova-blueprint
export OUTPUT_DIR=/tmp/blueprint_output
export TIME_BUDGET_SEC=120  # 2 minutes for testing
```

3. **Run**:
```bash
python miner.py
```

### For Elite Miner (Phase 1)

1. **Run with dry-run mode**:
```bash
python -m elite_miner.miner --dry-run --no-boltz2
```

---

## Success Criteria

### Phase 1 (elite_miner)
- [ ] All imports resolve without errors
- [ ] Can initialize EliteMiner class
- [ ] Can run a mock epoch without crashing
- [ ] Outputs valid SMILES

### Phase 2 (blueprint_miner)
- [ ] All imports resolve without errors
- [ ] Can read from SQLite database
- [ ] Can generate valid product names
- [ ] Writes valid result.json
- [ ] Completes within time budget

---

## Performance Benchmarks

### Phase 2 Target Metrics
- Database queries: < 100ms
- Fingerprint computation: < 10ms per molecule
- Surrogate prediction: < 1ms per molecule
- Full iteration: < 30s

### Expected Throughput
- Molecules generated per minute: ~1000
- Molecules scored per minute: ~200 (PSICHIC bottleneck)
- Target total scored in 30 min: ~5000-6000

---

## Known Issues to Address

1. **Import paths**: Need to handle relative imports correctly
2. **PSICHIC availability**: Need mock scorer for testing without GPU
3. **Database path**: Must work in sandbox environment
4. **Time management**: Need accurate time tracking

---

## Next Steps

1. Create test directory structure
2. Implement unit tests
3. Run tests and fix issues
4. Document results
