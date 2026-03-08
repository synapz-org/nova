"""Tests for CombinatorialSearcher."""

import os
import sqlite3
import pytest

DB_PATH = os.path.join(os.path.dirname(__file__), "../../combinatorial_db/molecules.sqlite")


def test_db_exists():
    """Verify molecules.sqlite exists and has data."""
    assert os.path.exists(DB_PATH), f"Database not found at {DB_PATH}"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT COUNT(*) FROM molecules")
    count = cursor.fetchone()[0]
    conn.close()
    assert count > 0, "molecules table is empty"


def test_load_building_blocks():
    """Verify scaffolds and boronic acids load with correct role_masks."""
    from elite_miner.searcher import CombinatorialSearcher

    searcher = CombinatorialSearcher(DB_PATH)
    searcher.load_rxn5_building_blocks()

    assert len(searcher.scaffolds) > 0, "No scaffolds loaded"
    assert len(searcher.boronic_acids) > 0, "No boronic acids loaded"

    # Verify all scaffolds have role_mask & 384 == 384
    for mol_id, smiles, role_mask in searcher.scaffolds:
        assert role_mask & 384 == 384, f"Scaffold {mol_id} has wrong role_mask: {role_mask}"

    # Verify all boronic acids have role_mask & 1024 == 1024
    for mol_id, smiles, role_mask in searcher.boronic_acids:
        assert role_mask & 1024 == 1024, f"Boronic acid {mol_id} has wrong role_mask: {role_mask}"


def test_prioritized_scaffolds():
    """Verify winning scaffold IDs are at the front of the list."""
    from elite_miner.searcher import CombinatorialSearcher

    searcher = CombinatorialSearcher(DB_PATH)
    searcher.load_rxn5_building_blocks()

    priority_ids = {192490, 192488, 192710, 192713}
    scaffold_ids = [mol_id for mol_id, _, _ in searcher.scaffolds]

    # All priority IDs should be present
    for pid in priority_ids:
        assert pid in scaffold_ids, f"Priority scaffold {pid} not found"

    # Priority IDs should appear before non-priority IDs
    first_non_priority_idx = None
    for i, mol_id in enumerate(scaffold_ids):
        if mol_id not in priority_ids:
            first_non_priority_idx = i
            break

    if first_non_priority_idx is not None:
        for pid in priority_ids:
            idx = scaffold_ids.index(pid)
            assert idx < first_non_priority_idx, (
                f"Priority scaffold {pid} at index {idx} is after "
                f"non-priority scaffold at index {first_non_priority_idx}"
            )


def _make_searcher():
    """Helper to create and initialize a CombinatorialSearcher."""
    from elite_miner.searcher import CombinatorialSearcher

    searcher = CombinatorialSearcher(DB_PATH)
    searcher.load_rxn5_building_blocks()
    return searcher


def test_generate_batch():
    """Verify batch returns correct format (rxn:5:... names, non-empty SMILES)."""
    searcher = _make_searcher()
    batch = searcher.generate_batch(batch_size=10)

    assert len(batch) > 0, "Batch should not be empty"
    for mol_name, smiles in batch:
        assert mol_name.startswith("rxn:5:"), f"Unexpected mol_name format: {mol_name}"
        parts = mol_name.split(":")
        assert len(parts) == 5, f"mol_name should have 5 colon-separated parts: {mol_name}"
        # All ID parts should be integers
        for p in parts[1:]:
            int(p)  # raises ValueError if not an int
        assert isinstance(smiles, str) and len(smiles) > 0, f"SMILES should be a non-empty string: {smiles}"


def test_generate_batch_no_duplicates():
    """Verify no duplicate mol_names within a single batch."""
    searcher = _make_searcher()
    batch = searcher.generate_batch(batch_size=50)

    names = [mol_name for mol_name, _ in batch]
    assert len(names) == len(set(names)), "Batch contains duplicate mol_names"


def test_generate_batch_skips_already_seen():
    """Verify second batch has no overlap with first batch."""
    searcher = _make_searcher()
    batch1 = searcher.generate_batch(batch_size=10)
    batch2 = searcher.generate_batch(batch_size=10)

    names1 = {mol_name for mol_name, _ in batch1}
    names2 = {mol_name for mol_name, _ in batch2}

    overlap = names1 & names2
    assert len(overlap) == 0, f"Second batch overlaps with first: {overlap}"


def test_end_to_end_pipeline():
    """Full pipeline: generate -> filter -> mock score -> rank."""
    from elite_miner.searcher import CombinatorialSearcher
    from elite_miner.filters import ValidityFilter
    from elite_miner.scorer import ProxyScorer
    from rdkit import Chem
    import random

    searcher = CombinatorialSearcher(DB_PATH)
    searcher.load_rxn5_building_blocks()

    vfilter = ValidityFilter(min_heavy_atoms=10, min_rotatable_bonds=1,
                             max_rotatable_bonds=10, banned_atoms=["Se"])

    # Generate
    raw = searcher.generate_batch(batch_size=50)
    assert len(raw) > 0

    # Filter
    valid = vfilter.filter_batch(raw)
    assert len(valid) > 0

    # Mock scoring (PSICHIC needs GPU)
    scored = []
    for name, smiles in valid:
        mol = Chem.MolFromSmiles(smiles)
        ha = mol.GetNumHeavyAtoms() if mol else 0
        scored.append((name, smiles, random.uniform(0.3, 0.9), random.uniform(0.1, 0.5), ha))

    # Rank
    ranked = ProxyScorer.rank_with_antitarget(scored, antitarget_weight=0.9)
    assert len(ranked) == len(scored)

    # Verify descending proxy score
    for i in range(len(ranked) - 1):
        assert ranked[i][5] >= ranked[i+1][5]

    assert ranked[0][0].startswith("rxn:5:")
