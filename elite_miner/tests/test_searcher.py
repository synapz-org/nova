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
