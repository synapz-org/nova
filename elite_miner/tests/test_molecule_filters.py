"""Tests for molecule ValidityFilter against validator-equivalent checks."""

import pytest

from elite_miner.molecule.filters import ValidityFilter


def test_rejects_unparseable():
    f = ValidityFilter()
    assert not f.is_valid("not-a-smiles!")


def test_rejects_below_min_heavy_atoms():
    f = ValidityFilter(min_heavy_atoms=10)
    assert not f.is_valid("CCO")  # 3 heavy atoms


def test_rejects_banned_atoms():
    f = ValidityFilter(min_heavy_atoms=1, min_rotatable_bonds=0, max_rotatable_bonds=99, banned_atoms=["Se"])
    assert not f.is_valid("CC[Se]CC")  # contains Se


def test_rejects_too_few_rotatable_bonds():
    f = ValidityFilter(min_heavy_atoms=1, min_rotatable_bonds=5, max_rotatable_bonds=99)
    # Benzene has 0 rotatable bonds
    assert not f.is_valid("c1ccccc1")


def test_rejects_too_many_rotatable_bonds():
    f = ValidityFilter(min_heavy_atoms=1, min_rotatable_bonds=0, max_rotatable_bonds=2)
    # Long alkyl chain has many rotatable bonds
    assert not f.is_valid("CCCCCCCCCCCCC")


def test_accepts_reasonable_molecule():
    f = ValidityFilter.from_config({
        "min_heavy_atoms": 10,
        "min_rotatable_bonds": 1,
        "max_rotatable_bonds": 10,
        "banned_atom_types": ["Se", "Na", "Fe", "Zn"],
    })
    # Drug-like molecule: ibuprofen-ish
    assert f.is_valid("CC(C)Cc1ccc(C(C)C(=O)O)cc1")


def test_from_config_defaults():
    """Filter built from a partial config should use validator defaults for missing keys."""
    f = ValidityFilter.from_config({})
    # defaults: min_heavy_atoms=10, min_rotatable_bonds=1, max_rotatable_bonds=10, no banned
    assert not f.is_valid("C")  # heavy atom count fails
