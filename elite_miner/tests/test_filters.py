import pytest
from elite_miner.filters import ValidityFilter


class TestValidityFilter:
    def test_accepts_good_molecule(self):
        """Valid drug-like molecule with HA=22, RB=4 should pass."""
        vf = ValidityFilter()
        assert vf.is_valid("c1ccc(-c2cc(-c3ccoc3)ccc2CN2CCC2)cc1") is True

    def test_rejects_too_few_heavy_atoms(self):
        """CCO (ethanol) has only 3 heavy atoms, should be rejected."""
        vf = ValidityFilter()
        assert vf.is_valid("CCO") is False

    def test_rejects_selenium(self):
        """Selenophene should be rejected when Se is banned."""
        vf = ValidityFilter(min_heavy_atoms=1, min_rotatable_bonds=0, banned_atoms=["Se"])
        assert vf.is_valid("c1cc[se]c1") is False

    def test_rejects_bad_rotatable_bonds(self):
        """Benzene has 0 rotatable bonds, should be rejected with min=1."""
        vf = ValidityFilter(min_heavy_atoms=1)
        assert vf.is_valid("c1ccccc1") is False

    def test_filter_batch(self):
        """Mixed list should keep only valid molecules."""
        vf = ValidityFilter()
        molecules = [
            ("good", "c1ccc(-c2cc(-c3ccoc3)ccc2CN2CCC2)cc1"),
            ("too_small", "CCO"),
            ("also_good", "c1ccc(-c2cc(-c3ccoc3)ccc2CN2CCC2)cc1"),
        ]
        result = vf.filter_batch(molecules)
        assert len(result) == 2
        assert result[0][0] == "good"
        assert result[1][0] == "also_good"
