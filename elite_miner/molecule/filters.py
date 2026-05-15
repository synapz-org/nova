"""Molecule validity filter — mirrors neurons/validator/molecule_validity.py.

Drops candidates the validator would reject before we spend Boltz2 budget on them.
Delegates the Boltz-safety check to utils.molecules.is_boltz_safe_smiles so it
stays in lockstep with the validator.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit.Chem import AllChem


def _is_boltz_safe_smiles(smiles: str) -> bool:
    """Mirror of utils.molecules.is_boltz_safe_smiles.

    Inlined to avoid importing utils.__init__ which transitively pulls in the
    metanano/abnumber/anarci stack (large heavy deps the miner doesn't need).
    """
    try:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = AllChem.AddHs(mol)
        canonical_order = AllChem.CanonicalRankAtoms(mol)
        for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
            atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
            if len(atom_name) > 4:
                return False
        return True
    except Exception:
        return False


class ValidityFilter:
    def __init__(
        self,
        min_heavy_atoms: int = 10,
        min_rotatable_bonds: int = 1,
        max_rotatable_bonds: int = 10,
        banned_atoms: list[str] | None = None,
    ):
        self.min_heavy_atoms = min_heavy_atoms
        self.min_rotatable_bonds = min_rotatable_bonds
        self.max_rotatable_bonds = max_rotatable_bonds
        self.banned_atoms = {a.upper() for a in (banned_atoms or [])}

    @classmethod
    def from_config(cls, cfg: dict) -> "ValidityFilter":
        """Build from subnet config (config.yaml molecule_requirements section)."""
        return cls(
            min_heavy_atoms=cfg.get("min_heavy_atoms", 10),
            min_rotatable_bonds=cfg.get("min_rotatable_bonds", 1),
            max_rotatable_bonds=cfg.get("max_rotatable_bonds", 10),
            banned_atoms=cfg.get("banned_atom_types", []),
        )

    def is_valid(self, smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        if mol.GetNumHeavyAtoms() < self.min_heavy_atoms:
            return False

        nrb = Descriptors.NumRotatableBonds(mol)
        if nrb < self.min_rotatable_bonds or nrb > self.max_rotatable_bonds:
            return False

        if self.banned_atoms:
            for atom in mol.GetAtoms():
                if atom.GetSymbol().upper() in self.banned_atoms:
                    return False

        return _is_boltz_safe_smiles(smiles)

    def filter_batch(self, candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
        return [(n, s) for n, s in candidates if self.is_valid(s)]
