from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


class ValidityFilter:
    def __init__(
        self,
        min_heavy_atoms=10,
        min_rotatable_bonds=1,
        max_rotatable_bonds=10,
        banned_atoms=None,
    ):
        self.min_heavy_atoms = min_heavy_atoms
        self.min_rotatable_bonds = min_rotatable_bonds
        self.max_rotatable_bonds = max_rotatable_bonds
        self.banned_atoms = banned_atoms if banned_atoms is not None else []

    def is_valid(self, smiles: str) -> bool:
        """Returns True if molecule passes ALL validity checks."""
        # 1. RDKit can parse it
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # 2. Heavy atom count >= min_heavy_atoms
        if mol.GetNumHeavyAtoms() < self.min_heavy_atoms:
            return False

        # 3. Rotatable bonds in [min, max] range
        num_rotatable = Descriptors.NumRotatableBonds(mol)
        if num_rotatable < self.min_rotatable_bonds or num_rotatable > self.max_rotatable_bonds:
            return False

        # 4. No banned atom symbols
        if self.banned_atoms:
            banned_set = set(a.upper() for a in self.banned_atoms)
            for atom in mol.GetAtoms():
                if atom.GetSymbol().upper() in banned_set:
                    return False

        # 5. Boltz-safe: atom names must be <= 4 chars
        mol_h = AllChem.AddHs(mol)
        ranks = AllChem.CanonicalRankAtoms(mol_h)
        for atom, rank in zip(mol_h.GetAtoms(), ranks):
            atom_name = atom.GetSymbol().upper() + str(rank + 1)
            if len(atom_name) > 4:
                return False

        return True

    def filter_batch(self, molecules: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Filter list of (mol_name, smiles), keeping only valid ones."""
        return [(name, smi) for name, smi in molecules if self.is_valid(smi)]
