"""Batch InChIKey uniqueness check against Metanova/Submission-Archive on HF.

Defers the import of utils.challenge until first call so importing this
module doesn't trigger utils/__init__.py's heavy transitive deps (metanano,
abnumber, anarci, etc).
"""

from __future__ import annotations

from typing import Optional

from rdkit import Chem


def _check_fn():
    from utils.challenge import entry_unique_for_protein_hf
    return entry_unique_for_protein_hf


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


def is_molecule_unique(target: str, smiles: str) -> bool:
    """True if InChIKey of `smiles` not seen for `target` in Submission-Archive."""
    return _check_fn()(target, smiles, entity_type="molecules")


def warm_archive_cache(target: str) -> None:
    """Prefetch the InChIKey set for `target` so first-call latency hits at startup."""
    _check_fn()(target, "", entity_type="molecules")


def filter_unique(target: str, candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return only candidates whose InChIKey is not in the archive.

    candidates: list of (mol_name, smiles).
    """
    warm_archive_cache(target)
    check = _check_fn()
    return [(n, s) for n, s in candidates if check(target, s, entity_type="molecules")]
