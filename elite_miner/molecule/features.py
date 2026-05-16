"""Molecule featurization for the Boltz2 surrogate.

Vector = ECFP4 (2048 bits) + curated RDKit descriptors (~25).

Per the Phase 2 design (architect review):
- ECFP4 captures local substructure (radius 2). 2048 bits is the standard size.
- RDKit descriptors add physicochemistry that ECFP misses (e.g. MW, logP, TPSA, QED).
- heavy_atoms is kept as a feature even though we also use it in label normalization —
  the predictor needs to know HA to model (pb - pv) which may have HA-dependent structure.

Deterministic per SMILES. Cached on disk by canonical SMILES hash so we don't
recompute across runs.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, Crippen, Lipinski, QED


ECFP_RADIUS = 2
ECFP_BITS = 2048


# Curated RDKit descriptors. Names map to numeric outputs.
RDKIT_DESCRIPTOR_NAMES = [
    "MolWt",
    "ExactMolWt",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHeteroatoms",
    "FractionCSP3",
    "TPSA",
    "MolLogP",
    "MolMR",
    "BalabanJ",
    "BertzCT",
    "LabuteASA",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "qed",
    "NHOHCount",
]


def _descriptor_value(mol: Chem.Mol, name: str) -> float:
    """Compute a single RDKit descriptor by name. Returns 0.0 on failure."""
    try:
        if name == "MolLogP":
            return float(Crippen.MolLogP(mol))
        if name == "MolMR":
            return float(Crippen.MolMR(mol))
        if name == "qed":
            return float(QED.qed(mol))
        fn = getattr(Descriptors, name, None) or getattr(Lipinski, name, None)
        if fn is None:
            return 0.0
        return float(fn(mol))
    except Exception:
        return 0.0


def _ecfp4(mol: Chem.Mol) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=ECFP_RADIUS, nBits=ECFP_BITS)
    arr = np.zeros((ECFP_BITS,), dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def _descriptors(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(
        [_descriptor_value(mol, name) for name in RDKIT_DESCRIPTOR_NAMES],
        dtype=np.float32,
    )


def mol_features(smiles: str) -> Optional[np.ndarray]:
    """Featurize a single SMILES.

    Returns float32 vector of shape (ECFP_BITS + len(RDKIT_DESCRIPTOR_NAMES),)
    = (2048 + 25,) = (2073,). Returns None for unparseable SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.concatenate([_ecfp4(mol), _descriptors(mol)])


def mol_features_batch(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    """Featurize many SMILES. Returns (feature_matrix, kept_indices).

    feature_matrix has shape (len(kept_indices), 2073).
    Indices that couldn't be parsed are dropped.
    """
    rows = []
    kept = []
    for i, smi in enumerate(smiles_list):
        v = mol_features(smi)
        if v is not None:
            rows.append(v)
            kept.append(i)
    if not rows:
        return np.zeros((0, ECFP_BITS + len(RDKIT_DESCRIPTOR_NAMES)), dtype=np.float32), []
    return np.stack(rows), kept


def feature_dim() -> int:
    return ECFP_BITS + len(RDKIT_DESCRIPTOR_NAMES)


def canonical_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def smiles_cache_key(smiles: str) -> str:
    """A short, deterministic key for caching features by canonical SMILES."""
    canon = canonical_smiles(smiles) or smiles
    return hashlib.sha256(canon.encode()).hexdigest()[:16]
