"""
§ZZ — Mini-Surrogate Boltz Predictor from Disk Cache

Fits a lightweight Ridge regression on ~20 RDKit molecular descriptors using
Boltz scores stored in the persistent disk cache.  When ≥ 40 data points are
available for the current weekly target (typically epoch 3+), the surrogate
re-ranks candidate molecules with a Boltz-calibrated signal, complementing the
general PSICHIC ranking.

Key design decisions:
- Ridge regression (alpha=1.0) prevents overfitting under sparse data.
- 20-feature descriptor vector covers MW, logP, TPSA, H-bond counts, ring
  counts, and stereo information — all known correlates of binding affinity.
- Falls back silently to PSICHIC ordering when the cache has < 40 entries.
- rank_pool_by_surrogate drops the temporary 'surrogate_score' column before
  returning so the pool schema remains unchanged.
"""

import sqlite3
import numpy as np
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors


def _descriptor_vector(smiles: str) -> Optional[list]:
    """
    Compute a 20-feature RDKit descriptor vector for surrogate fitting.
    Returns None if the molecule cannot be parsed or descriptors fail.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        vec = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            float(Descriptors.NumHDonors(mol)),
            float(Descriptors.NumHAcceptors(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            float(Descriptors.RingCount(mol)),
            float(Descriptors.NumAromaticRings(mol)),
            float(Descriptors.NumAliphaticRings(mol)),
            Descriptors.FractionCSP3(mol),
            float(Descriptors.NumHeteroatoms(mol)),
            float(Descriptors.HeavyAtomCount(mol)),
            float(Descriptors.NumSaturatedRings(mol)),
            float(Descriptors.NumAliphaticCarbocycles(mol)),
            float(Descriptors.NumAromaticCarbocycles(mol)),
            Descriptors.BertzCT(mol),
            Descriptors.MolMR(mol),
            Descriptors.LabuteASA(mol),
            float(Descriptors.NumStereocenters(mol)),
            float(Descriptors.NumUnspecifiedAtomStereoCenters(mol)),
        ]
        if any(v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) for v in vec):
            return None
        return vec
    except Exception:
        return None


def fit_surrogate(db_path: str, protein: str, min_points: int = 40):
    """
    Fit a Ridge regression surrogate on Boltz-2 scores from the disk cache.

    Reads all (smiles, score) pairs for *protein* from the SQLite cache and
    fits a StandardScaler→Ridge(alpha=1.0) pipeline on the 20-feature descriptor
    vectors.  The scaler normalises each descriptor to zero mean / unit variance
    before regularisation so that Ridge penalises all features equally regardless
    of their absolute range (e.g. MW 200-500 vs NumHDonors 0-5).  This is
    §CCC: StandardScaler pipeline.

    Returns the fitted pipeline, or None if:
    - sklearn is unavailable
    - fewer than *min_points* valid training examples exist in the cache
    - fitting itself raises an exception
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT smiles, score FROM boltz_cache WHERE protein=?",
                (protein,),
            ).fetchall()
    except Exception:
        return None

    if len(rows) < min_points:
        return None

    X, y = [], []
    for smiles, score in rows:
        vec = _descriptor_vector(smiles)
        if vec is not None:
            X.append(vec)
            y.append(float(score))

    if len(X) < min_points:
        return None

    try:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0)),
        ])
        model.fit(X, y)
        return model
    except Exception:
        return None


def rank_pool_by_surrogate(pool_df, model, smiles_col: str = 'product_smiles'):
    """
    Re-rank *pool_df* rows by surrogate model score predictions.

    Adds a temporary 'surrogate_score' column, sorts descending, drops the
    column, and resets the index.  Falls back to the original order when:
    - descriptor computation fails for > 50% of rows
    - the predict call raises an exception

    The returned DataFrame has the same columns as the input.
    """
    try:
        vecs = [_descriptor_vector(s) for s in pool_df[smiles_col]]
        n_valid = sum(1 for v in vecs if v is not None)
        if n_valid < max(1, len(pool_df) * 0.5):
            return pool_df

        placeholder = [0.0] * 20
        X = [v if v is not None else placeholder for v in vecs]
        scores = model.predict(X)

        pool_copy = pool_df.copy()
        pool_copy['surrogate_score'] = scores
        return (
            pool_copy
            .sort_values('surrogate_score', ascending=False)
            .drop(columns=['surrogate_score'])
            .reset_index(drop=True)
        )
    except Exception:
        return pool_df
