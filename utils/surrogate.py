"""
§ZZ — Mini-Surrogate Boltz Predictor from Disk Cache

Fits a lightweight Ridge regression on ~84 features (20 physicochemical RDKit
descriptors + 64-bit Morgan fingerprint) using Boltz scores stored in the
persistent disk cache.  When ≥ 40 data points are available for the current
weekly target (typically epoch 3+), the surrogate re-ranks candidate molecules
with a Boltz-calibrated signal, complementing the general PSICHIC ranking.

Key design decisions:
- Ridge regression (alpha=1.0) prevents overfitting under sparse data.
- 20-feature physicochemical descriptor vector covers MW, logP, TPSA, H-bond
  counts, ring counts, and stereo information — known correlates of binding.
- §DDD: 64-bit folded Morgan fingerprint (radius=2) appended to the physicochemical
  vector.  The low bit-count reduces sparsity vs. standard 1024-bit FPs, keeping
  the feature:sample ratio manageable at 40–100 training points.  StandardScaler
  normalises each bit to zero-mean/unit-variance before Ridge regularisation, so
  the penalty is applied equally across physicochemical and structural features.
  This lets the surrogate learn scaffold-level patterns ("molecules with this
  ring system bind well") that physicochemical features alone cannot capture.
- Falls back silently to PSICHIC ordering when the cache has < 40 entries.
- rank_pool_by_surrogate drops the temporary 'surrogate_score' column before
  returning so the pool schema remains unchanged.
"""

import sqlite3
import numpy as np
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

_N_MORGAN_BITS = 64   # §DDD: folded FP; low count keeps feature/sample ratio sane
_N_PHYSCHEM    = 20
_N_FEATURES    = _N_PHYSCHEM + _N_MORGAN_BITS  # 84 total


def _descriptor_vector(smiles: str) -> Optional[list]:
    """
    Compute an 84-feature descriptor vector for surrogate fitting.

    Returns a list of 84 floats (20 physicochemical + 64 Morgan FP bits),
    or None if the molecule cannot be parsed or any physicochemical descriptor
    produces NaN/Inf.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        physchem = [
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
        if any(v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) for v in physchem):
            return None
        # §DDD: 64-bit Morgan fingerprint (radius=2)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=_N_MORGAN_BITS)
        return physchem + list(fp)
    except Exception:
        return None


def fit_surrogate(db_path: str, protein: str, min_points: int = 40):
    """
    Fit a Ridge regression surrogate on Boltz-2 scores from the disk cache.

    Reads all (smiles, score) pairs for *protein* from the SQLite cache and
    fits a StandardScaler→Ridge(alpha=1.0) pipeline on the 84-feature descriptor
    vectors (20 physicochemical + 64 Morgan FP bits).  The scaler normalises each
    feature to zero mean / unit variance so that Ridge penalises all features
    equally regardless of their absolute range.  This is §CCC (StandardScaler
    pipeline) combined with §DDD (Morgan fingerprint augmentation).

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

        placeholder = [0.0] * _N_FEATURES
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
