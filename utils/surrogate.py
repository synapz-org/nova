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


_RF_THRESHOLD = 100  # §QQQQ: switch to RandomForest above this many training points


def fit_surrogate(db_path: str, protein: str, min_points: int = 40):
    """
    Fit a surrogate model on Boltz-2 scores from the disk cache.

    Reads all (smiles, score) pairs for *protein* from the SQLite cache and
    fits a StandardScaler→model pipeline on the 84-feature descriptor vectors
    (20 physicochemical + 64 Morgan FP bits).

    §QQQQ — Adaptive model selection:
    - < 100 training points: Ridge(alpha=1.0).  Linear model, strong regularisation,
      suitable for the sparse early-epoch regime where non-linearities cannot be
      estimated reliably.
    - >= 100 training points: RandomForestRegressor(n_estimators=100).  Tree ensembles
      capture non-linear scaffold→Boltz-score relationships (ring system preferences,
      halogen placement, heteroatom patterns) that Ridge cannot express.  n_jobs=1
      avoids spawning extra processes inside the miner's async event loop.
      StandardScaler is included in the pipeline for interface consistency; it has no
      effect on RF predictions (trees are scale-invariant).

    Returns the fitted pipeline, or None if:
    - sklearn is unavailable
    - fewer than *min_points* valid training examples exist in the cache
    - fitting itself raises an exception
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                # §DDDDDD: COALESCE ligand_iptm to 1.0 for pre-§DDDDDD cache rows.
                "SELECT smiles, score, COALESCE(ligand_iptm, 1.0) FROM boltz_cache WHERE protein=?",
                (protein,),
            ).fetchall()
    except Exception:
        return None

    if len(rows) < min_points:
        return None

    # §DDDDDD: weight each training example by ligand_iptm — Boltz's confidence in
    # the binding pose.  Noisy runs (ligand_iptm < 0.25) contribute up to 4× less
    # than well-calibrated runs (ligand_iptm ≈ 1.0), reducing surrogate overfitting
    # to uncertain measurements.  NULL → 1.0 so pre-§DDDDDD cache rows are unaffected.
    X, y, weights = [], [], []
    for smiles, score, lig_iptm in rows:
        vec = _descriptor_vector(smiles)
        if vec is not None:
            X.append(vec)
            y.append(float(score))
            weights.append(max(0.1, float(lig_iptm)))

    if len(X) < min_points:
        return None

    try:
        if len(X) >= _RF_THRESHOLD:
            # §QQQQ: RandomForest for non-linear scaffold-score patterns.
            learner = RandomForestRegressor(
                n_estimators=100,
                max_features='sqrt',
                random_state=68,
                n_jobs=1,
            )
        else:
            learner = Ridge(alpha=1.0)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', learner),
        ])
        model.fit(X, y, model__sample_weight=np.array(weights))
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


def predict_with_uncertainty(model, X: list) -> tuple:
    """
    Return (mean_preds, std_preds) arrays.

    For RandomForestRegressor: std comes from variance across individual trees,
    a free uncertainty proxy with no extra calibration cost.
    For Ridge and other non-ensemble models: std is all-zeros, so UCB reduces
    to plain mean ranking (identical to rank_pool_by_surrogate).
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        preds = model.predict(X)
        return preds, np.zeros(len(preds))

    scaler = model.named_steps['scaler']
    learner = model.named_steps['model']
    X_arr = np.asarray(X, dtype=float)
    X_scaled = scaler.transform(X_arr)

    if isinstance(learner, RandomForestRegressor):
        # (n_trees, n_samples) — each tree.predict is a cheap lookup table call
        tree_preds = np.array([t.predict(X_scaled) for t in learner.estimators_])
        return tree_preds.mean(axis=0), tree_preds.std(axis=0)
    else:
        preds = learner.predict(X_scaled)
        return preds, np.zeros(len(preds))


def ucb_rank_pool(pool_df, model, beta: float = 1.0, smiles_col: str = 'product_smiles'):
    """
    §RRRR: Re-rank pool_df by UCB = surrogate_mean + beta * surrogate_std.

    When the surrogate is a RandomForestRegressor (≥100 cache points, §QQQQ),
    per-tree variance provides an uncertainty estimate at zero extra inference
    cost.  UCB(β=1.0) selects candidates that are either predicted-good OR
    underexplored, balancing exploitation with exploration.  Falls back to
    plain mean ranking (rank_pool_by_surrogate) for Ridge models and on any
    error so the call is always safe.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        _is_rf = isinstance(model.named_steps.get('model'), RandomForestRegressor)
    except Exception:
        _is_rf = False

    if not _is_rf:
        return rank_pool_by_surrogate(pool_df, model, smiles_col)

    try:
        vecs = [_descriptor_vector(s) for s in pool_df[smiles_col]]
        n_valid = sum(1 for v in vecs if v is not None)
        if n_valid < max(1, len(pool_df) * 0.5):
            return pool_df

        placeholder = [0.0] * _N_FEATURES
        X = [v if v is not None else placeholder for v in vecs]

        mean_preds, std_preds = predict_with_uncertainty(model, X)
        ucb_scores = mean_preds + beta * std_preds

        pool_copy = pool_df.copy()
        pool_copy['surrogate_score'] = ucb_scores
        return (
            pool_copy
            .sort_values('surrogate_score', ascending=False)
            .drop(columns=['surrogate_score'])
            .reset_index(drop=True)
        )
    except Exception:
        return rank_pool_by_surrogate(pool_df, model, smiles_col)


def fit_dual_surrogate(db_path: str, protein: str, min_points: int = 40):
    """
    §YYYYY: Fit separate surrogates for APB and APV components.

    Reads (smiles, affinity_prob_binary, affinity_pred_val) from the cache
    (populated after the §YYYYY component-caching update).  Trains one Ridge
    or RF model per component.  Returns (model_apb, model_apv) or None when
    fewer than *min_points* rows have complete component data.

    The combined surrogate score for a molecule is then:
        (apb_pred − apv_pred) / heavy_atom_count

    Separate models capture the distinct functional forms: APB is a soft
    probability calibrated by a classification head, while APV is a
    continuous free-energy estimate.  Predicting them independently avoids
    the model trying to learn a single linear combination of features that
    approximates a nonlinear product of two structurally-distinct outputs.

    Falls back to None on any error so callers can degrade to fit_surrogate.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                # §DDDDDD: COALESCE ligand_iptm to 1.0 for pre-§DDDDDD cache rows.
                "SELECT smiles, affinity_prob_binary, affinity_pred_val, COALESCE(ligand_iptm, 1.0) "
                "FROM boltz_cache "
                "WHERE protein=? "
                "  AND affinity_prob_binary IS NOT NULL "
                "  AND affinity_pred_val IS NOT NULL",
                (protein,),
            ).fetchall()
    except Exception:
        return None

    if len(rows) < min_points:
        return None

    # §DDDDDD: confidence-weighted training — same rationale as fit_surrogate.
    X, y_apb, y_apv, weights = [], [], [], []
    for smiles, apb, apv, lig_iptm in rows:
        vec = _descriptor_vector(smiles)
        if vec is not None:
            X.append(vec)
            y_apb.append(float(apb))
            y_apv.append(float(apv))
            weights.append(max(0.1, float(lig_iptm)))

    if len(X) < min_points:
        return None

    def _make_pipeline(n: int):
        if n >= _RF_THRESHOLD:
            learner = RandomForestRegressor(
                n_estimators=100, max_features='sqrt', random_state=68, n_jobs=1
            )
        else:
            learner = Ridge(alpha=1.0)
        return Pipeline([('scaler', StandardScaler()), ('model', learner)])

    try:
        _w = np.array(weights)
        model_apb = _make_pipeline(len(X))
        model_apb.fit(X, y_apb, model__sample_weight=_w)
        model_apv = _make_pipeline(len(X))
        model_apv.fit(X, y_apv, model__sample_weight=_w)
        return (model_apb, model_apv)
    except Exception:
        return None


def dual_surrogate_rank_pool(
    pool_df,
    dual_model,
    smiles_col: str = 'product_smiles',
    ha_col: str = 'heavy_atoms',
):
    """
    §YYYYY: Re-rank pool_df using the dual APB+APV surrogate.

    Predicts (apb - apv) / ha for each row using the two component models
    returned by fit_dual_surrogate.  Falls back gracefully when the dual model
    is None, when descriptor computation fails for > 50% of rows, or on any
    exception.  The returned DataFrame has the same schema as the input.
    """
    if dual_model is None:
        return pool_df

    model_apb, model_apv = dual_model
    try:
        vecs = [_descriptor_vector(s) for s in pool_df[smiles_col]]
        n_valid = sum(1 for v in vecs if v is not None)
        if n_valid < max(1, len(pool_df) * 0.5):
            return pool_df

        placeholder = [0.0] * _N_FEATURES
        X = [v if v is not None else placeholder for v in vecs]

        apb_preds = model_apb.predict(X)
        apv_preds = model_apv.predict(X)

        # HA from pool column when available; fall back to RDKit.
        if ha_col in pool_df.columns:
            ha_vals = pool_df[ha_col].fillna(25).values.astype(float)
        else:
            from utils.molecules import get_heavy_atom_count
            ha_vals = np.array([
                float(get_heavy_atom_count(s) or 25)
                for s in pool_df[smiles_col]
            ])
        ha_vals = np.where(ha_vals > 0, ha_vals, 25.0)

        combined = (apb_preds - apv_preds) / ha_vals

        pool_copy = pool_df.copy()
        pool_copy['surrogate_score'] = combined
        return (
            pool_copy
            .sort_values('surrogate_score', ascending=False)
            .drop(columns=['surrogate_score'])
            .reset_index(drop=True)
        )
    except Exception:
        return pool_df


def dual_surrogate_ucb_rank_pool(
    pool_df,
    dual_model,
    beta: float = 1.0,
    smiles_col: str = 'product_smiles',
    ha_col: str = 'heavy_atoms',
):
    """
    §AAAAAA: UCB acquisition on the dual APB+APV surrogate.

    When both component models are RandomForestRegressors, uses per-tree variance
    to compute an optimistic upper-confidence-bound score:

        UCB = (mean_apb + β·std_apb − mean_apv + β·std_apv) / ha
            = (mean_apb − mean_apv + β·(std_apb + std_apv)) / ha

    The derivation follows the standard UCB principle applied independently to
    each Boltz component: the optimistic estimate of APB is (mean + β·std) and
    the optimistic estimate of −APV (we want APV as negative as possible) is
    (−mean_apv + β·std_apv).  Summing them yields the formula above.

    This rewards both high-confidence good molecules (exploitation) and
    structurally novel ones where either APB or APV is uncertain (exploration),
    without needing a separate UCB pass on the combined score.

    Falls back to dual_surrogate_rank_pool (mean-only) when:
    - either model is Ridge rather than RF (no tree variance available)
    - descriptor computation fails for > 50% of rows
    - any exception is raised
    """
    if dual_model is None:
        return pool_df

    try:
        from sklearn.ensemble import RandomForestRegressor
        model_apb, model_apv = dual_model
        _apb_rf = isinstance(model_apb.named_steps.get('model'), RandomForestRegressor)
        _apv_rf = isinstance(model_apv.named_steps.get('model'), RandomForestRegressor)
    except Exception:
        return dual_surrogate_rank_pool(pool_df, dual_model, smiles_col, ha_col)

    if not (_apb_rf and _apv_rf):
        return dual_surrogate_rank_pool(pool_df, dual_model, smiles_col, ha_col)

    try:
        vecs = [_descriptor_vector(s) for s in pool_df[smiles_col]]
        n_valid = sum(1 for v in vecs if v is not None)
        if n_valid < max(1, len(pool_df) * 0.5):
            return pool_df

        placeholder = [0.0] * _N_FEATURES
        X = [v if v is not None else placeholder for v in vecs]

        mean_apb, std_apb = predict_with_uncertainty(model_apb, X)
        mean_apv, std_apv = predict_with_uncertainty(model_apv, X)

        if ha_col in pool_df.columns:
            ha_vals = pool_df[ha_col].fillna(25).values.astype(float)
        else:
            from utils.molecules import get_heavy_atom_count
            ha_vals = np.array([
                float(get_heavy_atom_count(s) or 25)
                for s in pool_df[smiles_col]
            ])
        ha_vals = np.where(ha_vals > 0, ha_vals, 25.0)

        ucb_scores = (mean_apb - mean_apv + beta * (std_apb + std_apv)) / ha_vals

        pool_copy = pool_df.copy()
        pool_copy['surrogate_score'] = ucb_scores
        return (
            pool_copy
            .sort_values('surrogate_score', ascending=False)
            .drop(columns=['surrogate_score'])
            .reset_index(drop=True)
        )
    except Exception:
        return dual_surrogate_rank_pool(pool_df, dual_model, smiles_col, ha_col)
