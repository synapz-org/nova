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
                # §WWWWWWW: COALESCE boltz_le_std to 0.0 (no variance penalty for old rows).
                # §XXXXXXXX: COALESCE boltz_ww_std to 0.0 (no inter-seed penalty for rows
                # that §WW never re-scored or that pre-date §XXXXXXXX).
                # §FFFFFFFFFF: COALESCE confidence_score to 1.0 for pre-§FFFFFFFFFF rows
                # (no additional penalty when the column doesn't exist yet).
                "SELECT smiles, score, COALESCE(ligand_iptm, 1.0), "
                "COALESCE(boltz_le_std, 0.0), COALESCE(boltz_ww_std, 0.0), "
                "COALESCE(confidence_score, 1.0) "
                "FROM boltz_cache WHERE protein=?",
                (protein,),
            ).fetchall()
    except Exception:
        return None

    if len(rows) < min_points:
        return None

    # §DDDDDD: weight each training example by ligand_iptm.
    # §WWWWWWW: additionally divide by (1 + 10 × boltz_le_std) so molecules where
    # the 3 affinity diffusion samples disagreed strongly contribute less signal.
    # §XXXXXXXX: additionally divide by (1 + 10 × boltz_ww_std) so molecules where
    # the §WW cross-seed re-scoring showed high inter-seed variance (computed from
    # the std of [s_68, s_42, s_123]) contribute even less to surrogate training.
    # NULL ww_std → 0.0 → no additional penalty (pre-§XXXXXXXX or §WW never fired).
    # §FFFFFFFFFF: additionally multiply by COALESCE(confidence_score, 1.0) — low overall
    # complex confidence flags predictions where the Boltz structure is unreliable,
    # complementing ligand_iptm (global inter-chain alignment) with a holistic signal.
    X, y, weights = [], [], []
    for smiles, score, lig_iptm, le_std, ww_std, conf_score in rows:
        vec = _descriptor_vector(smiles)
        if vec is not None:
            X.append(vec)
            y.append(float(score))
            w = max(0.1, float(lig_iptm)) * max(0.1, float(conf_score)) / (
                (1.0 + 10.0 * float(le_std)) * (1.0 + 10.0 * float(ww_std))
            )
            weights.append(max(0.05, w))

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
                # §WWWWWWW: COALESCE boltz_le_std to 0.0 (no variance penalty for old rows).
                # §XXXXXXXX: COALESCE boltz_ww_std to 0.0 (no inter-seed penalty for rows
                # that §WW never re-scored or that pre-date §XXXXXXXX).
                # §FFFFFFFFFF: COALESCE confidence_score to 1.0 for pre-§FFFFFFFFFF rows.
                # §IIIIIIIIII: COALESCE psichic_le to 0.0 for legacy rows (neutral prior).
                "SELECT smiles, affinity_prob_binary, affinity_pred_val, "
                "COALESCE(ligand_iptm, 1.0), COALESCE(boltz_le_std, 0.0), "
                "COALESCE(boltz_ww_std, 0.0), COALESCE(confidence_score, 1.0), "
                "COALESCE(psichic_le, 0.0) "
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

    # §DDDDDD + §WWWWWWW + §XXXXXXXX + §FFFFFFFFFF: combined confidence-weighted training.
    # Weight = ligand_iptm × confidence_score / ((1 + 10×le_std) × (1 + 10×ww_std)):
    # - low-confidence poses (ligand_iptm): reduce training influence
    # - high intra-run sample variance (boltz_le_std): further reduce
    # - high inter-seed variance (boltz_ww_std): further reduce
    # - low overall complex confidence (confidence_score): additional reduction
    # §IIIIIIIIII: append psichic_le as feature 85 so the surrogate learns the
    # PSICHIC→Boltz correction.  NULL rows get 0.0 (neutral; StandardScaler centres).
    X, y_apb, y_apv, weights = [], [], [], []
    for smiles, apb, apv, lig_iptm, le_std, ww_std, conf_score, psichic_le in rows:
        vec = _descriptor_vector(smiles)
        if vec is not None:
            X.append(vec + [float(psichic_le)])  # §IIIIIIIIII: 85D total
            y_apb.append(float(apb))
            y_apv.append(float(apv))
            w = max(0.1, float(lig_iptm)) * max(0.1, float(conf_score)) / (
                (1.0 + 10.0 * float(le_std)) * (1.0 + 10.0 * float(ww_std))
            )
            weights.append(max(0.05, w))

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
    psichic_le_col: str = 'combined_score',
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

        # §IIIIIIIIII: append psichic_le as feature 85 to match fit_dual_surrogate training.
        placeholder = [0.0] * _N_FEATURES
        psichic_le_vals = (
            pool_df[psichic_le_col].fillna(0.0).values.astype(float)
            if psichic_le_col in pool_df.columns else np.zeros(len(pool_df))
        )
        X = [
            (v if v is not None else placeholder) + [float(pl)]
            for v, pl in zip(vecs, psichic_le_vals)
        ]

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
    psichic_le_col: str = 'combined_score',
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
        return dual_surrogate_rank_pool(pool_df, dual_model, smiles_col, ha_col, psichic_le_col)

    if not (_apb_rf and _apv_rf):
        return dual_surrogate_rank_pool(pool_df, dual_model, smiles_col, ha_col, psichic_le_col)

    try:
        vecs = [_descriptor_vector(s) for s in pool_df[smiles_col]]
        n_valid = sum(1 for v in vecs if v is not None)
        if n_valid < max(1, len(pool_df) * 0.5):
            return pool_df

        # §IIIIIIIIII: append psichic_le as feature 85 to match fit_dual_surrogate training.
        placeholder = [0.0] * _N_FEATURES
        psichic_le_vals = (
            pool_df[psichic_le_col].fillna(0.0).values.astype(float)
            if psichic_le_col in pool_df.columns else np.zeros(len(pool_df))
        )
        X = [
            (v if v is not None else placeholder) + [float(pl)]
            for v, pl in zip(vecs, psichic_le_vals)
        ]

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
        return dual_surrogate_rank_pool(pool_df, dual_model, smiles_col, ha_col, psichic_le_col)


def augment_pool_with_surrogate_blend(
    pool_df,
    dual_model,
    alpha: float = 0.6,
    smiles_col: str = 'product_smiles',
    psichic_col: str = 'combined_score',
    ha_col: str = 'heavy_atoms',
):
    """
    §HHHHHH: Augment pool_df with a surrogate-PSICHIC blended score column.

    Adds 'surrogate_salsa_score' = (1-alpha) * norm(psichic) + alpha * norm(surrogate)
    where norm() is min-max normalisation to [0, 1].  Both signals are placed on the
    same scale before blending so neither dominates due to magnitude differences.

    The surrogate component is (apb_pred - apv_pred) / ha using the dual RF models from
    §YYYYY — the same formula the validator uses to score Boltz-2 outputs.  Feeding this
    into SALSA's score_col makes hill-climbing converge toward molecules the miner has
    already learned correlate with high Boltz scores for the weekly target.

    Only activates when BOTH dual models are RandomForestRegressors (>=100 cache points,
    §QQQQ/§YYYYY): Ridge surrogates at low data density (<100 pts) generalize poorly
    across the large SAVI pool and would mislead SALSA more than help.

    Returns pool_df unchanged (without 'surrogate_salsa_score') when:
    - dual_model is None
    - either model uses Ridge rather than RF
    - descriptor computation fails for >50% of pool rows
    - any exception is raised
    """
    if dual_model is None:
        return pool_df

    try:
        from sklearn.ensemble import RandomForestRegressor
        model_apb, model_apv = dual_model
        if not (
            isinstance(model_apb.named_steps.get('model'), RandomForestRegressor)
            and isinstance(model_apv.named_steps.get('model'), RandomForestRegressor)
        ):
            return pool_df
    except Exception:
        return pool_df

    try:
        vecs = [_descriptor_vector(s) for s in pool_df[smiles_col]]
        n_valid = sum(1 for v in vecs if v is not None)
        if n_valid < max(1, len(pool_df) * 0.5):
            return pool_df

        # §IIIIIIIIII: append psichic_le as feature 85 to match fit_dual_surrogate training.
        placeholder = [0.0] * _N_FEATURES
        psichic_le_vals = (
            pool_df[psichic_col].fillna(0.0).values.astype(float)
            if psichic_col in pool_df.columns else np.zeros(len(pool_df))
        )
        X = [
            (v if v is not None else placeholder) + [float(pl)]
            for v, pl in zip(vecs, psichic_le_vals)
        ]

        apb_preds = model_apb.predict(X)
        apv_preds = model_apv.predict(X)

        if ha_col in pool_df.columns:
            ha_vals = pool_df[ha_col].fillna(25).values.astype(float)
        else:
            from utils.molecules import get_heavy_atom_count
            ha_vals = np.array([float(get_heavy_atom_count(s) or 25) for s in pool_df[smiles_col]])
        ha_vals = np.where(ha_vals > 0, ha_vals, 25.0)

        surrogate_raw = (apb_preds - apv_preds) / ha_vals

        def _minmax(arr):
            lo, hi = arr.min(), arr.max()
            span = hi - lo
            return (arr - lo) / span if span > 1e-9 else np.zeros_like(arr)

        psichic_vals = (
            pool_df[psichic_col].fillna(0.0).values.astype(float)
            if psichic_col in pool_df.columns
            else np.zeros(len(pool_df))
        )
        blended = (1.0 - alpha) * _minmax(psichic_vals) + alpha * _minmax(surrogate_raw)

        pool_copy = pool_df.copy()
        pool_copy['surrogate_salsa_score'] = blended
        return pool_copy
    except Exception:
        return pool_df


# ---------------------------------------------------------------------------
# §HHHHHHHHHH — Boltz-2 Embedding-Augmented Dual Surrogate
# ---------------------------------------------------------------------------
# Uses mean-pooled Boltz-2 evoformer trunk embeddings (384D ligand chain rows
# of the single representation `s`) stored per-molecule in the SQLite cache.
# PCA-reduces to 32D and concatenates with the existing 84D descriptor vector
# (84 + 32 = 116D features).  For pool molecules not yet Boltz-scored,
# the 32 PCA components are zero-padded (neutral prior = mean embedding).
# Only activates for RandomForest tier (≥ 100 cache points, ≥ 20 with embeddings).

_N_EMB_COMPONENTS = 32
_MIN_EMB_ROWS = 20


def _load_embeddings_from_cache(db_path: str, protein: str) -> tuple:
    """Load (smiles → 32D PCA vector) dict from SQLite, or (None, {}) on failure.

    Returns (fitted_pca, emb_dict) where emb_dict maps canonical SMILES to a
    float32 array of shape (_N_EMB_COMPONENTS,).  Returns (None, {}) when fewer
    than _MIN_EMB_ROWS embedding rows exist or sklearn/numpy are unavailable.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return None, {}

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT smiles, boltz_embedding FROM boltz_cache "
                "WHERE protein=? AND boltz_embedding IS NOT NULL",
                (protein,),
            ).fetchall()
    except Exception:
        return None, {}

    if len(rows) < _MIN_EMB_ROWS:
        return None, {}

    smiles_list, raw_embs = [], []
    for smiles, blob in rows:
        try:
            emb = np.frombuffer(blob, dtype=np.float32)
            if emb.shape == (384,):
                smiles_list.append(smiles)
                raw_embs.append(emb)
        except Exception:
            pass

    if len(smiles_list) < _MIN_EMB_ROWS:
        return None, {}

    try:
        X = np.stack(raw_embs)
        n_comp = min(_N_EMB_COMPONENTS, X.shape[0] - 1, X.shape[1])
        if n_comp < 1:
            return None, {}
        pca = PCA(n_components=n_comp, random_state=68)
        X_red = pca.fit_transform(X)
        if n_comp < _N_EMB_COMPONENTS:
            pad = np.zeros((X_red.shape[0], _N_EMB_COMPONENTS - n_comp), dtype=np.float32)
            X_red = np.hstack([X_red, pad])
        emb_dict = {s: X_red[i].astype(np.float32) for i, s in enumerate(smiles_list)}
        return pca, emb_dict
    except Exception:
        return None, {}


def fit_dual_surrogate_with_embeddings(db_path: str, protein: str, min_points: int = 40):
    """§HHHHHHHHHH: Dual APB/APV surrogate optionally augmented with PCA embeddings.

    Returns (model_apb, model_apv, emb_pca, emb_dict) where emb_pca and emb_dict
    are populated when ≥ _MIN_EMB_ROWS embedding rows exist; otherwise both are
    None/{} and the models use only the 84D descriptor features (identical to
    fit_dual_surrogate).  Returns None when fewer than min_points valid training
    examples exist or sklearn is unavailable.

    Callers pass the 4-tuple to dual_surrogate_ucb_rank_pool_emb() for ranking.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    emb_pca, emb_dict = _load_embeddings_from_cache(db_path, protein)
    has_emb = emb_pca is not None

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                # §IIIIIIIIII: COALESCE psichic_le to 0.0 for legacy rows.
                "SELECT smiles, affinity_prob_binary, affinity_pred_val, "
                "COALESCE(ligand_iptm, 1.0), COALESCE(boltz_le_std, 0.0), "
                "COALESCE(boltz_ww_std, 0.0), COALESCE(confidence_score, 1.0), "
                "COALESCE(psichic_le, 0.0) "
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

    # §IIIIIIIIII: feature order is [84D desc] + [psichic_le] + [32D PCA] = 117D.
    # psichic_le sits between descriptors and PCA so StandardScaler normalises it
    # alongside the other continuous features before the PCA block.
    zeros_emb = [0.0] * _N_EMB_COMPONENTS
    X, y_apb, y_apv, weights = [], [], [], []
    for smiles, apb, apv, lig_iptm, le_std, ww_std, conf_score, psichic_le in rows:
        vec = _descriptor_vector(smiles)
        if vec is None:
            continue
        vec = vec + [float(psichic_le)]  # §IIIIIIIIII: 85D descriptor+psichic block
        if has_emb:
            emb_feat = emb_dict.get(smiles)
            vec = vec + (list(emb_feat) if emb_feat is not None else zeros_emb)
        X.append(vec)
        y_apb.append(float(apb))
        y_apv.append(float(apv))
        w = max(0.1, float(lig_iptm)) * max(0.1, float(conf_score)) / (
            (1.0 + 10.0 * float(le_std)) * (1.0 + 10.0 * float(ww_std))
        )
        weights.append(max(0.05, w))

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
        return (model_apb, model_apv, emb_pca, emb_dict)
    except Exception:
        return None


def dual_surrogate_ucb_rank_pool_emb(
    pool_df,
    emb_result,
    beta: float = 1.0,
    gamma: float = 0.0,
    smiles_col: str = 'product_smiles',
    ha_col: str = 'heavy_atoms',
    psichic_le_col: str = 'combined_score',
):
    """§HHHHHHHHHH: UCB ranking using the embedding-augmented dual surrogate.

    emb_result is the (model_apb, model_apv, emb_pca, emb_dict) tuple from
    fit_dual_surrogate_with_embeddings().  Pool molecules not in emb_dict get
    zero-padded PCA components (neutral prior).  Falls back to plain mean ranking
    when RF variance is unavailable (Ridge tier or error).

    §KKKKKKKKKK: When gamma > 0 and embeddings are available, a cosine-distance
    bonus from the centroid of already-scored molecules is added to the UCB score,
    rewarding candidates that occupy unexplored binding-pose space.
    """
    if emb_result is None:
        return pool_df

    model_apb, model_apv, emb_pca, emb_dict = emb_result
    has_emb = emb_pca is not None

    try:
        from sklearn.ensemble import RandomForestRegressor
        _apb_rf = isinstance(model_apb.named_steps.get('model'), RandomForestRegressor)
        _apv_rf = isinstance(model_apv.named_steps.get('model'), RandomForestRegressor)
    except Exception:
        return pool_df

    try:
        smiles_list = pool_df[smiles_col].tolist()
        desc_vecs = [_descriptor_vector(s) for s in smiles_list]
        n_valid = sum(1 for v in desc_vecs if v is not None)
        if n_valid < max(1, len(pool_df) * 0.5):
            return pool_df

        # §IIIIIIIIII: feature order [84D] + [psichic_le] + [32D PCA] = 117D.
        zeros_desc = [0.0] * _N_FEATURES
        zeros_emb = [0.0] * _N_EMB_COMPONENTS
        psichic_le_vals = (
            pool_df[psichic_le_col].fillna(0.0).values.astype(float)
            if psichic_le_col in pool_df.columns else np.zeros(len(pool_df))
        )
        if has_emb:
            X = [
                (d if d is not None else zeros_desc) + [float(pl)] +
                list(emb_dict.get(s, np.zeros(_N_EMB_COMPONENTS, dtype=np.float32)))
                for d, s, pl in zip(desc_vecs, smiles_list, psichic_le_vals)
            ]
        else:
            X = [
                (d if d is not None else zeros_desc) + [float(pl)]
                for d, pl in zip(desc_vecs, psichic_le_vals)
            ]

        if ha_col in pool_df.columns:
            ha_vals = pool_df[ha_col].fillna(25).values.astype(float)
        else:
            from utils.molecules import get_heavy_atom_count
            ha_vals = np.array([float(get_heavy_atom_count(s) or 25) for s in smiles_list])
        ha_vals = np.where(ha_vals > 0, ha_vals, 25.0)

        if _apb_rf and _apv_rf:
            mean_apb, std_apb = predict_with_uncertainty(model_apb, X)
            mean_apv, std_apv = predict_with_uncertainty(model_apv, X)
            scores = (mean_apb - mean_apv + beta * (std_apb + std_apv)) / ha_vals
        else:
            apb_preds = model_apb.predict(X)
            apv_preds = model_apv.predict(X)
            scores = (apb_preds - apv_preds) / ha_vals

        # §KKKKKKKKKK: embedding centroid diversity bonus.
        # When gamma > 0 and the cache has scored molecules with stored embeddings,
        # compute the centroid of their PCA vectors and reward pool candidates whose
        # embedding is far (in cosine distance) from that centroid.  This steers the
        # next Boltz evaluation round toward unexplored protein–ligand interaction modes
        # beyond the scaffold family that hill-climbing has already converged to.
        # Only activates when ≥2 distinct scored embeddings exist; otherwise the
        # centroid is undefined and the bonus is silently skipped.
        if has_emb and emb_dict and gamma > 0.0:
            try:
                scored_embs = np.array(
                    [list(v) for v in emb_dict.values()], dtype=np.float32
                )
                if scored_embs.shape[0] >= 2:
                    centroid = scored_embs.mean(axis=0)
                    centroid_norm = float(np.linalg.norm(centroid))
                    if centroid_norm > 1e-9:
                        for i, s in enumerate(smiles_list):
                            mol_emb = emb_dict.get(s)
                            if mol_emb is not None:
                                mol_norm = float(np.linalg.norm(mol_emb))
                                if mol_norm > 1e-9:
                                    cos_sim = float(
                                        np.dot(mol_emb, centroid)
                                        / (mol_norm * centroid_norm)
                                    )
                                    scores[i] += gamma * (1.0 - cos_sim)
            except Exception:
                pass

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
