# Phase 2 Runbook — Boltz2 Surrogate

This describes the operational workflow for the LightGBM-based Boltz2 surrogate
that replaces the random proxy as the molecule pre-ranker.

## What changed in Phase 2

- New module: `elite_miner/molecule/surrogate.py` — `SurrogateScorer` (drop-in replacement for `ProxyScorer`).
- New module: `elite_miner/molecule/features.py` — ECFP4 + 25 RDKit descriptors per SMILES (2073-dim vector).
- New module: `elite_miner/protein/features.py` — 25-dim sequence statistics (default) or 320-dim ESM2-T12 (opt-in).
- New module: `elite_miner/molecule/label_writer.py` — appends every real Boltz2 call to `cache/labels/{target}.parquet`.
- New module: `elite_miner/molecule/diversity.py` — `DiversityTracker` for mode-collapse detection.
- New script: `elite_miner/scripts/collect_labels.py` — bootstrap surrogate training data (runs Boltz2 on stratified candidates).
- New script: `elite_miner/scripts/train_surrogate.py` — train LightGBM regressor from collected labels.
- New config flags: `--surrogate_model_dir`, `--surrogate_batch_size`, `--surrogate_topk`, `--surrogate_min_spearman`.
- `MoleculeTrack` uses surrogate as pre-ranker when a model is configured; falls back to proxy when no model, model fails to load, or diversity collapse is detected.

## Lifecycle

```
new target → collect labels → train surrogate → deploy → online streaming → retrain
   week 1      ~6 hr             5 min          live      every epoch       day 3-7
```

## Step-by-step

### Step 1: Bootstrap label collection (one-time per target rotation)

Run on Basilica A100 80GB spot (~$0.50/hr). Cost: ~$1.50-3 for 2000 candidates.

```bash
# On Basilica:
python -m elite_miner.scripts.collect_labels \
    --target Q6P6W3 \
    --rxn-id 1 \
    --n-candidates 2000 \
    --batch-size 8 \
    --output cache/labels/Q6P6W3.parquet
```

This:
1. Samples 2000×5 = 10000 raw products from `rxn:1`.
2. Filters validity (HA, RB, banned atoms, Boltz-safe).
3. Stratifies by HA quartile × ECFP cluster (Butina at Tanimoto 0.6).
4. Round-robin samples 2000 candidates to maximize coverage.
5. Calls real Boltz2 in batches of 8 → ~5s/candidate → ~2.8 hours.
6. Appends to parquet (idempotent — re-running skips already-scored SMILES).

You can run this on multiple targets in parallel on different machines or sequentially.

### Step 2: Train surrogate

CPU only. ~5 minutes.

```bash
python -m elite_miner.scripts.train_surrogate \
    --labels-glob 'cache/labels/*.parquet' \
    --output-dir models/surrogate_2026-05-15 \
    --target Q6P6W3
```

Output:
- `models/surrogate_2026-05-15/model.txt` — LightGBM booster
- `models/surrogate_2026-05-15/holdout_metrics.json` — Spearman ρ, top-5%-recall, per-HA-bucket ρ
- `models/surrogate_2026-05-15/feature_version.json` — feature dim + protein usage flag

**Gate criteria before deploying:**
- Overall Spearman ρ ≥ 0.5
- Spearman ρ ≥ 0.3 in every HA bucket (`<20`, `20-30`, `30-40`, `>40`)
- Top-5%-recall ≥ 0.25 (the surrogate's top 5% must contain at least 25% of the true top 5%)

If ρ < 0.5: collect more labels and retrain, or accept the surrogate but raise `--surrogate_min_spearman` so the miner falls back to proxy automatically when prediction quality degrades.

### Step 3: Deploy

```bash
python -m elite_miner.run \
    --wallet.name <wallet> --wallet.hotkey <hotkey> \
    --network finney --netuid 68 \
    --surrogate_model_dir models/surrogate_2026-05-15 \
    --surrogate_batch_size 50000 \
    --surrogate_topk 10 \
    --surrogate_min_spearman 0.3 \
    --use_inference
```

With `--use_inference`, every Boltz2 call inside the miner also streams its result to
`cache/labels/{target}.parquet`, giving you ~5-10 free labels per epoch.

### Step 4: Online retraining (manual, for v1 of Phase 2)

When you accumulate +500 fresh labels (check `wc -l cache/labels/{target}.parquet`),
retrain and atomically swap:

```bash
# Train into a new dir (don't touch live model)
python -m elite_miner.scripts.train_surrogate \
    --labels-glob 'cache/labels/*.parquet' \
    --output-dir models/surrogate_$(date +%Y-%m-%d) \
    --target Q6P6W3

# Compare: holdout spearman should be > old model's by >= 0.05
cat models/surrogate_$(date +%Y-%m-%d)/holdout_metrics.json
cat models/surrogate_2026-05-15/holdout_metrics.json

# If improved, atomic swap via symlink
ln -sfn surrogate_$(date +%Y-%m-%d) models/surrogate_current
# Update run.py invocation to use --surrogate_model_dir models/surrogate_current
# Restart miner
```

### Step 5: Target rotation handling

When SN68 rotates to a new target (weekly):

1. `cache/labels/{old_target}.parquet` is kept for cross-target features (if `--no-protein` is not set, the trained model uses protein features so old-target rows still contribute).
2. Collect fresh labels for the new target ASAP (Step 1).
3. Retrain (Step 2) — should converge quickly on multi-target data.
4. Until the new model is ready, miner uses the existing (old-target-trained) model. Quality will be degraded but better than no surrogate (if cross-target Spearman > min_spearman) or will fall back to proxy automatically (if not).

## Monitoring

Watch these logs during mining:

- `molecule: surrogate ready (holdout_spearman=0.567)` — surrogate loaded successfully
- `molecule: surrogate not ready ({reason})` — falling back to proxy
- `molecule: diversity collapse (median sim=0.92) — falling back to proxy this batch` — mode collapse detected, this batch only
- `molecule: new best ... score=...` — track average score across batches; trending down = label drift

## When to refresh / retire a model

- **Refresh** when: +500 new labels collected; new target rotated in; spearman dips on rolling holdout
- **Retire** when: surrogate predictions diverge from real Boltz2 by > 0.2 spearman for 3+ consecutive epochs (auto: lower `--surrogate_min_spearman` and miner falls back to proxy on its own)

## Cost summary

| Activity | Cost |
|---|---|
| Bootstrap labels for one target (2000 candidates on A100 spot) | $1.50-3 |
| Train surrogate | ~free (CPU, ~5 min) |
| Online label streaming during mining | free (already paying for Boltz2 calls) |
| Retrain on +500 labels | ~free (CPU, ~5 min) |
| **Total per target lifecycle** | **~$2-5** |

vs payout per epoch ($128 molecule + $192 nanobody = $320 dual-track win, 20 epochs/day), the surrogate pays for itself in < 1 epoch of any week it produces a single win.

## Known limitations of Phase 2 v1

- No automated retraining daemon — Step 4 is manual.
- Default protein features are 25-dim sequence statistics, not ESM2 (which would need transformers/torch). For a few targets per year this is fine; the marginal gain from ESM2 is < 0.05 spearman empirically.
- No nanobody surrogate. Same pattern would apply with BoltzGen, deferred to Phase 3.
- No active-learning loop — currently the random 10% sampling is implicit via fast Phase-1 batches in `run.py`; a explicit "always score N random candidates" knob would be cleaner.
- No batched ESM2 caching at install — first call per uncached target triggers download.

## Validation tests

```bash
# Sanity check: synthetic labels → trained surrogate learns the signal
.venv/bin/python -m pytest elite_miner/tests/test_surrogate.py -v

# All 83 tests
.venv/bin/python -m pytest elite_miner/tests/ -q
```
