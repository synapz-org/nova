# iiptm² reweighting doesn't improve high-band Spearman — features are the limit

## Change tested
Trained a new nb LightGBM surrogate on archive (7982) + our offline labels (19) with sample weights `iiptm**2` to emphasize the high-iiptm band. Compared against the current `nb_surrogate_Q9NZQ7_archive` baseline.

## Measurement
Both models have the same architecture (LightGBM regressor) and same features (58-dim: amino-acid composition + structural counts + 25-dim seqstat protein embedding for Q9NZQ7).

| Model                           | Holdout Spearman | HIGH-BAND Spearman (iiptm ≥ 0.78) |
|---------------------------------|------------------|------------------------------------|
| baseline (uniform weights)      | 0.934            | **0.465**                          |
| hb_v1 (weight_exp=2.0)          | 0.934            | **0.465**                          |
| hb_v2 (weight_exp=4.0)          | 0.931            | **0.319** (worse — overfit)        |

Reweighting at exp=2 was a no-op. Pushing to exp=4 actively hurt high-band performance — the model started overfitting to the small high-band sample (sample weight max went from 1.7 → 2.3 with min 0.001, very skewed distribution).

## Why I think it didn't work
The surrogate's feature representation is **amino-acid composition + summary statistics + a 25-d protein embedding**. That's enough to discriminate "is this a vaguely-VHH-shaped sequence at all" (global Spearman 0.93) but not enough to tell two sequences that are both 1-3 mutations from a winning seed apart. The signal that separates a real-iiptm-0.82 sequence from a real-iiptm-0.71 sequence lives in residue-level interactions and 3D structure, not in summary statistics.

So the data itself doesn't have the discriminative signal at the high band; reweighting doesn't help when the features are blind.

## When to retry
- After swapping `seqstat` features for **ESM2 embeddings** (per-position 1280-d → mean-pooled to 1280-d). The richer representation might separate near-identical sequences. ~half-day build: download ESM2 weights, extract embeddings for all 8000 archive sequences, retrain.
- After collecting many more offline labels (~thousands) — even a poor model with lots of data can outperform a clever model with few. Box 2 produces ~700/day so we'd need ~3-4 days.

## Where
Scripts: `elite_miner/scripts/retrain_nb_surrogate_highband.py` (orchestrator), `elite_miner/scripts/train_nb_surrogate.py` (added `--weight-exp` and `--high-band-cutoff` flags). New models go to `models/nb_surrogate_Q9NZQ7_hb_v*/`.
