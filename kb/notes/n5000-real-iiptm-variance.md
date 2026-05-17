# n=5000 fast-phase batch: real iiptm has high variance, sometimes archive-top

## Setup
Bumped nb fast-phase candidate pool from 20 → 5000, picking the surrogate's top-1 for chain submission. See `kb/wins/fast-phase-batch-size-5000.md` for the change itself. Submissions logged to `cache/submission_queue/nb.jsonl`; ground-truth labels collected by the BoltzGen worker into `cache/submission_labels/Q9NZQ7.parquet`.

## Measurement (n=3 so far, very high variance)

Three ground-truth labels of actual chain-submitted candidates (cache/submission_labels/Q9NZQ7.parquet):

| label | predicted iiptm | real iiptm | bias    | archive percentile |
|-------|-----------------|------------|---------|--------------------|
| 1     | 0.8187          | 0.7840     | -0.0347 | p87                |
| 2     | 0.8202          | 0.8171     | -0.0031 | **p99.7**          |
| 3     | 0.8227          | 0.7121     | -0.111  | ~p50               |
| mean  | 0.8205          | 0.7711     | -0.049  | —                  |

Predictions are clustered (0.819-0.823) but real iiptm spans 0.71 to 0.82 — over 10pp. Mean is ~p77 (below archive median for recent entries). The p99.7 result (label #2) was the outlier, not the norm.

So **n=5000 is a lottery ticket** in our current neighborhood: sometimes we draw a competitive sequence, sometimes middle-pack, sometimes well below. Surrogate's per-sample noise (~0.05 RMS in the high band) dominates the rank order. The Spearman ρ=0.60 in the high band shows on per-sample basis: ranking is right on average but absolute level is noisy.

## Why I had it wrong
Earlier benchmark on archive top-10 sequences showed surrogate **under**-predicting by ~0.02 (real 0.83 ↔ pred 0.81). I extrapolated that bias to our generator's output. The bias flipped sign: in our sampled neighborhood (1-3 mutations from top-200 seeds), the surrogate **over**-predicts by ~0.035. The high-band Spearman is only 0.60 (not 0.94 like the full archive), so per-sample accuracy is poor — the surrogate ranks well but doesn't measure absolute level accurately.

n=5000 still helped vs n=20 (we now reliably hit 0.78 real instead of an unknown lower number), but the absolute ceiling is set by the surrogate's local accuracy, not its global rho.

## When to retry
If we train a new surrogate that's specifically accurate in the high band (e.g. weight high-iiptm training examples 10x, or use a different head), this change could pay off more. Or if we move to a less-saturated neighborhood where competitors aren't sampling, the surrogate's bias might flip again.

n=5000 itself stays in — it gives top-1 from 5000 instead of from 20, which is still better. The disappointment is the *absolute level* of the resulting real iiptm.

## Commit
Original change: `6d6e169`. This measurement informed [[archive-seeded-strategy-likely-shared]] in notes.
