# n=5000 fast-phase batch: real iiptm has high variance, sometimes archive-top

## Setup
Bumped nb fast-phase candidate pool from 20 → 5000, picking the surrogate's top-1 for chain submission. See `kb/wins/fast-phase-batch-size-5000.md` for the change itself. Submissions logged to `cache/submission_queue/nb.jsonl`; ground-truth labels collected by the BoltzGen worker into `cache/submission_labels/Q9NZQ7.parquet`.

## Measurement (n=5)

Five ground-truth labels of actual chain-submitted candidates (cache/submission_labels/Q9NZQ7.parquet):

| label | predicted iiptm | real iiptm | bias    | archive percentile |
|-------|-----------------|------------|---------|--------------------|
| 1     | 0.8187          | 0.7840     | -0.0347 | p87                |
| 2     | 0.8202          | 0.8171     | -0.0031 | **p99.7**          |
| 3     | 0.8227          | 0.7121     | -0.111  | ~p15               |
| 4     | 0.8232          | 0.7514     | -0.072  | ~p55               |
| 5     | 0.8245          | 0.7037     | -0.121  | ~p10               |
| mean  | 0.8219          | 0.7537     | -0.068  | —                  |

Predictions are tightly clustered (0.819-0.825 — spread of 0.006). Real iiptm spans **0.70 to 0.82** — 12pp of noise on top of a 0.006pp signal. Only 1 of 5 (the p99.7 result) was archive-competitive. Mean is at archive p~40.

**n=5000 is not "sometimes competitive" — it's "real iiptm ≈ 0.75 ± 0.05 with occasional 0.82 outliers."** The surrogate's per-sample noise (~0.06 RMS in the high band) IS the bottleneck. Bigger n cannot fix this because the rank order across the batch is noisier than the differences between rank-1 and rank-5 sequences.

This aligns with the high-band Spearman ρ=0.60 measured against the archive. Picking top-1 from a 5000-sample batch is essentially picking a random sample from the surrogate's predicted-0.82 cluster, which on real BoltzGen has 0.06 std around its mean.

## Why I had it wrong
Earlier benchmark on archive top-10 sequences showed surrogate **under**-predicting by ~0.02 (real 0.83 ↔ pred 0.81). I extrapolated that bias to our generator's output. The bias flipped sign: in our sampled neighborhood (1-3 mutations from top-200 seeds), the surrogate **over**-predicts by ~0.035. The high-band Spearman is only 0.60 (not 0.94 like the full archive), so per-sample accuracy is poor — the surrogate ranks well but doesn't measure absolute level accurately.

n=5000 still helped vs n=20 (we now reliably hit 0.78 real instead of an unknown lower number), but the absolute ceiling is set by the surrogate's local accuracy, not its global rho.

## When to retry
If we train a new surrogate that's specifically accurate in the high band (e.g. weight high-iiptm training examples 10x, or use a different head), this change could pay off more. Or if we move to a less-saturated neighborhood where competitors aren't sampling, the surrogate's bias might flip again.

n=5000 itself stays in — it gives top-1 from 5000 instead of from 20, which is still better. The disappointment is the *absolute level* of the resulting real iiptm.

## Commit
Original change: `6d6e169`. This measurement informed [[archive-seeded-strategy-likely-shared]] in notes.
