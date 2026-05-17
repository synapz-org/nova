# n=5000 fast-phase batch: real iiptm has high variance, sometimes archive-top

## Setup
Bumped nb fast-phase candidate pool from 20 → 5000, picking the surrogate's top-1 for chain submission. See `kb/wins/fast-phase-batch-size-5000.md` for the change itself. Submissions logged to `cache/submission_queue/nb.jsonl`; ground-truth labels collected by the BoltzGen worker into `cache/submission_labels/Q9NZQ7.parquet`.

## Measurement (n=2 so far, high variance)

Two ground-truth labels of actual chain-submitted candidates (cache/submission_labels/Q9NZQ7.parquet):

| label | predicted iiptm | real iiptm | bias    | archive percentile |
|-------|-----------------|------------|---------|--------------------|
| 1     | 0.8187          | 0.7840     | -0.0347 | 87.4               |
| 2     | 0.8202          | 0.8171     | -0.0031 | **99.7**           |
| mean  | 0.8194          | 0.8006     | -0.019  | —                  |

The samples are very similar by prediction (within 0.002) but land 4pp apart in real iiptm. The high-iiptm sample is at archive p99.7 — only 20 of 7822 archive entries above it, likely an epoch-winning result.

So this isn't cleanly a loss — n=5000 is producing competitive candidates **some** of the time, with high variance. The surrogate ranks well but absolute level is unreliable per sample.

## Why I had it wrong
Earlier benchmark on archive top-10 sequences showed surrogate **under**-predicting by ~0.02 (real 0.83 ↔ pred 0.81). I extrapolated that bias to our generator's output. The bias flipped sign: in our sampled neighborhood (1-3 mutations from top-200 seeds), the surrogate **over**-predicts by ~0.035. The high-band Spearman is only 0.60 (not 0.94 like the full archive), so per-sample accuracy is poor — the surrogate ranks well but doesn't measure absolute level accurately.

n=5000 still helped vs n=20 (we now reliably hit 0.78 real instead of an unknown lower number), but the absolute ceiling is set by the surrogate's local accuracy, not its global rho.

## When to retry
If we train a new surrogate that's specifically accurate in the high band (e.g. weight high-iiptm training examples 10x, or use a different head), this change could pay off more. Or if we move to a less-saturated neighborhood where competitors aren't sampling, the surrogate's bias might flip again.

n=5000 itself stays in — it gives top-1 from 5000 instead of from 20, which is still better. The disappointment is the *absolute level* of the resulting real iiptm.

## Commit
Original change: `6d6e169`. This measurement informed [[archive-seeded-strategy-likely-shared]] in notes.
