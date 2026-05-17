# n=5000 fast-phase batch alone doesn't reach 0.83 — surrogate over-predicts in our neighborhood

## Change tested
Bumped nb fast-phase candidate pool from 20 → 5000, picking the surrogate's top-1 for chain submission. See `kb/wins/fast-phase-batch-size-5000.md` for the change itself.

## Measurement
First ground-truth label of an actual chain-submitted candidate (cache/submission_labels/Q9NZQ7.parquet, epoch 22724, block 8203576):

| metric                  | value   |
|-------------------------|---------|
| predicted iiptm (surrogate) | 0.8187  |
| real iiptm (BoltzGen)       | 0.7840  |
| bias (real - predicted)     | -0.0347 |

That puts our submission at archive percentile **87.6**, below 43/100 of recent additions. Decent middle-of-pack, **not** archive-top (which is 0.826). Predicted iiptm of 0.82+ does **not** mean real iiptm of 0.83+ in our sampled neighborhood.

## Why I had it wrong
Earlier benchmark on archive top-10 sequences showed surrogate **under**-predicting by ~0.02 (real 0.83 ↔ pred 0.81). I extrapolated that bias to our generator's output. The bias flipped sign: in our sampled neighborhood (1-3 mutations from top-200 seeds), the surrogate **over**-predicts by ~0.035. The high-band Spearman is only 0.60 (not 0.94 like the full archive), so per-sample accuracy is poor — the surrogate ranks well but doesn't measure absolute level accurately.

n=5000 still helped vs n=20 (we now reliably hit 0.78 real instead of an unknown lower number), but the absolute ceiling is set by the surrogate's local accuracy, not its global rho.

## When to retry
If we train a new surrogate that's specifically accurate in the high band (e.g. weight high-iiptm training examples 10x, or use a different head), this change could pay off more. Or if we move to a less-saturated neighborhood where competitors aren't sampling, the surrogate's bias might flip again.

n=5000 itself stays in — it gives top-1 from 5000 instead of from 20, which is still better. The disappointment is the *absolute level* of the resulting real iiptm.

## Commit
Original change: `6d6e169`. This measurement informed [[archive-seeded-strategy-likely-shared]] in notes.
