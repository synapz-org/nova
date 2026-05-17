# First labelseeded variant label: surrogate over-predicted by 0.091

## Setup
LabelSeededGenerator mutates from sequences we've labeled at real_iiptm ≥ 0.80 (instead of public archive top-200). The standalone test showed surrogate top-1 predicting 0.838 vs 0.825 for ArchiveSeededGenerator — a 0.013 uplift in predicted iiptm.

## First label result (n=1)
- Predicted iiptm: **0.8381**
- Real iiptm: **0.7469**
- Bias: real - pred = **-0.091**
- Archive percentile: ~p65 (middle-of-pack)

## What this might mean
- **Surrogate is even less calibrated on label-seeded sequences than archive-seeded ones.** Average bias for archive-seeded variants is ~-0.05; this one was -0.091. Possibly the mutations to OUR seeds push them off-distribution from the surrogate's training data.
- The label-seed pool is small (~13 sequences) compared to archive top-200, so seeds repeat and produce similar candidates. The +0.013 surrogate boost may be the surrogate's own bias toward sequences similar to our offline-labeled set, not a real iiptm improvement.

n=1 is not enough. With 4x weight in the sampler, labelseeded should accumulate more labels quickly. Re-evaluate after n≥5.

## Where
Commit `aebb8e7` (4x weight for labelseeded). Logged in queue entries with `variant_id == 'labelseeded_n5000_m13'`.
