# Archive-seeded strategy is likely shared by at least one competitor

## Finding (not yet a win — strategic note)
Our nb submissions land 2-3 Hamming distance from existing archive entries with real iiptm 0.80-0.81. As we keep submitting from `ArchiveSeededGenerator` (top-200 seeds, 1-3 PSSM-weighted mutations), new archive entries continue appearing in that same neighborhood — but they aren't ours.

That means at least one other miner is using a very similar archive-seeded strategy, and they're getting into the archive with real iiptm around 0.81.

## Implication
The "easy" leverage in our current strategy may be saturated:
- We can't trivially beat them on the surrogate axis (we're already at predicted 0.82+ with n=5000, surrogate plateaus past that).
- We can't trivially beat them on the generator axis (same seeds, same PSSM, same Hamming neighborhood).

The next leverage is **finding a different neighborhood**: sequence regions where the surrogate predicts well *and* competitors aren't sampling. Candidate moves:
- Wider mutation count (e.g. 5-10 instead of 1-3), accepting more risk for less crowding
- A different seed set (top 200-500 vs top-50, or only seeds *not* hit by recent archive entries)
- A different generator entirely (PSSM-from-scratch consensus rather than seed-based)

## When to revisit
Once we have direct BoltzGen labels of our own submissions (the labeling worker is running), we'll know our actual real iiptm. If it's <0.83 we need to change neighborhood, not widen the batch.

## Data point
Archive at 7754 entries (2026-05-17 ~10:15 UTC). Our 4 queued submissions all 2-3 hamming from archive entries at real 0.80-0.81. None of our 4 hashes directly in archive (expected — likely too recent).
