# Diversity guard falls back to proxy on winner-neighborhood generators

## Symptom
Queue / labels show some submissions with `predicted_iiptm ≈ 0` (or in mol's case, `score ≈ 0`) interspersed with normal surrogate-ranked ones in the 0.82 range. Bittensor log shows:

```
nanobody: diversity collapse (median identity=1.000) — falling back to proxy this batch
molecule: diversity collapse (median sim=1.000) — falling back to proxy this batch
```

When the fallback fires, the submitted candidate is a random pick from the proxy scorer (length-penalty + uniform noise for nb, or random pb/pv components for mol). Real iiptm of these submissions is mediocre, dragging down our archive percentile.

## Cause
`NanobodyDiversityTracker` / mol `DiversityTracker` flags "mode collapse" when pairwise identity/similarity across recent submissions exceeds a threshold (0.95 for nb, 0.85 for mol). The original intent was to detect a degraded generator and fall back to proxy.

But `ArchiveSeededGenerator` and `WeightedCombinatorialSearcher` **intentionally** produce sequences/molecules in a winner-neighborhood — high pairwise identity is the design, not a bug. The diversity guard fires after ~10 submissions because they're all 1-3 mutations from the same seeds, and the response (fall back to proxy) is strictly worse than continuing to submit near-winners.

## Handling
`_pre_rank_scorer` now keeps using the surrogate when the active generator is a known winner-neighborhood one (`ArchiveSeededGenerator` for nb, `WeightedCombinatorialSearcher` for mol). The collapse-warning logs at INFO level instead of WARNING in that case.

If you swap to a different generator (e.g. `NanobodyGenerator` or `CombinatorialSearcher`), the original collapse-detect-and-proxy logic still applies.

## How it was caught
A few labeled submissions had `predicted_iiptm` of -0.0002 or near-zero, breaking the otherwise-clustered-at-0.82 pattern. Grep for "diversity collapse" in miner.log surfaced the cause immediately.

## Commit
[next commit] — `elite_miner/run.py` MoleculeTrack._pre_rank_scorer + NanobodyTrack._pre_rank_scorer.
