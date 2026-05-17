# Negating the nb surrogate prediction fixed candidate ranking

## Change
`NanobodySurrogateScorer.score_batch` was returning the raw positive `design_iiptm` prediction as `score`. The downstream ranker sorts ascending (lower-is-better), so we were selecting the candidates predicted to have the **lowest** iiptm. Fix: negate before storing as `score`.

## Measurement
Comparing predicted top-1 iiptm of a 20-sample ArchiveSeededGenerator batch on Q9NZQ7, before vs after the sign flip:

- Before fix: top-1 surrogate prediction ≈ 0.55 (lowest of batch — wrong end)
- After fix: top-1 surrogate prediction ≈ 0.80–0.82 (highest of batch)

Local refine labels from before the fix maxed at real iiptm 0.65 (consistent with submitting bottom-of-batch). Post-fix submissions haven't been verified end-to-end against the archive yet, but the rank order is correct.

## Where
`8d66c71` — `elite_miner/nanobody/surrogate.py` line ~139.

## When it might stop working
If the nb surrogate is ever retrained to predict `-design_iiptm` directly (matching the lower-is-better convention everywhere else), the negation here would double-negate. Whoever retrains it should also delete the negation in `score_batch`. Linked: [[nb-surrogate-sign-convention]] in gotchas.
