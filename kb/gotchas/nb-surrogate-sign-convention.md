# nb surrogate sign: predict positive iiptm, negate before ranking

## Symptom
nb miner is "working" — submissions land on chain — but real `design_iiptm` of our submissions is bottom-of-pack (e.g. max 0.65 when archive top is 0.83), despite the surrogate having Spearman ρ=0.94 vs the archive.

If you score the archive's top winners with the surrogate, the predictions look reasonable (real 0.83 ↔ pred 0.81). So the surrogate isn't broken — but ranking somehow picks bad candidates.

## Cause
Scoring-direction mismatch:

- `NanobodySurrogateScorer.score_batch` returns the model's positive `design_iiptm` prediction directly.
- `nanobody.scorer.combined_score_from_metrics()` (used by the real BoltzGen path) returns `-design_iiptm` to enforce **lower-is-better**.
- `nanobody.scorer.rank()` sorts **ascending** (lower first).

So when the pre-ranker is the surrogate, `nb_rank(pre_scored)[:topk]` picks the candidates predicted to have the **lowest** iiptm — exactly the wrong ones.

## Handling
Negate in the surrogate's `score_batch` so it matches the lower-is-better convention used by the rest of the pipeline:

```python
iiptm_pred = float(next(pred_iter))
out.append(ScoredNanobody(sequence=seq, score=-iiptm_pred,
                          raw={"surrogate_pred": iiptm_pred}))
```

Fixed in `8d66c71`. Keep the raw positive prediction in `.raw["surrogate_pred"]` for debugging — the negation is purely for the ranker.

Mol surrogate does *not* have this bug: mol predicts `numerator = pb - pv` (higher-is-better) and `mol_rank` sorts `reverse=True` (descending). Aligned.

**Lesson:** whenever a scorer feeds into a ranker, write down both conventions and double-check the signs.
