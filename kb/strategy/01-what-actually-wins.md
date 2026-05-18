# What actually wins on SN68 — re-derived from the archive

The validator scores submissions by **dense-ranking each of 10 BoltzGen metrics** then **summing the 10 ranks** (lower is better). Not by design_iiptm alone. This was hiding in plain sight in `external_tools/boltzgen/src/boltzgen/boltzgen_wrapper.py:229` but we built our whole strategy around iiptm.

## The numbers (Q9NZQ7 archive, n=8129)

| Metric                          | iiptm percentile | rank_sum percentile |
|---------------------------------|------------------|---------------------|
| median                          | 0.688            | 23,051              |
| top 5%                          | ≥ 0.80           | ≤ ~5,000            |
| top 1% (real winners)           | ≥ 0.82           | ≤ ~2,300            |
| top 0.16% (epoch champions)     | ≥ 0.82           | ≤ ~1,807            |

**Top-20-by-iiptm vs Top-20-by-rank_sum**: only **5 overlap**. 75% of "best iiptm" submissions are NOT in the top 20 by rank_sum. The real winners have iiptm 0.79–0.83, mean **0.81**, std **0.009** — narrowly clustered, but the rest of their metrics are uniformly strong.

## Which metrics actually decide it

Pearson correlation of `metric_rank` with `rank_sum` (top-1000 winners):

| Metric                       | r vs rank_sum | Notes |
|------------------------------|---------------|-------|
| **interaction_pae**          | **0.515**     | Strongest predictor of winning |
| **design_to_target_iptm**    | **0.449**     | Tight confidence on the binding interface |
| **delta_sasa_refolded**      | **0.437**     | Physical contact area |
| design_ptm                   | 0.388         | Global confidence |
| min_design_to_target_pae     | 0.351         | Worst-case interface error |
| **design_iiptm**             | **0.289**     | Our former sole target — 5th most important |
| plip_saltbridge_refolded     | 0.170         | Minor at the top |
| plip_hbonds_refolded         | 0.086         | Almost irrelevant at the top |
| liability_score              | 0.071         | Filter, not differentiator |
| liability_num_violations     | 0.057         | Filter, not differentiator |

The big four — **interaction_pae, design_to_target_iptm, delta_sasa_refolded, design_ptm** — collectively dominate. These are interface-quality signals from the *refold* and *folding* pipeline steps, not the *design* step we optimized for.

## Cross-correlation: can iiptm proxy for these?

| Pair                                      | r     |
|-------------------------------------------|-------|
| design_iiptm × design_ptm                 | 0.39  |
| design_iiptm × design_to_target_iptm      | 0.41  |
| design_iiptm × interaction_pae            | -0.47 |
| design_ptm × design_to_target_iptm        | **0.96** |
| design_ptm × interaction_pae              | **-0.92** |
| design_to_target_iptm × interaction_pae   | **-0.95** |

**iiptm correlates only weakly (r=0.39–0.47) with the more important metrics.** Meanwhile the four interface-confidence metrics are tightly mutually correlated (r=0.86–0.96) — they're effectively a single "interface confidence" signal that iiptm is largely *separate from*.

So a strategy that picks the top-1 by predicted iiptm will systematically miss sequences with strong interface confidence but mediocre iiptm — exactly where many real winners live.

## What this means for the strategy

1. **Train on rank_sum, not iiptm.** Or at minimum on a weighted combination. The current `combined_score_from_metrics` returns `-iiptm`; that's wrong for the validator's actual scoring.
2. **Refold-step metrics are decisive.** `delta_sasa_refolded`, `plip_hbonds_refolded`, `plip_saltbridge_refolded` come from the **folding + design_folding** steps, not design. A surrogate that predicts only confidence-step output cannot capture these.
3. **Liability is a floor, not a ceiling.** Don't waste features modeling it; just check it's not zero.

This finding alone is worth the 12-hour autopsy from yesterday. Every previous strategy decision was made under the wrong objective.
