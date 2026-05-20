# Blueprint diversity-matrix rule lands 2026-05-20 — strategic signal

Source: SN68 Discord #nova-sn68, urdof (Nova team), 2026-05-18T15:15 UTC, message id `1505951900565114991`. Confirmed live as of 2026-05-20T02:04 UTC (epoch 20593).

## The rule

Blueprint submissions (100-molecule sets) now reject if *any* molecule pair has ECFP4 / Morgan radius=2 Tanimoto >= 0.90. One offending pair rejects the entire set. Stacks on top of the existing 0.25 entropy threshold. Implementation on `metanova-labs/nova-blueprint` `diversity-matrix` branch, helper `find_too_similar_pairs` in `utils/molecules.py`.

## Why this is interesting for us (we don't run Blueprint)

We compete on the BoltzGen / rank_sum protein-binding track. Blueprint is a separate track in the same subnet. We are currently deregistered and gating round-two on the ESM2 surrogate experiment (`kb/strategy/06`).

The signal worth recording:

- The team is explicitly trying to reward **broader chemical exploration** over **near-duplicate scaffold tricks**. Quote: *"reward algorithms that can search broadly and return useful, distinct chemical ideas, more than many small variations around the same scaffold."*
- This is the same family of complaint the protein-binding archive shows: many high-scoring submissions cluster around a handful of seeds. If they apply the same diversity logic to BoltzGen submissions (no announcement of this, just pattern-matching), our `LabelSeededGenerator` and any archive-seeded strategy gets nerfed.
- Blueprint may be a less-crowded entry point if/when round-two doesn't pan out. The new rule favors broad search algorithms over fine-tuning the same scaffold, which is a different skill set than our current rank-sum surrogate pipeline.

## What to do with this

Nothing immediate. Note it so when ESM2 gate triggers (pass or fail) we have a real second option on the table instead of "shelve NOVA."

If ESM2 fails: Blueprint becomes a more serious candidate. Diversity-first generation (MaxMin / Butina on Morgan FP, scaffold-level diversification) maps cleanly onto the new rule.

If ESM2 passes: watch for similar rule changes on the protein-binding track. The team's design philosophy is now on record.
