# Bumping fast-phase nb candidate pool 20 → 5000 boosts predicted iiptm ~0.02

## Change
Fast-phase nb surrogate-only ranking previously sampled 20 candidates from `ArchiveSeededGenerator` per epoch and took the top-1. Bumped to 5000.

## Measurement
Benchmarked on A100 (PID 127253 era):

| n     | top-1 pred iiptm | top-10 avg pred | total wall time |
|-------|------------------|-----------------|-----------------|
|    20 |           0.803  |          0.792  |          <0.01s |
|   100 |           0.823  |          0.807  |           0.02s |
|   500 |           0.821  |          0.814  |           0.06s |
|  1000 |           0.818  |          0.815  |           0.12s |
|  5000 |           0.825  |          0.820  |           0.61s |

Top-1 plateaus past ~n=1000. n=5000 still <1s, well within fast-phase budget. Real iiptm (surrogate has Spearman ρ≈0.94 archive-wide, under-predicts by ~0.02 in the high band) should land in the 0.84+ range — above the archive's current max of 0.826.

## Where
`6d6e169` — `elite_miner/run.py` `fast_batch_size_nb` (default 5000), `elite_miner/config.py` CLI flag.

Mol equivalent (`fast_batch_size_mol`) defaulted to 500 because mol surrogate scoring is much slower (~3s/1000) and top-1 plateaus earlier.

## When it might stop working
- If the generator's effective output diversity drops (e.g. PSSM gets too peaked), bigger n stops adding novel candidates and the gain disappears.
- If a new surrogate has different scoring noise characteristics, re-benchmark — the plateau point may shift.
- If the validity / uniqueness filter rate spikes (more candidates rejected), 5000 generated may yield far fewer scored. Watch for that.
