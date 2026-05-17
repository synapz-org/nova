# BoltzGen real inference is too slow for per-epoch refine

## Symptom
Calling `BoltzGenScorer.score_batch(...)` (real BoltzGen, not the surrogate) inside an epoch loop blocks for ~75–165 minutes on a single A100 — far longer than an epoch (~72 min on netuid 68). The current epoch passes by, the next epoch's fast-phase never runs, and the miner only commits to chain once every several epochs.

## Cause
BoltzGen's full pipeline is design → folding → design_folding → analysis, each step producing ~80 samples per input sequence. On A100 each sample is ~12s (0.08 it/s). With topk=2 inputs the design step alone is ~30 min; full pipeline ~75–165 min. Reducing `topk` only goes so far — `topk=1` still takes ~25 min for the design step.

`asyncio.to_thread(scorer.score_batch, …)` is not cancellable (the underlying thread keeps running even if the awaiting coroutine is cancelled). BoltzGen spawns its own subprocess but doesn't expose the PID, so cleanly killing it requires modifying the wrapper.

## Handling
For per-epoch competition, **skip real BoltzGen and trust the surrogate**. The nb LightGBM surrogate on archive labels has Spearman ρ=0.94 globally and ~0.60 in the high band (0.82+) with systematic under-prediction of ~0.02 — accurate enough that surrogate top-1 from a 5000-candidate pool is competitive with archive winners.

Implemented as `--nb_disable_inference` in `aa52265`. Mol track keeps real Boltz2 inference because Boltz2 is much faster (~1–2 min per call).

If you need real BoltzGen labels for surrogate training, run them offline, **not** in the epoch loop.
