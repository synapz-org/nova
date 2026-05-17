# kb/ — knowledge base

A working memory for this repo. Three categories, all greppable from one place.

```
kb/
├── gotchas/   # non-obvious bugs and constraints (see kb/gotchas/README.md)
├── wins/      # things we tried that measurably improved outcomes
├── losses/    # things we tried that didn't work, with the measurement
├── notes/     # observations / hypotheses too uncertain for wins or losses
└── raw/       # paper summaries and external research (input, not output)
```

## When to read kb/

**Before debugging.** Grep `kb/gotchas/` for the symptom. Five minutes of grepping beats five hours of rediscovery.

**Before tuning.** Grep `kb/wins/` and `kb/losses/` for the knob you're about to turn. Someone may have already measured what you're guessing about.

**Before proposing a strategy change.** Grep all of `kb/` for the relevant concept. The literature in `kb/raw/` often has prior art.

## When to write kb/

**A gotcha:** any non-obvious bug or constraint where the fix isn't grep-able from the code alone. Symptom → cause → handling, with a commit reference if applicable.

**A win:** any measured improvement to a metric we care about (iiptm, mol score, archive hit rate, epoch coverage). Include the measurement, not just the change.

**A loss:** any tested change that didn't move the metric, or moved it the wrong way. As valuable as wins — prevents redoing them.

Don't write kb entries for things already obvious from the code or git log. Symptoms-of-stale-cache, normal patterns, refactors. Save the file for things that took a while to figure out.

## Format

Each file has one job. Name it for the term you'd grep for at 2am. Lead with the symptom or finding (the thing you'd recognize), then the details. Keep it short — a kb that nobody reads is worse than no kb.

Link to commit SHAs where relevant — the why behind a change is often in the commit message, the how-we-found-it is in the kb.
