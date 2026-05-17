# Working in this repo

## Read kb/ first

Before debugging, tuning, or proposing a strategy change, grep `kb/` for the relevant term. The kb is where we record things that took a while to figure out — both the bugs and the things that measurably worked.

- `kb/gotchas/` — non-obvious bugs and constraints. Read first when something doesn't behave like the code suggests.
- `kb/wins/` — changes that measurably improved an outcome. Read before tuning a knob someone may already have tuned.
- `kb/losses/` — changes we tried that didn't help. Read before running an experiment someone may have already run.
- `kb/raw/` — paper summaries and external research (input only — don't add notes here).

Five minutes of grepping beats five hours of rediscovery. See `kb/README.md` for the full format.

## Write kb/ second

When you finish work, ask:

- **Did something non-obvious bite me?** → new file in `kb/gotchas/`
- **Did a change measurably improve a metric?** → new file in `kb/wins/` (with the measurement, not just the change)
- **Did a change fail to move a metric, or move it the wrong way?** → new file in `kb/losses/`

A claim without a measurement isn't a win — don't write it down as one.

Don't write kb entries for things already obvious from the code, the git log, or an existing kb entry. The kb only helps if it stays signal-dense.

## Where running state lives

The miner runs on a Basilica A100 rental (id in conversation context). Live logs at `/home/ubuntu/miner.log` on the box (BoltzGen progress dominates — filter `grep -vE 'DEPRECATION|MSA|Step '`). Surrogate labels at `cache/{labels,nb_labels}/*.parquet`.

Chain commits are visible via `subtensor.query_module("Commitments", "CommitmentOf", params=[68, hotkey])`. Submission content is drand-encrypted; the corresponding GitHub uploads are at `github.com/synapz-org/nova-submissions`.
