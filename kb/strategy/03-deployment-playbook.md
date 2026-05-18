# Deployment playbook — pre-flight, not pre-mortem

Yesterday's session lost ~6 hours and our UID to a problem documented on line 29 of `docs/MINER.md`. This playbook codifies the steps that would have prevented every operational failure of that run.

**Rule: a step is not optional just because it's annoying. If a step is in this playbook, the miner does not start until it's green.**

## Phase 0 — pre-deploy (≤ 30 min)

Before paying for any GPU or running any miner code:

1. **Read `docs/MINER.md` end to end.** Not skim. *Read.* The most important sentences are not the ones you'd guess.
2. **Read `docs/VALIDATOR.md`.** Knowing the validator's code path is the only way to write a verifier that catches gate failures.
3. **Read `kb/post-mortems/`.** All of it. Should take 10 minutes.
4. **Read `kb/gotchas/`.** All of it. Should take another 10.

If you skipped any of these, stop and go back. Yesterday's session would not have happened with this step done.

## Phase 1 — pre-flight verification (≤ 30 min)

Before starting the miner main loop, run these in order. **Each must exit 0 with all checks green.** If any fails, stop and fix it.

1. **Repo is public.**
   ```sh
   curl -sw "%{http_code}\n" \
     https://raw.githubusercontent.com/$GITHUB_REPO_OWNER/$GITHUB_REPO_NAME/$GITHUB_REPO_BRANCH/README.md \
     -o /dev/null
   # must print 200
   ```

2. **Hotkey is registered on netuid.**
   ```sh
   python3 -c "
   import bittensor as bt
   sub = bt.subtensor(network='finney')
   mg = sub.metagraph(netuid=68)
   import os; hk = open('/home/ubuntu/.bittensor/wallets/<wallet>/hotkeys/<hk>.txt').read()
   assert hk in mg.hotkeys, 'hotkey not on subnet — register first'
   print('uid:', mg.hotkeys.index(hk))
   "
   ```

3. **Coldkey has runway.** Balance ≥ 2 × `subtensor.recycle(netuid)`. Re-registration after pruning costs 1×. Below 2× and we're one prune cycle from being stuck without funds.

4. **`elite_miner/scripts/verify_submission_e2e.py`** on a recent commit. Steps 1–5 must all pass (drand-pending on step 6 is OK).

5. **`elite_miner/scripts/run_validator_decrypt.py`** with our hotkey + netuid. `push_timestamps` must populate (proves validator can read our GitHub).

If anything fails: do not start the miner. Fix it. Re-run all 5 checks.

## Phase 2 — health checks inside the main loop

The miner must self-detect:

1. **Deregistration**: every epoch boundary, re-fetch the metagraph and assert our hotkey is still in `mg.hotkeys`. If not, **stop, do not re-register automatically** (re-reg is a financial decision), but page the operator.

2. **Block-advance watchdog** (already in run.py at `30e88e8`). If `current_block` doesn't change for 300s, force-reconnect the subtensor.

3. **Diversity-collapse fallback** (already in run.py at `7e59785`). When using winner-neighborhood generators, do NOT fall back to proxy on collapse — that's by-design behavior, not a bug.

4. **Emission tracking**: every 10 epochs, query our UID's emission. If it's been zero for 20 epochs in a row, **something is wrong with our submissions or our pipeline**. Don't wait for deregistration to find out.

## Phase 3 — periodic out-of-band sanity checks

Once an hour (script run from a cron or a 1h ScheduleWakeup):

1. **Public file check**: fetch 1 of our recent GitHub uploads via raw URL, confirm 200.
2. **Validator-side decrypt**: `run_validator_decrypt.py` confirms a recent submission parses (modulo drand-pending).
3. **Archive crawl**: pull the latest `Q9NZQ7_nanobodies.csv` from HF and check how many of our recently-submitted sequences appear. **If zero across many hours, our submissions aren't being scored at all** — equivalent to deregistered or worse.

## What changes for the next run

| Yesterday | Today's playbook |
|-----------|------------------|
| Deployed without reading MINER.md | Phase 0 step 1 |
| Discovered private-repo bug after 4 hours | Phase 1 step 1 (10 seconds via curl) |
| Discovered empty-side bug after 5 hours | Phase 1 step 4 (verify_submission_e2e) |
| Discovered deregistration after 3.5h of wasted compute | Phase 2 step 1 (loop-internal check) |
| Discovered zero archive presence by manual inspection | Phase 3 step 3 (hourly automated check) |

All five caught at Phase 0 or 1 — *before* spending real money on Basilica.

## Cost discipline

- Don't spin up box 2 before validating box 1 produces archive-visible submissions for ≥ 2 hours.
- Per-epoch budget: ≤ 25 min of BoltzGen oracle compute. If labeling pipeline outruns this, scale down topk before scaling up GPUs.
- Daily budget: ≤ $50 of Basilica spend until first archive-confirmed submission lands.
- Keep coldkey ≥ 1 τ at all times. Top up before deploying, not after running out.

## Re-deployment checklist (the one-pager)

When resuming after deregistration or any outage:

```
[ ] Read MINER.md again (it may have updated)
[ ] Hotkey registered on subnet?
[ ] Coldkey balance ≥ 2× recycle cost?
[ ] GitHub repo public? (raw URL returns 200)
[ ] verify_submission_e2e.py passes through step 5?
[ ] run_validator_decrypt.py populates push_timestamps?
[ ] Health checks wired into main loop?
[ ] Hourly out-of-band cron scheduled?
[ ] Cost alarms set?
```

Nine boxes. Fifteen minutes. If you skip these the next time, you deserve the post-mortem.
