# Post-mortem: deregistered on netuid 68 after a session of mostly-invisible submissions

**Date:** 2026-05-17
**Duration:** ~12 hours of active work
**Outcome:** Hotkey `5EnzmPAvpRaQ...` pruned from netuid 68. Lost UID 4 to another miner. ~$50 in Basilica compute spent, ~6 hours of validly-scored submissions (the rest invisible to validators). Zero emission earned.

## The one-sentence version

We spent hours debugging a submission pipeline that was failing for one reason explicitly stated on **line 29 of `docs/MINER.md`** — "Your github repo must be public" — and by the time we caught it and fixed it, we'd accumulated so much zero-emission history that another miner re-registered and took our UID before we could earn anything.

## What actually happened, in order

### Phase 1 — submitting into the void (hours 0–4)

Session started with the elite miner already deployed on a Basilica A100. The miner was producing chain commits per epoch, GitHub uploads succeeded, the code path appeared healthy. **The miner kept submitting; nothing was reaching validators.**

Multiple secondary issues were debugged during this phase — None-score crashes (`c4c6985`), bittensor `Config` None-semantics (`e3e22ec`), nb surrogate sign bug (`8d66c71`), nb refine blocking new-epoch submissions (`aa52265`), async subtensor SSL leak (`30e88e8`). All were real bugs, all worth fixing. **None of them were the actual reason we weren't scoring.**

### Phase 2 — finding the actual problem (hours 4–5)

Noticed that despite hundreds of submissions, zero of our sequences appeared in `Metanova/Submission-Archive`. Dug into the validator code at `neurons/validator/commitments.py:144`:

```python
content_url = f"https://raw.githubusercontent.com/{path}"
resp = requests.get(content_url, headers={**github_headers, ...})
```

The validator fetches via the public CDN, with only its own GitHub token in `github_headers`. That token cannot see private repos. Checked our repo's privacy via API: `private: True`. Flipped it to public (`c43367a`).

**Cost up to this point: ~4 hours of compute and many epochs of zero emission.**

### Phase 3 — a second invisible failure (hour 5)

Even after public-flip, audited the submission format against `parse_decrypted_submission` and found `build_message` was sending `mol|` (empty seq side) when nb had no candidate. Validator rejects this format — but we never saw the rejection because we never see the validator's logs. Fixed to use `~` per the doc (`d273507`).

This means **only the most recent ~6 hours of submissions had a chance of scoring.** The first ~4 hours were all rejected by either the private-repo fetch or the empty-side parse gate. Many of those went to slot pruning math.

### Phase 4 — verifying it actually worked (hour 6)

Built two verifiers:
- `verify_submission_e2e.py`: replays the validator's flow step-by-step
- `run_validator_decrypt.py`: calls the validator's literal `decrypt_submissions()` function

Bulk-decrypted 44 historical submissions and confirmed:
- 39/44 decrypt cleanly as valid `mol|nb` (the ones uploaded after both fixes)
- 3/44 had empty-side bug (the ones from before `d273507`)
- 2/44 still timelock-pending

So we proved the pipeline was correct, but only for the most recent third of our history.

### Phase 5 — labeling, retraining, A/B testing (hours 6–11)

Stood up a second Basilica A100 to collect real-iiptm labels offline (`d4bb906`). Hit several gotchas (uv not on PATH in install_deps, MIG mode enabled by default, missing lightgbm, BoltzGen tmp dir accumulation — that one took 4 fix iterations to get right).

Eventually had ~75 real labels. Built a label-seeded generator using sequences we'd measured at iiptm ≥ 0.80 as seeds, hoping to escape the "shared archive neighborhood" effect. First label came back at real iiptm 0.747 — surrogate over-predicted by 0.09, no improvement.

The high-band-weighted surrogate retrain (`eae9bf6`) showed reweighting doesn't help when the features themselves can't see the high-band signal. Spearman in iiptm ≥ 0.78 was 0.465 regardless of `weight_exp`.

### Phase 6 — the discovery (hour 12)

Querying the chain to debug something unrelated, noticed the commitment at UID 4 was attributed to `bitty-labs/novasubs11/...` — a totally different repo. Cross-checked: `mg.hotkeys.index(our_hotkey)` → not in metagraph. **We'd been deregistered.**

The miner had been happily submitting all evening to a UID it cached at startup, but that UID now belonged to someone else's hotkey. ~3.5 hours of compute since deregistration was wasted.

Stopped the miner. User decided not to re-register. Boxes terminated.

## Root causes

**1. We didn't read the docs before deploying.** `docs/MINER.md` line 29 explicitly says the repo must be public. We saw `GITHUB_TOKEN` in `example.env` and assumed it gated access — it gates the *miner's* upload, not the *validator's* read. Five seconds of reading the dedicated miner doc would have caught this.

**2. We had no end-to-end verification before going live.** The scripts I eventually wrote (`verify_submission_e2e.py`, `run_validator_decrypt.py`) should have been step 0 of deployment, not step N. If those had been run against our first submission, the private-repo failure would have shown up immediately as "raw.githubusercontent.com returned 404."

**3. We had no deregistration health check in the miner.** The miner caches `miner_uid` at startup and never re-validates whether `wallet.hotkey` is still in the metagraph. After deregistration the miner keeps running, keeps spending compute, keeps writing to chain — all under the wrong identity. A 4-line guard catches this:

```python
if wallet.hotkey.ss58_address not in metagraph.hotkeys:
    bt.logging.error("hotkey not in metagraph — deregistered. Stopping.")
    sys.exit(1)
```

**4. We never had enough validly-scored time to earn emission and avoid pruning.** The ~6 hours of valid submissions after both fixes weren't enough to climb out of the prune list, especially because our real iiptm distribution was mean 0.75 (middle-of-pack, not winning epochs).

## What we actually got right

- **kb/** is a real artifact. 22 entries (`kb/gotchas/*` 8, `kb/wins/*` 4, `kb/losses/*` 2, `kb/notes/*` 3, plus this post-mortem) documenting concrete traps with detection patterns and the exact code paths that fixed them.
- **The labeling pipeline works and is well-engineered.** 77 real-iiptm labels across both boxes by the end, label cadence stabilized at ~110s/each after the BoltzGen tmp-dir fix.
- **Several fixes are genuine improvements** that would benefit any future run: the watchdog (`30e88e8`), the build_message `~` fix (`d273507`), the surrogate sign fix (`8d66c71`), the `_cfg` helper for bittensor's None-semantics (`e3e22ec`), the diversity-guard skip for winner-neighborhood generators (`7e59785`).
- **We characterized the surrogate's limits honestly.** High-band Spearman is ~0.46, real iiptm has std ~0.04 around predicted top-1, mean ~0.75. Reweighting can't extract signal that isn't in the features. ESM2 embeddings or thousands more labels would be needed.

## What a re-run should do differently

1. **Read `docs/MINER.md` start to finish.** Not skim — read. Every line.
2. **Run `verify_submission_e2e.py` against the FIRST submission and confirm steps 1–5 pass before letting the miner run a second epoch.** Wait for the drand round to sign, then run `run_validator_decrypt.py` to confirm 7-8.
3. **Add the deregistration health check** in the main loop. 4 lines.
4. **Don't deploy a miner without an end-to-end test that uses the validator's literal code path.** Mine is `run_validator_decrypt.py`. It exists now. Use it.
5. **Track emission directly** as a per-epoch metric. If we're at zero emission for N consecutive epochs, that's the early warning before pruning.
6. **Get a higher quality surrogate before optimizing other axes.** Our retrain experiments showed `n=5000` batch and `labelseeded` and `wide/tight` mutations are all small effects swamped by surrogate noise. Better features (ESM2 ≈ half-day build) or 1000+ high-band labels (≈ 1-2 days of dual-box labeling) would shift the ceiling far more than any of those.

## The unpleasant counterfactual

If on hour 1 we had run the e2e verifier against a freshly-submitted commit, the `raw.githubusercontent.com 404` would have surfaced in two minutes. We'd have flipped the repo public, fixed `build_message`, and had **12 hours of validly-scored submissions** instead of 6. That's enough emission accumulation to likely survive the prune cycle. We might have earned something, or at least kept the UID for a future run.

The lesson isn't "Bittensor is hard" or "the surrogate is too noisy." Both are true but tertiary. The lesson is **verify the gate before testing what's behind it**.

## What's left

- All code, all kb entries, all post-mortem material pushed to `synapz-org/nova` branch `competitive-miner`.
- Coldkey `5C5CsVw...` has 0.373 τ remaining.
- Both Basilica boxes terminated. ~$50 spent total today.
- No active processes, no chain identity, no in-flight work.

If anyone (future me, or anyone else) picks this up, the kb has a ready-made checklist of traps. The single most valuable file is probably `kb/gotchas/github-submissions-repo-must-be-public.md` — the one that should have been read before deployment, not written after the autopsy.
