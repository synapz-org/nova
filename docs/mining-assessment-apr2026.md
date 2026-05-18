# SN68 Mining Assessment — April 2026

**Prepared by:** Hermes (researcher agent) for SYN-55.
**Date:** 2026-04-24.
**TL;DR:** Do not mine SN68 right now. The subnet's validator config is routing 100% of non-burn emissions to UID 54 via `emission_target_override_uid`, so unregistered submissions earn nothing regardless of molecular quality. Keep the elite_miner code warm, monitor for the override being removed, and prioritize other revenue streams until then.

---

## 1. What changed

The task brief assumed two SN68 competitions ("NOVA drug discovery" vs "Blueprint code optimization"). That framing is wrong. There is one subnet and one competition:

- `metanova-labs/nova` (the repo we forked as `synapz-org/nova`) is the legacy codebase. Upstream is only receiving operational updates — weekly target rotations, allowed-reaction tweaks, new MSA files. No architectural change since our last sync.
- `metanova-labs/nova-blueprint` is the **new validator + miner-sandbox architecture** for the same drug-discovery subnet. Miners submit signed `miner.py` code via SDK; validators run each submission in an isolated Docker sandbox with a fixed time budget and score `/output/result.json`.

Blueprint is not a separate competition — it is SN68's new execution surface. The "code optimization" framing does not match what Blueprint actually is.

## 2. Current emission reality (blocker)

From `nova-blueprint/config/config.yaml`:

```yaml
run:
  time_budget_sec: 900
  competition_interval_seconds: 86400          # 1 competition / day
  min_improvement_margin: 0.03
  min_improvement_decay_rate: 0.0285
  emission_target_override_uid: 54
  emission_target_override_share: 1.0          # 100% of non-burn emissions → UID 54
```

With `emission_target_override_share: 1.0`, validators are instructed to send 100% of non-burn emission weight to UID 54. Any submission from our hotkey would place on the scoreboard but earn zero TAO. The decaying `min_improvement_margin` (0.03 initial, ~2.85% relative decay per day) also means the winner is harder to dislodge per epoch — this is a soft lock on UID 54 for the duration of this config.

This must be verified against the currently-deployed validator config (config pulled via the auto-update / watchtower path), not just the git repo. But the checked-in default is unambiguous: today's design intent is a single-winner subsidy.

## 3. What our existing code is worth

Our `elite_miner/` (run.py, searcher.py, scorer.py, filters.py — ~1k LOC) was built against the legacy in-process miner loop. The Blueprint miner contract is much smaller:

- Single file: `miner.py` at repo root, executed as `python /workspace/miner.py`.
- Read `/workspace/input.json` (contains `config` + `challenge`), write `/output/result.json` incrementally.
- No network. Read-only root FS. Scratch on `/tmp`.
- Output format: `{"molecules": ["rxn:4:…", "rxn:5:…"]}` — reaction-formatted only.
- Combinatorial SQLite DB comes in read-only; must open with `mode=ro&immutable=1`.

The reference `neurons/miner/miner.py` in Blueprint is essentially our elite_miner minus the ownership: iterative sampling loop that keeps a top-N pool and rewrites `/output/result.json` each iteration. Our components map cleanly:

- `elite_miner/searcher.py` (combinatorial search) → replaces Blueprint's `random_sampler.run_sampler`.
- `elite_miner/scorer.py` (Boltz2 proxy ranking) → pre-filter before PSICHIC.
- `elite_miner/filters.py` (HA, RB, banned atoms, Boltz-safe) → same role.
- `elite_miner/run.py` main loop → adapt to the `iterative_sampling_loop` signature, remove validator imports, route scoring through `validator.scoring.score_molecules_json` which is already in the sandbox.

Estimated port effort: 1–2 days of focused work, mostly glue + path adjustments + DB open-mode fix, assuming the Boltz2 ProxyScorer weights fit under the sandbox image size and time budget. The 900s `time_budget_sec` is the real constraint — every second we spend on cold-start (ESM, PSICHIC, Boltz) is lost sampling time.

## 4. Recommendation

**Do not mine SN68 until the emission override is lifted.** Specifically:

1. **Hold off on a port.** Porting elite_miner to Blueprint's sandbox contract is cheap enough that we don't need to do it speculatively; we can do it in 1–2 days once emissions open up.
2. **Set up a cheap watcher.** Schedule a recurring check that either (a) polls the live Blueprint config / runtime for `emission_target_override_share`, or (b) pulls `nova-blueprint` upstream and alerts on config changes. When `override_share` drops below 1.0 or `override_uid` changes, wake us to execute the port.
3. **Keep the fork healthy but minimal.** `synapz-org/nova` is already merged with upstream. We don't need to keep syncing aggressively — the only weekly-updating artifacts that matter (target protein, allowed reactions, MSA files) are fetched at runtime by the sandbox input, not compiled into miner code.
4. **Do not split effort with Templar / TIG.** Those subnets have live emissions and our code is further along there. SN68 is a no-yield seat-warmer right now.

## 5. Open questions to resolve before any port

- Is the `emission_target_override_uid: 54` config actually live on validators today, or is it a staging / default that validators are overriding? Worth a btcli check of current weight distribution on netuid 68 and a cross-reference with Taostats.
- UID 54 — who owns it? If it's a MetaNova-operated baseline miner (subsidy during the Blueprint rollout), the override is temporary by design and we should be ready. If it's a persistent third-party preference, the subnet is effectively closed.
- What does the on-chain Yuma Consensus say vs what the config says? Validator weights are what actually mint emissions — the config is just the default scheduler behavior. A mismatch here is the real signal.

## 6. Followups to file

- Ticket: "SN68: verify emission override status via on-chain weights (btcli / Taostats)" — cheap research, clarifies section 5.
- Ticket: "SN68: port elite_miner to Blueprint sandbox contract" — blocked on the above; keep as backlog with clear entry criteria.
- Ticket (routine): weekly check of `nova-blueprint/config/config.yaml` and SN68 top-UID weight concentration.
