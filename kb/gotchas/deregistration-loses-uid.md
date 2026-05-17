# Hotkey deregistration: miner keeps submitting to a UID owned by someone else

## Symptom
The miner has been running and the local queue shows new chain commits, but querying the chain returns commitment data for a totally different repo (e.g., `bitty-labs/novasubs11/...` instead of our `synapz-org/nova-submissions/...`). The UID our miner submits to is still UID 4, but the hotkey at UID 4 in the metagraph is someone else's — they re-registered and took the slot.

Querying `metagraph.hotkeys` shows our hotkey is no longer in the list. We've been deregistered.

## Cause
When the network's pruning fires (typically based on inactivity / low score), low-emission UIDs get freed for new registrations. Our UID 4 was the slot we registered into at session start — once that slot pruned us and someone else paid the recycle to register, we lost it.

The miner reads `miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)` at startup and caches it. After deregistration, **the cached UID points at someone else's hotkey**. All commits and submissions go to chain under the wrong identity (the new owner can't decrypt our submissions because timelock binds UID, but the chain entry exists, just nobody's). Validators ignore us entirely.

The local labeling pipeline keeps running fine since it doesn't care about chain identity — it labels whatever the queue contains.

## Handling
1. **Stop the miner immediately** — every chain commit is wasted effort.
2. **Check balance** vs `subtensor.recycle(netuid=68)` to confirm we can re-register.
3. **Re-register** (`btcli subnet register` or programmatic equivalent). This takes a fresh UID — likely different from the old one.
4. **Restart the miner** with the new UID. The miner re-reads metagraph on startup, so once the new registration lands, restart works.

## Detection
Periodic health check in the miner loop:
```python
if wallet.hotkey.ss58_address not in metagraph.hotkeys:
    bt.logging.error("hotkey not in metagraph — deregistered. Stopping miner.")
    sys.exit(1)
```

We don't have this guard yet — added to the "build later" list.

## Cost / consequence
- Wasted compute: every fast-phase + mol-refine + drand encryption between deregistration and detection is lost
- Lost incentive: any epochs where we would have placed are zero
- Eventual financial: re-registration fee (~0.16 τ at netuid 68 today)
