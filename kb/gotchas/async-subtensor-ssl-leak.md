# async_subtensor silently stops getting new blocks (SSL leak)

## Symptom
Miner is alive (process up, asyncio event loop idle in `select()`), but stops submitting per-epoch. No exception, no log message indicating a problem. The chain `last commit` block stops advancing relative to `current` block.

The log fills with hundreds of:
```
ResourceWarning: unclosed transport <asyncio.sslproto._SSLProtocolTransport object at 0x…>
```

Onset is typically ~30 min into a run.

## Cause
`bt.async_subtensor` creates a new SSL connection per request (or doesn't properly reuse) and silently fails to close them. Eventually new connections stop succeeding but no exception is raised — `subtensor.get_current_block()` returns a stale/cached value forever. The outer epoch loop sees no boundary change and never calls `run_epoch`.

## Handling
Block-advance watchdog: track when `current_block` last changed, force-reconnect if it stalls past a threshold (5 min is well past the ~12s block period).

```python
last_block_seen = -1
last_block_advance_at = time.monotonic()
STALL_S = 300

# inside main loop, after fetching epoch_state:
if epoch_state.current_block != last_block_seen:
    last_block_seen = epoch_state.current_block
    last_block_advance_at = time.monotonic()
elif time.monotonic() - last_block_advance_at > STALL_S:
    bt.logging.warning("watchdog: block stalled, reconnecting")
    try: await subtensor.close()
    except Exception: pass
    subtensor = await _open_subtensor(config.network)
    last_block_advance_at = time.monotonic()
    continue
```

Fixed in `30e88e8`. The `consecutive_errors` reconnect path doesn't help here because no exception ever fires — must check progress, not errors.
