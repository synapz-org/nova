# bittensor `Config` returns None for missing attrs (not the getattr default)

## Symptom
Code that looks like `getattr(config, "some_flag", 0.7)` returns `None`, not `0.7`. Downstream code that compares against the value crashes with errors like:

```
TypeError: '<' not supported between instances of 'float' and 'NoneType'
```

…even though the apparent default would have been a float.

## Cause
Bittensor's `bt.config.Config` overrides attribute access so missing keys return `None` instead of raising `AttributeError`. Python's `getattr(obj, "key", default)` only falls back to `default` when `AttributeError` is raised — so the default never applies.

## Handling
Use a helper that maps `None` back to the default:

```python
def _cfg(cfg, key, default):
    v = getattr(cfg, key, default)
    return default if v is None else v
```

Fixed in `e3e22ec` — `_cfg()` in `elite_miner/run.py`.

This affects every `getattr(self.config, …)` or `getattr(config, …)` call in code that consumes a bt.config. Audit all of them when picking up new code.
