# build_message must send `~` for missing track, not empty string

## Symptom
Submissions chain-commit and upload to GitHub successfully, but they don't get scored or appear in `Metanova/Submission-Archive`. Looks identical to the private-repo failure ([[github-submissions-repo-must-be-public]]), but persists even with the repo public.

## Cause
Validator's `parse_decrypted_submission` (in `neurons/validator/commitments.py`) requires both sides of the `|` delimiter to be non-empty:

```python
mol_part, seq_part = decrypted.split("|", 1)
if not mol_part or not seq_part:
    bt.logging.warning(f"UID {uid}: Missing molecules or sequences section")
    return None
```

`docs/MINER.md` documents that `~` is the **null placeholder** for "opting out of this track" — a `mol|~` submission scores the molecule and counts the nanobody as a burn-skip. But our `build_message` was generating `mol|` (empty seq side) when nanobody had no candidate, which the validator rejects entirely.

So if either track ever had no best candidate (disabled track, no valid generations, etc.), the whole submission got dropped silently.

## Handling
Use `~` instead of empty string for the missing side:

```python
m = molecule_name or "~"
n = nanobody_sequence or "~"
return f"{m}|{n}"
```

Both sides being `~` is technically valid format but useless — never let that happen (our `_has_candidate` check upstream already prevents it: returns False if both sides are None).

## Detection
Decrypt one of our own submissions and check the format. If you see `mol|` or `|seq`, the gate is being hit. The validator logs `UID X: Missing molecules or sequences section` when rejecting, but that's in validator logs we don't see.

## Commit
[next] — `elite_miner/submission.py:build_message`.
