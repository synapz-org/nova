# Validator's actual code path accepts our submissions through step 5

## Change tested
Made the submissions repo public (`kb/gotchas/github-submissions-repo-must-be-public.md`) and fixed `build_message` to send `~` for missing tracks (`kb/gotchas/empty-side-of-pipe-rejected.md`).

## Measurement
Ran the validator's literal `decrypt_submissions()` function (from `neurons/validator/commitments.py`) against our latest on-chain commitment with **no GitHub auth** (matching what a real validator would have). Result:

- `push_timestamps[uid=4]` populated with the commit timestamp — proves the validator's `raw.githubusercontent.com` fetch succeeded
- `decrypt_submissions` warned: `Skipping UID 4: Too early to decrypt: target_round=28748479, current_round=28748249` — confirming literal_eval and structural parsing succeeded, only blocker is the timelock round
- No errors about content fetch, hash mismatch, or payload format

So we know — via the exact code a real validator runs — that:
1. ✓ Validator's unauthenticated raw fetch returns our content
2. ✓ The `sha256(content)[:20] == filename` gate passes
3. ✓ `literal_eval(content)` returns the (target_round, ciphertext) tuple
4. ✓ The decrypt-attempt sees the right ciphertext bytes
5. ⏸ Decrypt + parse are pending the drand round being signed (timelock working as intended)

## Where
`elite_miner/scripts/run_validator_decrypt.py` — re-run any time to confirm the pipeline is healthy from the validator's perspective.

For step-by-step diagnostics (which line of validator code rejected us), use `elite_miner/scripts/verify_submission_e2e.py`.

## When it might stop working
- Repo flipped back to private — `raw.githubusercontent.com` would 404
- GitHub CDN cache poisoning after privacy changes — flush with any new commit
- A future validator version that changes the fetch path or content format
- We start sending content that fails the hash check (e.g. character encoding changes)

This verifier should be re-run **after any change to the submission path** (encryption, upload, commit format) and as a periodic health check.
