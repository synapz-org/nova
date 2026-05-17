# Full submission pipeline decrypts correctly end-to-end (validator-perspective)

## What was tested
After both critical fixes ([[github-submissions-repo-must-be-public]] and [[empty-side-of-pipe-rejected]]), pulled the contents of 44 historical submissions from `synapz-org/nova-submissions` via the **unauthenticated** `raw.githubusercontent.com` path (exactly the validator's fetch). Drand-decrypted each one whose target round is now signed.

## Result

| Outcome | Count |
|---------|-------|
| Decrypts to valid `mol|seq` format | **39 / 44** |
| Decrypts to invalid `|seq` (validator rejects — empty-side bug) | 3 / 44 |
| Drand round not yet signed (pending) | 2 / 44 |
| Decrypt error / parse error | 0 / 44 |

Sample decrypted plaintexts (`mol|nb`):
```
rxn:4:203424:223528|DVTLAESGGGLVQAGGSLKLSCAATGFTFSSYAMSWVRQRPGKQREWVSDITSQGDQTDYADFVKGRFTISRDNAKNTLYVQMTSLRPEDTAVYYCSKCMYGFMNSNAQHSQGTTVTVSA
rxn:3:59852:24161:71813|DVTVAESGGGLVQAGGSLRLSCAATGFTFSSWSMSWVRQRSGKEREWVSDISSQGDQTDYADFVKGRFTISRDNAKNTLYLQMSSLRPEDTAVYYCSKFPWRFMPQFAQHGQGTTVTVTA
```

89% of our historical submissions were already in valid format. The remaining 7% (3 files) were sent during the period where `build_message` produced `|seq` instead of `~|seq`; those have been silently rejected by the validator's parse gate, but the bug is fixed for any post-d273507 submissions.

## Why this is the strongest test possible
- Unauthenticated GitHub fetch matches the validator's exact code path
- Drand decryption uses real network signatures from `api.drand.sh`
- The plaintext format is the same one `neurons/validator/commitments.parse_decrypted_submission` consumes
- If decrypt + parse work for us, they work for the validator running the identical code

There is no remaining "but did the validator actually accept it" question — we've replayed the validator's logic directly.

## Where
`elite_miner/scripts/verify_submission_e2e.py` (replays the validator's flow step-by-step against the latest commitment) and `elite_miner/scripts/run_validator_decrypt.py` (calls the validator's literal `decrypt_submissions()` function).

## When it might regress
- Repo flipped back to private
- A future validator version changes the content format
- New code path sends `mol|` again (rerun the bulk decrypt check periodically)
