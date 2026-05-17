# Submissions GitHub repo must be public — validators fetch via raw.githubusercontent.com without auth

## Symptom
Miner is submitting per epoch (chain commits land, GitHub uploads succeed, file count in the repo grows), but:
- **0 of our submitted sequences appear in `Metanova/Submission-Archive`** over many hours
- On-chain `incentive` stays at 0
- No payout signal

The labeling worker confirms the surrogate is producing reasonable real iiptm values, so the issue isn't candidate quality.

## Cause
The validator fetches submission contents via `https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<filename>` (see `neurons/validator/commitments.py:144`) using only the validator's own `GITHUB_TOKEN` for auth. That token has access to **the validator's** repos, not the miner's. If the miner's submission repo is private, every `raw.githubusercontent.com` fetch returns 404 and the validator never decrypts or scores anything.

The miner-side upload (`utils/files.py:upload_file_to_github`) uses the miner's token via the GitHub API, which works fine against a private repo — so locally everything looks healthy. The break is in what the validator can see.

## Handling
The repo holding submission blobs **must be public**. Set it via:

```sh
curl -X PATCH -H "Authorization: token $TOKEN" \
  -d '{"private": false, "visibility": "public"}' \
  https://api.github.com/repos/<owner>/<repo>
```

After flipping visibility, the `raw.githubusercontent.com` CDN may still serve cached 404s for files uploaded while the repo was private. **Force a fresh commit** (any new file) to invalidate the CDN — without it, files in the listing return 200 via the API but 404 via raw.

Verification:
```sh
curl -sw "%{http_code}\n" \
  https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<filename>.txt \
  -o /dev/null
# should print 200
```

## Detection
A one-line health check: pick any submission filename from the GitHub API listing, fetch it unauthenticated via raw.githubusercontent.com. Any 404 means validators are blind.

## How it was caught
After hours of submissions with zero archive presence, dug into `commitments.py` to understand the validator's fetch path → noticed it uses raw.githubusercontent.com unauthenticated → checked our repo's `private` field via the API → it was True. The submission blobs were uploading successfully but were dark-pooled.
