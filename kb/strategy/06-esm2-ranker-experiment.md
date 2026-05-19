# The $25 ESM2 ranker experiment — go/no-go gate for NOVA round-two

This is the experiment that decides whether NOVA round-two happens. Past-self specified the gate in `05-probability-of-success.md` (*"if high-band Spearman ≤ 0.55 after training, stop, don't deploy live mining, the assumptions are wrong"*). This doc operationalizes it as a single sequenced run that you can execute on one Basilica A100 box.

## The one-sentence summary

**Train the multi-head rank-sum surrogate on ESM2-650M embeddings of the 8129-row Q9NZQ7 archive, measure held-out high-band Spearman, and abort NOVA round-two if it doesn't exceed 0.55.**

## Why this is the right gate (not a different one)

From `04-empirical-proof-of-concept.md`:

- Baseline (33d seqstat → iiptm regression): high-band Spearman = **0.465**
- Hand-engineered 734d features → rank-sum: high-band Spearman = **0.324** (worse)
- ESM2 hypothesis: lift to **0.7+** based on related-task literature

If ESM2 doesn't break 0.55, the surrogate-and-pick strategy is wrong on this archive. Throwing more compute at it (more labels, more seeds, more BoltzGen passes) cannot recover from a low-Spearman-feature ceiling. The strategy must change shape (brute-force candidate generation without selection) or NOVA gets shelved.

If ESM2 hits ≥ 0.55 (modestly), the strategy is viable but marginal; we proceed but bound spend tightly.
If ESM2 hits ≥ 0.70 (strong), the strategy is healthy and we should expect to compete.

## Cost ceiling

Hard budget: **$30** of Basilica time. Walk away if the experiment isn't done by then.

Expected breakdown:
- ~6h × $1.05/h A100 = **$6.30** for ESM2 feature extraction
- ~1h × $1.05/h A100 = **$1.05** for training the head
- ~3h × $1.05/h A100 = **$3.15** of slack for debugging / re-runs
- **Total expected: ~$10**, ceiling $30.

If you hit the $30 ceiling without a result, stop. Something's wrong that needs fixing on a non-rented box.

## Prerequisites (verify before starting)

These are all already in place; this is a checklist, not work:

- [ ] `cache/archive_Q9NZQ7.parquet` exists (8129 rows). Verified: 1,260,134 bytes on disk.
- [ ] `elite_miner/scripts/extract_esm2_features.py` exists. Verified.
- [ ] `elite_miner/scripts/train_rank_surrogate_esm2.py` exists. Verified.
- [ ] `fair-esm==2.0.0` is pinned in `requirements/requirements.txt`. Verified.
- [ ] `install_deps.sh` installs torch + the requirements file. Verified.
- [ ] Coldkey has TAO (only matters if we proceed past the gate). Per `kb/post-mortems/2026-05-17`, coldkey `5C5CsVw...` had 0.373 τ remaining after the last attempt. **Confirm balance before spend** — irrelevant for the experiment itself but you'll want to know.

## Step-by-step runbook

### Step 1 — provision the box (5 min)

Rent **one A100 80GB** on Basilica. Not 4×, just one. We need ~12 GB GPU and ~6h of wall clock for ESM2 extraction — a single A100 is overkill but the cheapest unit available.

```bash
# After Basilica gives you the IP + ssh key:
ssh ubuntu@<ip>
git clone --recurse-submodules https://github.com/synapz-org/nova.git
cd nova
git checkout competitive-miner
```

The branch already has the two scripts. Last commit on the branch: `744d87d` ("strategy: PoC + ESM2-ready training pipeline").

### Step 2 — install deps (10-15 min, $0.30)

```bash
./install_deps.sh --cuda cu126
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: True NVIDIA A100-SXM4-80GB (or similar)
python -c "import esm; print(esm.__version__)"
# Expect: 2.0.0
```

Known gotcha (`kb/gotchas/install-deps-uv-not-in-path.md`): `uv` may not be on PATH on a fresh box. The install script handles this but if step 2 fails on "uv: command not found", `source ~/.local/bin/env` and re-run.

### Step 3 — pull the archive parquet (1 min, $0.02)

```bash
# The parquet should be committed at cache/archive_Q9NZQ7.parquet. Confirm:
ls -lh cache/archive_Q9NZQ7.parquet
# Expect: ~1.2 MB
python -c "import pandas as pd; df = pd.read_parquet('cache/archive_Q9NZQ7.parquet'); print(df.shape, list(df.columns))"
# Expect: (8129, ~13) with design_iiptm, design_ptm, ..., sequence
```

### Step 4 — ESM2 feature extraction (~6h, $6)

This is the long step. Run in tmux/screen so it survives an SSH drop.

```bash
tmux new -s esm2
python -m elite_miner.scripts.extract_esm2_features \
    --archive cache/archive_Q9NZQ7.parquet \
    --output cache/archive_Q9NZQ7_esm2.parquet \
    --batch-size 8 --device cuda \
    --model esm2_t33_650M_UR50D
# Detach with Ctrl-B D; reattach with `tmux attach -t esm2`
```

**Smoke-test alternative (5 min, free):** if you want to verify the script works before committing 6h, run with the small model first:

```bash
python -m elite_miner.scripts.extract_esm2_features \
    --archive cache/archive_Q9NZQ7.parquet \
    --output cache/archive_Q9NZQ7_esm2_smoke.parquet \
    --batch-size 16 --device cuda \
    --model esm2_t12_35M_UR50D
# 8129 sequences at small model ≈ 10-15 min. Confirms the pipeline runs end-to-end.
# If smoke succeeds: rm the smoke parquet, re-run with the real model.
```

Sanity-check after extraction:

```bash
python -c "
import pandas as pd, pickle, numpy as np
df = pd.read_parquet('cache/archive_Q9NZQ7_esm2.parquet')
print('rows:', len(df))
v = pickle.loads(df['esm2'].iloc[0])
print('embedding shape:', v.shape, 'dtype:', v.dtype, 'mean:', v.mean(), 'std:', v.std())
"
# Expect: rows ~8129, shape (1280,) (650M model emits 1280d), float32, finite stats
```

If rows < 8129: the extraction was interrupted. Re-run with `--resume`.

### Step 5 — train the ranker (~1h, $1)

```bash
mkdir -p models/nb_rank_esm2_v1
python -m elite_miner.scripts.train_rank_surrogate_esm2 \
    --archive cache/archive_Q9NZQ7.parquet \
    --esm2 cache/archive_Q9NZQ7_esm2.parquet \
    --output-dir models/nb_rank_esm2_v1 \
    --epochs 100 --batch-size 256 --device cuda \
    --include-seqstat
# --include-seqstat appends the 33d seqstat features; costs nothing, may help marginally.
```

The script prints validation metrics every epoch. The number we care about is **holdout high-band Spearman** — likely printed as `high_band_spearman` or similar near end-of-training.

### Step 6 — read the gate (5 min)

```bash
cat models/nb_rank_esm2_v1/metrics.json
# Expect a JSON with at least: spearman_overall, spearman_high_band, mae_high_band, ...
```

Decision:

| `high_band_spearman` | Verdict | Action |
|---|---|---|
| ≥ 0.70 | **STRONG GREEN.** Strategy is healthy. | Proceed to full Phase 0–3 deploy playbook. Budget $250-450. |
| 0.55–0.69 | **MARGINAL GREEN.** Strategy is viable but bounded. | Proceed but cap round-1 spend at $200. Re-evaluate after first reward signal. |
| 0.45–0.54 | **YELLOW.** Worse than baseline iiptm (0.465). | Don't deploy. The mod hurt. Investigate: did `--include-seqstat` help? Try without. Try a different layer (`--layer 30`)? At most one re-run. |
| < 0.45 | **RED.** Strategy is wrong. | Shelve NOVA. Update `kb/losses/` with the result. The surrogate-and-pick approach is structurally broken on this archive. |

### Step 7 — write the result to kb/ (10 min, $0)

Regardless of outcome, write the result file. **A claim without a measurement isn't a win.**

Template:

```markdown
# ESM2 ranker on Q9NZQ7 archive — n=8129, layer 33, seqstat concat

## Setup
- Date: 2026-05-?? 
- Box: Basilica A100 80GB
- ESM2 model: esm2_t33_650M_UR50D, last-layer 33 mean-pooled
- Features: 1280d ESM2 + 33d seqstat (if --include-seqstat) = 1313d
- Holdout: 15%, stratified by rank-sum quartile (n_holdout ≈ 1220)

## Results
| Metric | Value |
|---|---|
| Spearman (overall) | ? |
| Spearman (top-10% rank-sum) | ? ← THE GATE |
| MAE (top-10%) | ? |

## Verdict
[STRONG GREEN | MARGINAL GREEN | YELLOW | RED] per kb/strategy/06 gate table.

## Decision
[Proceed with deploy / Single re-run with [change] / Shelve NOVA].

## Next
[Specific next action based on verdict.]
```

Save to `kb/wins/esm2-ranker-q9nzq7-n8129.md` if it passes the gate, or `kb/losses/esm2-ranker-q9nzq7-n8129.md` if it doesn't.

### Step 8 — destroy the box (immediate)

Whatever the outcome:

```bash
# Pull cache/archive_Q9NZQ7_esm2.parquet and models/nb_rank_esm2_v1/ to local
scp -r ubuntu@<ip>:nova/cache/archive_Q9NZQ7_esm2.parquet ~/Projects/nova/cache/
scp -r ubuntu@<ip>:nova/models/nb_rank_esm2_v1 ~/Projects/nova/models/

# Then on the box:
sudo shutdown -h now
# In Basilica console: confirm the rental is stopped to avoid idle bill.
```

The ESM2 parquet is reusable across future experiments (same archive, same embeddings). Don't re-run extraction.

## Failure modes to watch

These are the predictable ways this experiment goes sideways. None block the gate; they just need workarounds.

| Failure | Cause | Fix |
|---|---|---|
| `esm: command not found` after install | fair-esm wheel didn't install | `uv pip install fair-esm==2.0.0` |
| OOM during extraction | batch_size too high for the box | drop `--batch-size` from 8 → 4 → 2 |
| Extraction takes >10h | unusually slow GPU or thermal throttling | check `nvidia-smi` between batches; restart |
| Training NaNs | LR too high for the data | drop `--lr` from 5e-4 to 2e-4 |
| Holdout Spearman wildly different from train | overfit | the script's regularization (dropout 0.2, weight_decay) should prevent this; if not, drop `--hidden` from 512 to 256 |
| Different layer choice would change verdict | Per literature, last layer is usually best for ranking; layer 30 sometimes wins | only worth trying as the single allowed re-run on YELLOW |

## What this does NOT do

- It does NOT register a hotkey, submit a commitment, or interact with the chain. Pure offline experiment.
- It does NOT update the public submissions repo. (When/if we deploy, that's a separate gate.)
- It does NOT spend money on multiple boxes. Single A100. Single run.

## Time budget

Worst case: rent box morning, run extraction through midday, run training over coffee, read result by ~6h elapsed. Best case: smoke test passes → real extraction overnight → result by morning of day 2.

If it's been 24h since you rented the box and you don't have a result, something's wrong. Stop, audit, restart with a clearer head.

## What to come back to me with

When the experiment is done, drop me one of these openers:

- *"Spearman 0.72, full green, let's plan round-7-on-SN68."*
- *"Spearman 0.58, marginal, what should we change before deploying?"*
- *"Spearman 0.41, dead. Move on to SN81/SN17 only."*

I'll have the next move queued for any of the three.
