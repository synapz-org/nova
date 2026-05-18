# The plan — a coherent strategy for SN68 nanobody mining

This is the *technical* plan. The operational pre-flight (don't deploy without verifier passing, deregistration guard, etc) lives in `03-deployment-playbook.md`.

## The one-sentence thesis

**Train a multi-head sequence-only ranker on the 8k-row Q9NZQ7 archive that predicts the validator's rank-sum directly, sample diverse candidates with batched BO in ESM2 space, re-rank the chosen batch with multi-seed BoltzGen + ABAG-Rank-style ensembling, then submit the best.**

Every word of that is informed by Phase-1 archive analysis (`01-what-actually-wins.md`) and the arxiv survey (`kb/raw/papers-2026-05-sn68-nanobody-survey.md`).

## Why this works (when our previous attempts didn't)

Yesterday we trained on the wrong objective (iiptm), used too-crude features (33d seqstat + 25d protein embedding), and relied on a single noisy ranking step. The cascade:

- **Wrong objective** → top-1 by predicted iiptm overlaps with top-1 by true rank-sum only 25% of the time
- **Wrong features** → high-band Spearman 0.46, can't separate near-identical sequences
- **Single noisy step** → ~$0.06 std real-iiptm noise per submission; one shot ≈ a coin flip

Each of those has a known fix in the literature. The plan stacks them.

---

## Component 1 — Multi-head ESM2-based ranker (the new surrogate)

**What it is**: ESM2-650M (frozen) embeds candidate sequence + Q9NZQ7 target. A shared transformer head produces a representation. **Ten output heads**, one per BoltzGen metric. Loss is **margin-ranking pairwise** (AbRank, MosPro), not MSE.

**Why**: Yesterday's seqstat features had 0.46 Spearman in the high band; literature says ESM2 + ranking loss gets to 0.7-0.8 on related tasks (pLDDT-Predictor 0.79, Twin Peaks dual-head). One backbone, ten heads keeps capacity efficient (Twin Peaks).

**Training data**: 8129 archive labels for Q9NZQ7 with all 10 metrics — already on HuggingFace, no oracle calls needed to train.

**Compute**: ~6 hours on one A100 for embedding extraction + 1 hour for fine-tuning the heads. Total: 1 GPU-day ≈ $25.

**Validation**: Held-out 10% of archive, compute rank-sum Spearman directly. Target: ≥ 0.7 in the top decile. Yesterday's iiptm-only surrogate was 0.46.

**Risk if it fails**: We have a working baseline (the existing nb_surrogate_archive). New model only gets deployed if it beats it on held-out rank-sum.

## Component 2 — Diverse candidate generation in ESM2 space

**What it is**: Sample 5000 candidates per epoch. Mix three sources:
- 60%: `ArchiveSeededGenerator` (mutate from top-200 archive, 1-3 PSSM-weighted mutations). Proven baseline.
- 30%: ESM2-embedding-space BO (paper 2509.04998) — GP on ESM features over our labels, EI acquisition decoded to nearest archive + local mutation. Smooths the discrete sequence landscape.
- 10%: GFlowNet-sampled (paper 2203.04115) — diverse high-reward designs to escape archive saturation. Cheap implementation: weighted random-walk over CDR positions guided by the new ranker.

**Why mix**: 60% baseline is risk-control (we know it produces 0.75-mean candidates). 30% BO targets sample efficiency. 10% GFlowNet hedges the "shared archive neighborhood" problem (`kb/notes/archive-seeded-strategy-likely-shared.md`).

**Why not 100% BO**: yesterday showed model exploitation can outrun model accuracy. Mixing with baseline is the bandit-arm-pulling literature's exploration-vs-exploitation answer.

## Component 3 — Batched acquisition with correlated uncertainties

**What it is**: From the 5000 candidates, the ranker produces predicted rank-sum. Naively picking top-K gives near-duplicates (they all live on the same ridge). Use Batched-BO-with-correlated-candidates (paper 2410.06333) to select a batch of K=5 that **maximizes probability the true optimum is in the batch**, accounting for correlation among their uncertainties.

**Why K=5**: The validator only sees the latest commit, so we submit one. But picking 5 candidates for the **next-component re-ranker** lets us hedge against ranker error. We pick which of the 5 to actually submit after seeing their multi-seed BoltzGen output.

**Cost**: Negligible — the ranker is sub-second per batch.

## Component 4 — Multi-seed BoltzGen + ABAG-Rank decoy re-ranker

**What it is**: For the K=5 selected, run BoltzGen with **3 different seeds each** = 15 inferences per epoch. Use the 3 decoys per candidate to compute robust mean metrics (variance-reduce), then **ABAG-Rank-style listwise rank** to pick the best of the 5 candidates.

**Why**: Single BoltzGen inference has ~0.04 std per metric. With 3 seeds we cut variance by √3, separating real winners from lucky picks. ABAG-Rank (2026 preprint) shows listwise ranking over decoys beats both averaging and argmax.

**Cost**: 15 BoltzGen runs per epoch × ~100s each on A100 = 25 min. **Doesn't fit in a 72-min epoch with full pipeline**. Workaround: pre-compute multi-seed structures for the top-50 surrogate picks **continuously between epochs**, then per epoch just look up + select.

**Pipeline**: dedicated offline labeler box runs continuously, building a "verified rank-sum" table for every candidate the ranker has scored. Live miner consults this table for fast-phase selection.

## Component 5 — Adaptive metric weighting (MosPro)

**What it is**: At each epoch, look at the recently-submitted candidates and identify **which metric ranks them worst**. Bias the next batch's candidate selection toward fixing that metric.

**Concrete**: if our last 5 submissions had average per-metric ranks of [200, 180, 250, 1500, 400, 150, 100, 800, 50, 80], the `delta_sasa_refolded` rank of 1500 is dragging us. Next round, multiply the `delta_sasa_refolded_rank` head's contribution to the predicted rank_sum by 2×.

**Why**: rank_sum is dominated by the worst-ranked metric. One bad rank ruins the whole submission. Adaptive weighting matches the math.

## Component 6 — Liability hard filter

Liability metrics correlate weakly with winning (r ≤ 0.07) but they're a **floor**: a sequence with `liability_num_violations > 1` will rank poorly on those two metrics, adding ~thousands to rank_sum.

**Fix**: hard-filter candidates that fail AbNatiV2 humanness or have predicted liability violations *before* spending BoltzGen oracle calls.

## What we DON'T do (and why)

- **Full RFdiffusion / Chroma generators**: too compute-heavy, not VHH-specialized. BoltzGen already covers nanobody generation.
- **Train from scratch**: archive has 8k labels, that's tiny by deep-learning standards. Fine-tune frozen pretrained features instead.
- **Generic ΔG predictors**: validator scores 10 specific BoltzGen metrics, not raw affinity. Match the objective.
- **Optimize iiptm alone**: it's the 5th most important metric (r=0.29 to rank_sum). The interface-confidence cluster (ptm, target_iptm, interaction_pae) is more important and largely orthogonal.

## Expected outcome (honest)

| Component working | Expected effect |
|-------------------|-----------------|
| Multi-head ranker (Spearman 0.7 in high band) | +0.03 mean real iiptm = 0.78 |
| Batched diverse BO | +0.01 from less candidate redundancy |
| Multi-seed re-ranker | +0.02 (variance reduction at the selection step) |
| Adaptive metric weighting | +0.5 rank percentile (small but worth) |
| **Combined target** | **Mean rank-sum percentile ~p85, peak picks at p95+** |

Translation: we'd be at archive-top-15% on average instead of archive-top-25%, with occasional submissions hitting archive-top-5%. **That gets us into payout territory** (winners are top 0.16% per epoch; even top-5% per epoch gives nonzero emission via the validator's weight curve).

Not a guarantee. There is a real chance the multi-head ranker doesn't beat baseline. There is a real chance the archive top is so saturated that any submission within 3 mutations of a winning seed gets rejected for uniqueness before it can be scored. Both contingencies are addressed below.
