# Probability of success — no hype

You said "prove that we can do this." This doc is the honest assessment of probability per outcome, with the reasoning behind each estimate. If you can't stomach the numbers, this plan isn't ready for deployment.

## Definitions

**Compete** = average rank-sum percentile ≥ 85 across our submissions (currently we're at ~p70).

**Place** = at least one epoch-winning submission per day (top 0.16% of archive on rank-sum).

**Profitable** = net positive after compute costs (Basilica $1-2/hr × 2 boxes × N hours) vs accrued payout in τ at current emission rates.

## Outcome probabilities (12 hours of effort, $50 budget cap)

| Outcome | Probability | Reasoning |
|---------|-------------|-----------|
| **Don't get deregistered before any payout** | **0.85** | Playbook + deregistration guard make this nearly automatic. Failure mode: 1τ runway only buys ~6 prune cycles, so if our ranker is bad and we accrue 0 emission, we still get pruned eventually. |
| **Pipeline works end-to-end** (validators score our submissions) | **0.95** | We proved this empirically yesterday (39/44 historical submissions decrypted cleanly). Today's pre-flight verifier checks the gates explicitly. The remaining 5% is "something we don't know we don't know." |
| **Compete** (mean p85) | **0.45** | If ESM2 ranker hits high-band Spearman ≥ 0.65 we likely get there. PoC tonight is consistent with this assuming ESM2 closes the feature gap (literature suggests it does for related tasks). But Spearman gain from features is partly conjecture. |
| **Place** (win ≥ 1 epoch/day) | **0.25** | Conditional on compete, this requires hitting the top ~6 archive entries in a 24h window. With ranker variance ~0.04 on real iiptm, that's a tail event we'd see maybe 1-3× per day if we're submitting at archive p90+. |
| **Profitable** (positive net τ at end of week) | **0.30** | Requires placing on enough epochs to pay back ~$50-100 of Basilica compute. At current emission rates and burn structure, one epoch win is worth roughly τ0.05-0.20 (off-chain payout); we'd need 3-10 wins to break even. Plausible if we place, unlikely if we just compete. |
| **Dominate** (consistent winner) | **0.05** | Would require richer features than ESM2 (e.g., AbMPNN log-likelihoods, ipSAE rescoring) PLUS multi-seed BoltzGen re-ranking PLUS active learning. Each component has its own implementation risk. Stacking them all is ambitious for one person on a $50 budget. |

## Where the uncertainty lives

**The hardest unknown is whether ESM2 features actually lift high-band Spearman from 0.46 to 0.65+.** Tonight's PoC proves the architecture works but couldn't test ESM2 (CPU-only). Literature suggests it should — pLDDT-Predictor hits 0.79, AbRank crowd-sources data and gets strong ranking on AB-Ag — but those are different tasks.

If ESM2 doesn't deliver the lift, we're stuck at mean p75 like yesterday and the new strategy is mostly a no-op. **The first 24 hours of compute spend should be running the GPU ranker training and validating against held-out archive labels.** Decision point: if high-band Spearman ≤ 0.55 after training, **stop, don't deploy live mining**, the assumptions are wrong.

## What I'd bet on with my own money

If you gave me $100 and asked me to make a decision purely on probabilities above:

- **$50** to validate the ESM2 ranker (1 day of GPU + my time). High EV: either we get a working surrogate (probability 0.6) or we save the next $400 of compute by not deploying a known-bad strategy.
- **$50 reserve** for actual deployment, only released after ranker hits high-band Spearman ≥ 0.55 in validation.
- **Don't bet the remaining $5800 Basilica balance until we have evidence** that the new ranker works.

That's the rational allocation. Anything else is throwing money at a problem hoping it works.

## What could move these numbers up

**Add 0.10 to "Compete" probability if:**
- Multi-seed BoltzGen re-ranking gets implemented (variance reduction at the selection step is a known win)
- Adaptive metric weighting (MosPro) is operational and gets feedback after the first 50 labeled submissions
- We pre-validate that uniqueness isn't a hard blocker (i.e., the sequence space near the archive top isn't fully claimed)

**Add 0.10 to "Place" probability if:**
- We invest a second GPU-day in active learning (label our own top picks, retrain weekly)
- We implement the deregistration guard as a circuit breaker that auto-pauses and pages on detection

**Subtract 0.20 from everything if:**
- We deploy without running the GPU validation first
- We try to fix bugs while live in production
- We don't read MINER.md before deploying (yes, again, this is the meta-lesson)

## The honest bottom line

I think there's about a **30-45% chance** that 24 hours of careful work and $50 of compute gets us to "competing." There's about a **5% chance** we dominate. The most likely outcome is **a modest improvement over yesterday's mean p70 → maybe p80-85**, which is interesting science but not yet profitable mining.

The thing I'm most confident in is that **the gates we caught yesterday (private repo, empty side, deregistration) won't catch us again** if we follow the deployment playbook. So at worst we waste $50 doing real experiments instead of debugging known bugs.

If you want to fire me, fire me on the merits of this assessment, not on whether I sold you a sure thing. I won't promise that. The plan is sound; success is uncertain; the worst-case downside is bounded; the upside is modest but real.

Ship or don't.
