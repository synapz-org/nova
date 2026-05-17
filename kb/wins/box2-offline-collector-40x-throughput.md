# Second labeling box: BoltzGen at ~2 min/label vs box 1's ~85 min/label

## Setup
Rented a second Basilica A100 80GB spot ($1.05/hr) and provisioned with the same nova stack as box 1. Runs `elite_miner.scripts.offline_label_collector` which:
- Generates 5000 candidates from ArchiveSeededGenerator each iteration
- Surrogate-ranks them and picks top-K (K=1)
- Runs real BoltzGen on the pick(s)
- Writes `(seq, predicted_iiptm, real_iiptm, all_bg_metrics, machine_id)` to `cache/offline_labels/Q9NZQ7.parquet`

## Measurement
Box 2 first end-to-end label:
- Step 1 design: 50s
- Step 2 folding: 49.5s
- Step 3 analysis: 16s
- **Total: 136 seconds per label**

Box 1's live submission worker takes ~85 minutes per label — 40x slower. The difference is the input cardinality. Box 1 passes the full submitted sequence through BoltzGen with the default `diffusion_samples`/dataloader settings (171 samples per pipeline step). Box 2 passes a single sequence (K=1, topk=1) with `num_designs=1`.

Effective rate:
- Box 1: ~17 labels/day
- Box 2: ~640 labels/day theoretical, probably ~400-500 with disk-IO and BoltzGen overhead
- Combined: ~20x speedup on label collection vs single-box

## Where
Code: `elite_miner/scripts/offline_label_collector.py`. Output: `cache/offline_labels/Q9NZQ7.parquet`.

Rental: `05b14761-4359-4cf4-b46b-f338f737e5e3` at `185.216.22.220`. ~$1.05/hr.

## Provisioning notes (for future second-box bringup)
Three gotchas hit while setting up:
1. `install_deps.sh` silently exits 0 if `uv` isn't on PATH ([[install-deps-uv-not-in-path]])
2. Submodule `NOVA-nanobody-filter` must be initialized (`git submodule update --init --recursive`); otherwise `import utils` fails because `nanobodies.py` line 13 inserts the submodule path before importing metanano
3. `lightgbm` is not in `requirements.txt`; install separately or the surrogate silently falls back to proxy and picks random sequences
4. A100 spot instances may default to MIG mode enabled — disable with `sudo nvidia-smi -mig 0` or PyTorch sees device_count=1 but `cuda.is_available()==False`

## When it might stop working
- Box 2 rental expires or gets reclaimed (spot pricing)
- Surrogate model file goes stale (we'd want to refresh the model dir on box 2 when box 1 retrains)
- Generator config diverges between box 1 (live miner) and box 2 (offline collector) — currently both use ArchiveSeededGenerator m=1-3
