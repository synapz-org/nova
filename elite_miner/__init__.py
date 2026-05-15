"""Elite miner for NOVA SN68.

Submodules:
  molecule.*   — molecule track (combinatorial search + Boltz2)
  nanobody.*   — nanobody track (template-mutator + BoltzGen)
  config       — argparse + config.yaml loader
  timing       — epoch boundary detection
  submission   — drand encrypt + chain commit + GitHub upload
  run          — main async loop (entry point: python -m elite_miner.run)
"""
