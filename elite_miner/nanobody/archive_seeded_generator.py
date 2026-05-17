"""Archive-seeded nanobody generator.

Instead of starting from a single consensus sequence (PSSMGenerator), this
generator picks one of the top-N archive winners at random and mutates only
1-3 positions using PSSM-weighted residue sampling.

Why: PSSMGenerator independently samples residues at each variable position,
which can produce combinations that aren't actually winners (each residue
is common but the combination is novel). Archive-seeded generation
guarantees we start from a known-winning sequence and stay close to it.

Inputs:
- archive_top: list of (sequence, iiptm) tuples from the top of the archive
- pssm: position → {aa: freq} mapping (used to bias mutation choice)
- min_mutations / max_mutations: how many positions to mutate from the seed
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Iterator, Optional

from .filters import NanobodyFilter, NanobodyValidityConfig, ALLOWED_AAS, normalize_seq, seq_hash


@dataclass
class ArchiveSeededConfig:
    min_mutations: int = 1
    max_mutations: int = 3
    max_attempts_per_candidate: int = 30
    pssm_path: str = "cache/q9nzq7_pssm.json"
    seeds_path: str = "cache/q9nzq7_top_seeds.json"  # list of {seq, iiptm}
    # When generating, weight seeds by iiptm so we pick higher-scoring seeds more often.
    weight_seeds_by_iiptm: bool = True


class ArchiveSeededGenerator:
    """Mutate from an actual archive winner — strictly stays in winning neighborhood."""

    def __init__(
        self,
        validity_cfg: NanobodyValidityConfig,
        gen_cfg: Optional[ArchiveSeededConfig] = None,
        rng: Optional[random.Random] = None,
        seeds: Optional[list[tuple[str, float]]] = None,
        pssm: Optional[dict] = None,
    ):
        self.filter = NanobodyFilter(validity_cfg)
        self.gen_cfg = gen_cfg or ArchiveSeededConfig()
        self.rng = rng or random.Random()
        self._seen: set[str] = set()

        # Load PSSM
        if pssm is not None:
            self.pssm = pssm
        else:
            if not os.path.exists(self.gen_cfg.pssm_path):
                raise FileNotFoundError(f"PSSM file {self.gen_cfg.pssm_path} not found")
            with open(self.gen_cfg.pssm_path) as f:
                raw = json.load(f)
            self.pssm = {int(k): v for k, v in raw.items()}

        # Load seeds
        if seeds is not None:
            self.seeds = seeds
        else:
            if not os.path.exists(self.gen_cfg.seeds_path):
                raise FileNotFoundError(f"Seeds file {self.gen_cfg.seeds_path} not found")
            with open(self.gen_cfg.seeds_path) as f:
                raw = json.load(f)
            self.seeds = [(item["sequence"], float(item.get("iiptm", 1.0))) for item in raw]

        if not self.seeds:
            raise ValueError("No seeds provided")

        # Identify variable positions for mutation (PSSM positions where top freq < 0.95)
        self.variable_positions = []
        for pos in range(len(self.seeds[0][0])):
            if pos in self.pssm:
                top_freq = max(self.pssm[pos].values())
                if top_freq < 0.95:
                    self.variable_positions.append(pos)

    def _pick_seed(self) -> str:
        if self.gen_cfg.weight_seeds_by_iiptm:
            seqs = [s for s, _ in self.seeds]
            weights = [w for _, w in self.seeds]
            return self.rng.choices(seqs, weights=weights, k=1)[0]
        return self.rng.choice(self.seeds)[0]

    def _sample_residue(self, pos: int, current: str) -> str:
        """Sample residue at variable position weighted by PSSM, EXCLUDING current (force mutation)."""
        if pos not in self.pssm:
            return self.rng.choice([a for a in ALLOWED_AAS if a != current])
        choices = [a for a in self.pssm[pos] if a != current]
        if not choices:
            # Only the current residue available — return as-is
            return current
        weights = [self.pssm[pos][a] for a in choices]
        return self.rng.choices(choices, weights=weights, k=1)[0]

    def _mutate(self) -> str:
        seed = self._pick_seed()
        seq = list(seed)
        n_mut = self.rng.randint(self.gen_cfg.min_mutations, self.gen_cfg.max_mutations)
        # Pick distinct variable positions to mutate
        valid_positions = [p for p in self.variable_positions if p < len(seq)]
        n_mut = min(n_mut, len(valid_positions))
        positions = self.rng.sample(valid_positions, n_mut)
        for pos in positions:
            new_aa = self._sample_residue(pos, seq[pos])
            seq[pos] = new_aa
        return "".join(seq)

    def generate_one(self) -> Optional[str]:
        for _ in range(self.gen_cfg.max_attempts_per_candidate):
            seq = self._mutate()
            seq = normalize_seq(seq)
            h = seq_hash(seq)
            if h in self._seen:
                continue
            ok, _ = self.filter.is_valid(seq)
            if not ok:
                continue
            self._seen.add(h)
            return seq
        return None

    def generate_batch(self, batch_size: int) -> list[str]:
        out: list[str] = []
        budget = batch_size * self.gen_cfg.max_attempts_per_candidate
        attempts = 0
        while len(out) < batch_size and attempts < budget:
            attempts += 1
            seq = self.generate_one()
            if seq is not None:
                out.append(seq)
        return out

    def iter_samples(self) -> Iterator[str]:
        while True:
            seq = self.generate_one()
            if seq is None:
                self.gen_cfg.max_mutations = min(self.gen_cfg.max_mutations + 1, 10)
                continue
            yield seq

    def reset_seen(self) -> None:
        self._seen.clear()
