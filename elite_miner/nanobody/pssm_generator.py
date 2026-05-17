"""PSSM-guided nanobody generator.

Empirical finding from Metanova/Submission-Archive (May 2026):
- Top 200 winners are ALL length 120 with the same scaffold
- Variation is concentrated in ~35 specific positions, with strong residue
  preferences at each (e.g. pos 49 is D 93.5% of the time, Q 6.5%)
- Most positions in the framework are completely fixed

This generator:
1. Starts from a top-winner consensus sequence
2. At each "variable position" (defined by the PSSM), samples a residue
   weighted by the empirical winner frequency
3. Most submissions will be in the high-iiptm neighborhood

A `num_mutations` parameter controls diversity — at min_mutations=1, output
sequences are nearly identical to consensus; at higher values, more positions
are sampled away from consensus.

The PSSM is loaded from a JSON file at instantiation. Format:
    {
        "<position>": {"<aa>": <frequency>, ...},
        ...
    }
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Iterator, Optional

from .filters import NanobodyFilter, NanobodyValidityConfig, ALLOWED_AAS, normalize_seq, seq_hash


# Default consensus sequence: the most common residue at each position from
# top 200 archive winners. We can override with a different template.
# Position-by-position consensus from top 200 winners (May 2026 analysis).
DEFAULT_CONSENSUS_Q9NZQ7 = (
    "DVTLAESGGGLVQAGGSLRLSCAATGFTYSSYAMSWVRQRPGKEREWVSDITSQGDQTDYADFVKGRFTISRDNAKNTLYLQMTSLRPEDTAVYYCSKFPYRFMQTFAQHGQGTTVTVSA"
)


@dataclass
class PSSMGenerationConfig:
    min_mutations: int = 3
    max_mutations: int = 10
    max_attempts_per_candidate: int = 30
    pssm_path: str = "cache/q9nzq7_pssm.json"


class PSSMGenerator:
    """Generator that samples from per-position winner distributions."""

    def __init__(
        self,
        validity_cfg: NanobodyValidityConfig,
        gen_cfg: Optional[PSSMGenerationConfig] = None,
        consensus: Optional[str] = None,
        rng: Optional[random.Random] = None,
        pssm: Optional[dict] = None,
    ):
        self.filter = NanobodyFilter(validity_cfg)
        self.gen_cfg = gen_cfg or PSSMGenerationConfig()
        self.consensus = consensus or DEFAULT_CONSENSUS_Q9NZQ7
        self.rng = rng or random.Random()
        self._seen: set[str] = set()

        if pssm is not None:
            self.pssm = pssm
        else:
            if not os.path.exists(self.gen_cfg.pssm_path):
                raise FileNotFoundError(
                    f"PSSM file {self.gen_cfg.pssm_path} not found. "
                    "Build it via /tmp/build_pssm.py or pass pssm= explicitly."
                )
            with open(self.gen_cfg.pssm_path) as f:
                raw = json.load(f)
            self.pssm = {int(k): v for k, v in raw.items()}

        # Identify variable positions (those with non-1.0 max frequency)
        self.variable_positions = []
        for pos in range(len(self.consensus)):
            if pos in self.pssm:
                top_freq = max(self.pssm[pos].values())
                if top_freq < 0.95:  # variable
                    self.variable_positions.append(pos)

    def _sample_residue(self, pos: int, current: str) -> str:
        """Sample a new residue at this position weighted by winner frequency."""
        if pos not in self.pssm:
            # Mutate uniformly if no PSSM data for this position
            return self.rng.choice([a for a in ALLOWED_AAS if a != current])
        probs = self.pssm[pos]
        choices = list(probs.keys())
        weights = list(probs.values())
        # Sample with replacement from PSSM. Allow same-as-current (just makes
        # this "mutation" a no-op, which is fine — we'll pick another position).
        return self.rng.choices(choices, weights=weights, k=1)[0]

    def _mutate(self) -> str:
        """Start from consensus, mutate `n_mut` variable positions weighted by PSSM."""
        seq = list(self.consensus)
        n_mut = self.rng.randint(self.gen_cfg.min_mutations, self.gen_cfg.max_mutations)
        n_mut = min(n_mut, len(self.variable_positions))

        # Sample n_mut variable positions (no replacement)
        positions = self.rng.sample(self.variable_positions, n_mut)
        for pos in positions:
            new_aa = self._sample_residue(pos, seq[pos])
            seq[pos] = new_aa
        return "".join(seq)

    def generate_one(self) -> Optional[str]:
        """Try to generate one valid unique sequence."""
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
                # Widen mutations if running dry
                self.gen_cfg.max_mutations = min(self.gen_cfg.max_mutations + 1, 25)
                continue
            yield seq

    def reset_seen(self) -> None:
        self._seen.clear()
