"""Combinatorial searcher that prefers historically-winning building blocks.

Empirical finding from Metanova/Submission-Archive analysis (May 2026):
- A handful of building blocks dominate the archive (e.g. mol_a 89710 wins 99x
  in rxn:2 alone; 50 building blocks per role account for the bulk of wins)
- Random sampling explores billions of combinations, most of which are unproductive

This searcher samples building blocks with a Pareto-weighted distribution:
- 70% of the time, sample from the top-N historical winners for this reaction+role
- 30% of the time, sample uniformly from the full eligible mol-id list

Falls back to uniform random sampling (the base CombinatorialSearcher) when
no top-blocks file is provided.

The "winner-biased" sampling means we converge on the same regions of chemical
space that have historically scored well, while still exploring 30% of the time.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterator, Optional

from combinatorial_db.reactions import react_molecules, react_three_components

from .reactions import ReactionCatalog, ReactionInfo


class WeightedCombinatorialSearcher:
    """Like CombinatorialSearcher but biases mol-id sampling toward historical winners."""

    def __init__(
        self,
        rxn_id: int,
        db_path: Optional[str] = None,
        rng: Optional[random.Random] = None,
        top_blocks_path: Optional[str] = None,
        bias_strength: float = 0.7,  # P(sample from winners) vs uniform
    ):
        self.catalog = ReactionCatalog.get(db_path)
        if rxn_id not in self.catalog:
            raise ValueError(f"Unknown rxn_id={rxn_id}")
        self.rxn_id = rxn_id
        self.info: ReactionInfo = self.catalog[rxn_id]
        self.db_path = self.catalog.db_path
        self.rng = rng or random.Random()
        self.bias_strength = bias_strength
        self._seen: set[str] = set()

        # Load top building blocks if available
        self.top_a = None
        self.top_b = None
        self.top_c = None
        if top_blocks_path and os.path.exists(top_blocks_path):
            with open(top_blocks_path) as f:
                data = json.load(f)
            # Filter to only mol-ids that exist in the role lists
            valid_a = set(self.info.mol_ids_a)
            valid_b = set(self.info.mol_ids_b)
            self.top_a = [m for m in data.get("mol_a", []) if m in valid_a]
            self.top_b = [m for m in data.get("mol_b", []) if m in valid_b]
            if self.info.is_three_component:
                valid_c = set(self.info.mol_ids_c)
                self.top_c = [m for m in data.get("mol_c", []) if m in valid_c]

    @property
    def has_winner_data(self) -> bool:
        return self.top_a is not None and self.top_b is not None

    @property
    def space_size(self) -> int:
        return self.info.space_size

    def _sample_role(self, role: str) -> int:
        """Sample one mol_id for a given role, biased toward winners if available."""
        if role == "a":
            top = self.top_a
            full = self.info.mol_ids_a
        elif role == "b":
            top = self.top_b
            full = self.info.mol_ids_b
        elif role == "c":
            top = self.top_c
            full = self.info.mol_ids_c
        else:
            raise ValueError(f"unknown role {role}")

        if top and self.rng.random() < self.bias_strength:
            return self.rng.choice(top)
        return self.rng.choice(full)

    def _sample_combo(self) -> tuple[str, tuple[int, ...]]:
        a = self._sample_role("a")
        b = self._sample_role("b")
        if self.info.is_three_component:
            c = self._sample_role("c")
            name = f"rxn:{self.rxn_id}:{a}:{b}:{c}"
            return name, (a, b, c)
        name = f"rxn:{self.rxn_id}:{a}:{b}"
        return name, (a, b)

    def _react(self, ids: tuple[int, ...]) -> Optional[str]:
        if len(ids) == 2:
            return react_molecules(self.rxn_id, ids[0], ids[1], self.db_path)
        return react_three_components(self.rxn_id, ids[0], ids[1], ids[2], self.db_path)

    def generate_batch(self, batch_size: int, max_attempts: Optional[int] = None) -> list[tuple[str, str]]:
        if max_attempts is None:
            max_attempts = batch_size * 10
        results: list[tuple[str, str]] = []
        attempts = 0
        while len(results) < batch_size and attempts < max_attempts:
            attempts += 1
            name, ids = self._sample_combo()
            if name in self._seen:
                continue
            self._seen.add(name)
            smiles = self._react(ids)
            if smiles is None:
                continue
            results.append((name, smiles))
        return results

    def iter_samples(self, max_attempts: Optional[int] = None) -> Iterator[tuple[str, str]]:
        attempts = 0
        while max_attempts is None or attempts < max_attempts:
            attempts += 1
            name, ids = self._sample_combo()
            if name in self._seen:
                continue
            self._seen.add(name)
            smiles = self._react(ids)
            if smiles is None:
                continue
            yield name, smiles

    def reset_seen(self) -> None:
        self._seen.clear()
