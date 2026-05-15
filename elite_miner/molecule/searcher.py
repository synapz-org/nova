"""Combinatorial searcher for any reaction id.

Strategy: random sampling from role-compatible mol-id lists with dedup.
Exhaustive enumeration is infeasible — even the smallest reaction has
hundreds of millions of combinations.
"""

from __future__ import annotations

import random
from typing import Iterator, Optional

from combinatorial_db.reactions import react_molecules, react_three_components

from .reactions import ReactionCatalog, ReactionInfo


class CombinatorialSearcher:
    """Sample valid product combinations for one reaction id."""

    def __init__(
        self,
        rxn_id: int,
        db_path: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        self.catalog = ReactionCatalog.get(db_path)
        if rxn_id not in self.catalog:
            raise ValueError(f"Unknown rxn_id={rxn_id}; available: {self.catalog.all_ids()}")
        self.rxn_id = rxn_id
        self.info: ReactionInfo = self.catalog[rxn_id]
        self.db_path = self.catalog.db_path
        self.rng = rng or random.Random()
        self._seen: set[str] = set()

    @property
    def space_size(self) -> int:
        return self.info.space_size

    def _sample_combo(self) -> tuple[str, tuple[int, ...]]:
        """Pick one random combo. Returns (mol_name, id_tuple)."""
        a = self.rng.choice(self.info.mol_ids_a)
        b = self.rng.choice(self.info.mol_ids_b)
        if self.info.is_three_component:
            c = self.rng.choice(self.info.mol_ids_c)
            name = f"rxn:{self.rxn_id}:{a}:{b}:{c}"
            return name, (a, b, c)
        name = f"rxn:{self.rxn_id}:{a}:{b}"
        return name, (a, b)

    def _react(self, ids: tuple[int, ...]) -> Optional[str]:
        if len(ids) == 2:
            return react_molecules(self.rxn_id, ids[0], ids[1], self.db_path)
        return react_three_components(self.rxn_id, ids[0], ids[1], ids[2], self.db_path)

    def generate_batch(self, batch_size: int, max_attempts: Optional[int] = None) -> list[tuple[str, str]]:
        """Generate up to batch_size unique (mol_name, smiles) pairs.

        Stops early if max_attempts (default 10 * batch_size) is reached
        without filling the batch — protects against reactions where most
        random combinations fail to react.
        """
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
        """Yield (mol_name, smiles) one at a time forever (or up to max_attempts).

        Useful when you want to plug filters/scoring inline and stop on a budget.
        """
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
        """Clear deduplication memory. Use at epoch boundary."""
        self._seen.clear()
