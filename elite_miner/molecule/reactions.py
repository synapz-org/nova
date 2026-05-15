"""Reaction metadata cache for elite miner.

Wraps combinatorial_db.reactions with per-process caching so we don't re-query
SQLite for the same role lists every batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from combinatorial_db.reactions import (
    get_available_reactions,
    get_molecule_ids_for_role,
    get_db_path,
)


@dataclass
class ReactionInfo:
    rxn_id: int
    role_a: int
    role_b: int
    role_c: Optional[int]
    mol_ids_a: list[int] = field(default_factory=list)
    mol_ids_b: list[int] = field(default_factory=list)
    mol_ids_c: list[int] = field(default_factory=list)

    @property
    def is_three_component(self) -> bool:
        return self.role_c is not None and self.role_c != 0

    @property
    def space_size(self) -> int:
        a, b = len(self.mol_ids_a), len(self.mol_ids_b)
        if self.is_three_component:
            return a * b * len(self.mol_ids_c)
        return a * b


class ReactionCatalog:
    """Loads reactions once, caches mol-id lists per role."""

    _lock = Lock()
    _instance: Optional["ReactionCatalog"] = None

    @classmethod
    def get(cls, db_path: Optional[str] = None) -> "ReactionCatalog":
        with cls._lock:
            if cls._instance is None or (db_path and cls._instance.db_path != db_path):
                cls._instance = ReactionCatalog(db_path)
            return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_db_path()
        self._reactions: dict[int, ReactionInfo] = {}
        self._load()

    def _load(self) -> None:
        raw = get_available_reactions(self.db_path)
        for rxn_id, role_a, role_b, role_c in raw:
            info = ReactionInfo(
                rxn_id=int(rxn_id),
                role_a=int(role_a),
                role_b=int(role_b),
                role_c=int(role_c) if role_c not in (None, 0) else None,
            )
            info.mol_ids_a = get_molecule_ids_for_role(self.db_path, info.role_a)
            info.mol_ids_b = get_molecule_ids_for_role(self.db_path, info.role_b)
            if info.is_three_component:
                info.mol_ids_c = get_molecule_ids_for_role(self.db_path, info.role_c)
            self._reactions[info.rxn_id] = info

    def __getitem__(self, rxn_id: int) -> ReactionInfo:
        return self._reactions[rxn_id]

    def __contains__(self, rxn_id: int) -> bool:
        return rxn_id in self._reactions

    def all_ids(self) -> list[int]:
        return sorted(self._reactions.keys())


def parse_allowed_reaction(allowed_reaction: Optional[str]) -> Optional[int]:
    """Convert 'rxn:N' string from challenge_params to int N. None passes through."""
    if not allowed_reaction:
        return None
    if allowed_reaction.startswith("rxn:"):
        try:
            return int(allowed_reaction.split(":", 1)[1])
        except (ValueError, IndexError):
            return None
    try:
        return int(allowed_reaction)
    except ValueError:
        return None
