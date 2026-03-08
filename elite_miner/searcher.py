"""CombinatorialSearcher for rxn5 building block loading."""

import random
import sqlite3


PRIORITY_SCAFFOLD_IDS = [192490, 192488, 192710, 192713]


class CombinatorialSearcher:
    """Searches the combinatorial chemistry database for molecule combinations."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.scaffolds: list[tuple[int, str, int]] = []
        self.boronic_acids: list[tuple[int, str, int]] = []

    def load_rxn5_building_blocks(self):
        """Load scaffolds (roleA) and boronic acids (roleB/C) for rxn:5 double Suzuki coupling."""
        conn = sqlite3.connect(self.db_path)

        # Load scaffolds: role_mask & 384 == 384
        rows = conn.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE role_mask & 384 = 384"
        ).fetchall()

        priority_set = set(PRIORITY_SCAFFOLD_IDS)
        priority = [r for r in rows if r[0] in priority_set]
        rest = [r for r in rows if r[0] not in priority_set]
        random.shuffle(rest)

        # Sort priority scaffolds to match the defined order
        priority.sort(key=lambda r: PRIORITY_SCAFFOLD_IDS.index(r[0]))
        self.scaffolds = priority + rest

        # Load boronic acids: role_mask & 1024 == 1024
        rows = conn.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE role_mask & 1024 = 1024"
        ).fetchall()
        random.shuffle(rows)
        self.boronic_acids = rows

        conn.close()
