"""CombinatorialSearcher for rxn5 building block loading and batch generation."""

import itertools
import random
import sqlite3

from combinatorial_db.reactions import react_three_components


PRIORITY_SCAFFOLD_IDS = [192490, 192488, 192710, 192713]


class CombinatorialSearcher:
    """Searches the combinatorial chemistry database for molecule combinations."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.scaffolds: list[tuple[int, str, int]] = []
        self.boronic_acids: list[tuple[int, str, int]] = []
        self._seen_names: set[str] = set()
        self._combo_iter = None

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

    def _make_combo_iterator(self):
        """Yield (scaffold_id, boronic1_id, boronic2_id) tuples.

        For each scaffold in priority order, iterate all boronic acid pairs.
        Boronic acids are shuffled per scaffold for variety.
        """
        for scaffold_id, _, _ in self.scaffolds:
            acids = list(self.boronic_acids)
            random.shuffle(acids)
            acid_ids = [mid for mid, _, _ in acids]
            for b1_id, b2_id in itertools.permutations(acid_ids, 2):
                yield scaffold_id, b1_id, b2_id

    def generate_batch(self, batch_size: int = 100) -> list[tuple[str, str]]:
        """Generate a batch of (mol_name, smiles) from combinatorial reactions.

        Uses the combo iterator; skips duplicates and failed reactions.
        Resets the iterator when exhausted (reshuffles).
        """
        results: list[tuple[str, str]] = []

        while len(results) < batch_size:
            if self._combo_iter is None:
                self._combo_iter = self._make_combo_iterator()

            try:
                s_id, b1_id, b2_id = next(self._combo_iter)
            except StopIteration:
                # Iterator exhausted — reset with new shuffle
                self._combo_iter = self._make_combo_iterator()
                # If we got nothing at all, break to avoid infinite loop
                if not results:
                    break
                continue

            mol_name = f"rxn:5:{s_id}:{b1_id}:{b2_id}"
            if mol_name in self._seen_names:
                continue

            smiles = react_three_components(5, s_id, b1_id, b2_id, self.db_path)
            if smiles is None:
                continue

            self._seen_names.add(mol_name)
            results.append((mol_name, smiles))

        return results
