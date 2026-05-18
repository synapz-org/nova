# Competitive NOVA SN68 Miner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a miner that wins Boltz2 scoring epochs on SN68 by systematically searching rxn:5 combinatorial space with PSICHIC-guided ranking.

**Architecture:** New `elite_miner/` module with 4 files: `searcher.py` (combinatorial enumeration + reaction), `filters.py` (validity checks), `scorer.py` (PSICHIC scoring + proxy ranking), and `run.py` (main loop reusing existing BT infrastructure from `neurons/miner.py`). Reuses existing `combinatorial_db/reactions.py` for Suzuki reaction logic, `utils/molecules.py` for validation helpers, and `PSICHIC/wrapper.py` for scoring.

**Tech Stack:** Python 3.12, SQLite (combinatorial DB), RDKit (molecular validation), PSICHIC (affinity scoring), Bittensor SDK (chain interaction), HuggingFace Hub (uniqueness checks)

---

### Task 1: CombinatorialSearcher - Building Block Loading

**Files:**
- Create: `elite_miner/searcher.py`
- Test: `elite_miner/tests/test_searcher.py`

**Step 1: Write the failing test**

```python
# elite_miner/tests/test_searcher.py
import os
import sqlite3
import pytest

# DB path relative to repo root
DB_PATH = os.path.join(os.path.dirname(__file__), "../../combinatorial_db/molecules.sqlite")

def test_db_exists():
    assert os.path.exists(DB_PATH), f"molecules.sqlite not found at {DB_PATH}"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM molecules")
    assert cur.fetchone()[0] > 0
    conn.close()

def test_load_building_blocks():
    from elite_miner.searcher import CombinatorialSearcher
    searcher = CombinatorialSearcher(DB_PATH)
    searcher.load_rxn5_building_blocks()

    assert len(searcher.scaffolds) > 0, "No scaffolds loaded"
    assert len(searcher.boronic_acids) > 0, "No boronic acids loaded"

    # rxn5: roleA=384, roleB=roleC=1024
    # Every scaffold should have (role_mask & 384) == 384
    for mol_id, smiles, mask in searcher.scaffolds:
        assert (mask & 384) == 384, f"Scaffold {mol_id} has wrong role_mask {mask}"

    # Every boronic acid should have (role_mask & 1024) == 1024
    for mol_id, smiles, mask in searcher.boronic_acids:
        assert (mask & 1024) == 1024, f"Boronic acid {mol_id} has wrong role_mask {mask}"

def test_prioritized_scaffolds():
    from elite_miner.searcher import CombinatorialSearcher
    searcher = CombinatorialSearcher(DB_PATH)
    searcher.load_rxn5_building_blocks()

    # Known winning scaffold IDs should be in the list
    scaffold_ids = {s[0] for s in searcher.scaffolds}
    for winning_id in [192490, 192488, 192710, 192713]:
        assert winning_id in scaffold_ids, f"Winning scaffold {winning_id} missing"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest elite_miner/tests/test_searcher.py -v`
Expected: FAIL with "No module named 'elite_miner.searcher'" or ImportError

**Step 3: Write minimal implementation**

```python
# elite_miner/searcher.py
import sqlite3
import random
from typing import Optional

# Known winning scaffolds get searched first
PRIORITY_SCAFFOLD_IDS = [192490, 192488, 192710, 192713]

class CombinatorialSearcher:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.scaffolds = []      # [(mol_id, smiles, role_mask), ...]
        self.boronic_acids = []   # [(mol_id, smiles, role_mask), ...]

    def load_rxn5_building_blocks(self):
        """Load all rxn:5 building blocks from the database.

        rxn:5 (suzuki_bromide_then_chloride):
          roleA = 384 (Br+Cl scaffold)
          roleB = roleC = 1024 (boronic acids)
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Load scaffolds (roleA=384): molecules where (role_mask & 384) == 384
        cur.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & 384) = 384"
        )
        all_scaffolds = cur.fetchall()

        # Put priority scaffolds first, then the rest shuffled
        priority = [s for s in all_scaffolds if s[0] in PRIORITY_SCAFFOLD_IDS]
        rest = [s for s in all_scaffolds if s[0] not in PRIORITY_SCAFFOLD_IDS]
        random.shuffle(rest)
        self.scaffolds = priority + rest

        # Load boronic acids (roleB=roleC=1024)
        cur.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & 1024) = 1024"
        )
        self.boronic_acids = cur.fetchall()

        conn.close()
```

**Step 4: Ensure `elite_miner/__init__.py` and `elite_miner/tests/__init__.py` exist**

```python
# elite_miner/__init__.py
# (keep existing or create empty)

# elite_miner/tests/__init__.py
# (keep existing or create empty)
```

**Step 5: Run test to verify it passes**

Run: `python3 -m pytest elite_miner/tests/test_searcher.py -v`
Expected: 3 PASS

**Step 6: Commit**

```bash
git add elite_miner/searcher.py elite_miner/tests/test_searcher.py
git commit -m "feat: add CombinatorialSearcher with rxn5 building block loading"
```

---

### Task 2: CombinatorialSearcher - Batch Combination Generation

**Files:**
- Modify: `elite_miner/searcher.py`
- Modify: `elite_miner/tests/test_searcher.py`

**Step 1: Write the failing test**

```python
# Append to elite_miner/tests/test_searcher.py

def test_generate_batch():
    from elite_miner.searcher import CombinatorialSearcher
    searcher = CombinatorialSearcher(DB_PATH)
    searcher.load_rxn5_building_blocks()

    batch = searcher.generate_batch(batch_size=10)

    assert len(batch) == 10
    for mol_name, smiles in batch:
        # mol_name should be rxn:5:scaffold:boronic1:boronic2
        assert mol_name.startswith("rxn:5:"), f"Bad mol_name: {mol_name}"
        parts = mol_name.split(":")
        assert len(parts) == 5, f"Expected 5 parts, got {len(parts)}: {mol_name}"
        # smiles should be a non-empty string
        assert smiles and len(smiles) > 0, f"Empty SMILES for {mol_name}"

def test_generate_batch_no_duplicates_within_batch():
    from elite_miner.searcher import CombinatorialSearcher
    searcher = CombinatorialSearcher(DB_PATH)
    searcher.load_rxn5_building_blocks()

    batch = searcher.generate_batch(batch_size=50)
    mol_names = [name for name, _ in batch]
    assert len(mol_names) == len(set(mol_names)), "Batch contains duplicates"

def test_generate_batch_skips_already_seen():
    from elite_miner.searcher import CombinatorialSearcher
    searcher = CombinatorialSearcher(DB_PATH)
    searcher.load_rxn5_building_blocks()

    batch1 = searcher.generate_batch(batch_size=20)
    batch2 = searcher.generate_batch(batch_size=20)

    names1 = {name for name, _ in batch1}
    names2 = {name for name, _ in batch2}
    assert names1.isdisjoint(names2), "Second batch reused molecules from first"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest elite_miner/tests/test_searcher.py::test_generate_batch -v`
Expected: FAIL with AttributeError

**Step 3: Write implementation**

Add to `elite_miner/searcher.py` inside the `CombinatorialSearcher` class:

```python
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.scaffolds = []
        self.boronic_acids = []
        self._seen_names = set()  # track generated combinations
        self._combo_iter = None   # lazy iterator over combinations

    def _make_combo_iterator(self):
        """Yields (scaffold_id, boronic1_id, boronic2_id) tuples.

        Iterates scaffolds in priority order. For each scaffold,
        yields all boronic acid pairs (with boronic1 <= boronic2 to avoid
        symmetric duplicates since both coupling positions are equivalent
        up to Boltz2 scoring).
        """
        for scaffold in self.scaffolds:
            s_id = scaffold[0]
            # Shuffle boronic acids for each scaffold to vary exploration
            ba_shuffled = list(self.boronic_acids)
            random.shuffle(ba_shuffled)
            for i, ba1 in enumerate(ba_shuffled):
                for ba2 in ba_shuffled[i:]:
                    yield s_id, ba1[0], ba2[0]

    def generate_batch(self, batch_size: int = 100) -> list[tuple[str, str]]:
        """Generate a batch of (mol_name, smiles) tuples.

        Returns up to batch_size molecules that haven't been seen before.
        Reactions that fail to produce a valid SMILES are skipped.
        """
        from combinatorial_db.reactions import react_three_components

        if self._combo_iter is None:
            self._combo_iter = self._make_combo_iterator()

        db_path = self.db_path
        results = []

        while len(results) < batch_size:
            try:
                s_id, b1_id, b2_id = next(self._combo_iter)
            except StopIteration:
                # Exhausted current iterator; reshuffle and restart
                self._combo_iter = self._make_combo_iterator()
                break

            mol_name = f"rxn:5:{s_id}:{b1_id}:{b2_id}"
            if mol_name in self._seen_names:
                continue

            self._seen_names.add(mol_name)

            smiles = react_three_components(5, s_id, b1_id, b2_id, db_path)
            if smiles is None:
                continue

            results.append((mol_name, smiles))

        return results
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest elite_miner/tests/test_searcher.py -v`
Expected: 6 PASS

**Step 5: Commit**

```bash
git add elite_miner/searcher.py elite_miner/tests/test_searcher.py
git commit -m "feat: add batch combination generation to CombinatorialSearcher"
```

---

### Task 3: ValidityFilter

**Files:**
- Create: `elite_miner/filters.py`
- Modify: `elite_miner/tests/test_searcher.py`

**Step 1: Write the failing test**

```python
# Append to elite_miner/tests/test_searcher.py (or create test_filters.py)

def test_validity_filter_accepts_good_molecule():
    from elite_miner.filters import ValidityFilter
    # A known valid rxn5 product (from competition analysis)
    smiles = "c1ccc(-c2cc(-c3ccoc3)ccc2CN2CCC2)cc1"  # HA=22, RB=4
    vf = ValidityFilter(min_heavy_atoms=10, min_rotatable_bonds=1,
                        max_rotatable_bonds=10, banned_atoms=["Se"])
    assert vf.is_valid(smiles) == True

def test_validity_filter_rejects_too_few_heavy_atoms():
    from elite_miner.filters import ValidityFilter
    smiles = "CCO"  # ethanol, HA=3
    vf = ValidityFilter(min_heavy_atoms=10, min_rotatable_bonds=1,
                        max_rotatable_bonds=10, banned_atoms=["Se"])
    assert vf.is_valid(smiles) == False

def test_validity_filter_rejects_selenium():
    from elite_miner.filters import ValidityFilter
    smiles = "c1cc[se]c1"  # selenophene
    vf = ValidityFilter(min_heavy_atoms=1, min_rotatable_bonds=0,
                        max_rotatable_bonds=20, banned_atoms=["Se"])
    assert vf.is_valid(smiles) == False

def test_validity_filter_rejects_boltz_unsafe():
    from elite_miner.filters import ValidityFilter
    # Molecule with >99 atoms would produce atom names > 4 chars
    # Hard to construct a minimal example, so we test the check runs without crashing
    vf = ValidityFilter(min_heavy_atoms=1, min_rotatable_bonds=0,
                        max_rotatable_bonds=20, banned_atoms=["Se"])
    # Normal small molecule should pass
    assert vf.is_valid("c1ccccc1") == True

def test_filter_batch():
    from elite_miner.filters import ValidityFilter
    vf = ValidityFilter(min_heavy_atoms=10, min_rotatable_bonds=1,
                        max_rotatable_bonds=10, banned_atoms=["Se"])

    molecules = [
        ("rxn:5:1:2:3", "c1ccc(-c2cc(-c3ccoc3)ccc2CN2CCC2)cc1"),  # valid
        ("rxn:5:4:5:6", "CCO"),  # too small
        ("rxn:5:7:8:9", "c1ccc(-c2cc(-c3ccoc3)ccc2CN2CCC2)cc1"),  # valid
    ]
    valid = vf.filter_batch(molecules)
    assert len(valid) == 2
    assert valid[0][0] == "rxn:5:1:2:3"
    assert valid[1][0] == "rxn:5:7:8:9"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest elite_miner/tests/test_searcher.py::test_validity_filter_accepts_good_molecule -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# elite_miner/filters.py
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


class ValidityFilter:
    def __init__(self, min_heavy_atoms: int = 10, min_rotatable_bonds: int = 1,
                 max_rotatable_bonds: int = 10, banned_atoms: list[str] = None):
        self.min_heavy_atoms = min_heavy_atoms
        self.min_rotatable_bonds = min_rotatable_bonds
        self.max_rotatable_bonds = max_rotatable_bonds
        self.banned_atoms = set(banned_atoms or [])

    def is_valid(self, smiles: str) -> bool:
        """Check if a molecule passes all validity filters."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Heavy atom count
        if mol.GetNumHeavyAtoms() < self.min_heavy_atoms:
            return False

        # Rotatable bonds
        rb = Descriptors.NumRotatableBonds(mol)
        if rb < self.min_rotatable_bonds or rb > self.max_rotatable_bonds:
            return False

        # Banned atoms
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in self.banned_atoms:
                return False

        # Boltz safety: atom names must be <= 4 chars
        try:
            mol_h = AllChem.AddHs(mol)
            canonical_order = AllChem.CanonicalRankAtoms(mol_h)
            for atom, can_idx in zip(mol_h.GetAtoms(), canonical_order):
                atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
                if len(atom_name) > 4:
                    return False
        except Exception:
            return False

        return True

    def filter_batch(self, molecules: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Filter a list of (mol_name, smiles) tuples, keeping only valid ones."""
        return [(name, smi) for name, smi in molecules if self.is_valid(smi)]
```

**Step 4: Run tests**

Run: `python3 -m pytest elite_miner/tests/test_searcher.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add elite_miner/filters.py elite_miner/tests/test_searcher.py
git commit -m "feat: add ValidityFilter with HA, RB, banned atom, and Boltz-safe checks"
```

---

### Task 4: ProxyScorer - PSICHIC-based Boltz2 Proxy Ranking

**Files:**
- Create: `elite_miner/scorer.py`
- Modify: `elite_miner/tests/test_searcher.py`

**Step 1: Write the failing test**

```python
# Append to elite_miner/tests/test_searcher.py

def test_proxy_scorer_ranks_by_affinity_per_ha():
    from elite_miner.scorer import ProxyScorer

    # Mock molecules with known scores and HA counts
    scored_molecules = [
        ("mol_a", "CCO", 0.8, 3),   # proxy = 0.8/3 = 0.267
        ("mol_b", "CCCCC", 0.9, 5), # proxy = 0.9/5 = 0.180
        ("mol_c", "CC", 0.5, 2),    # proxy = 0.5/2 = 0.250
    ]

    ranked = ProxyScorer.rank_by_proxy(scored_molecules)
    # Highest proxy first: mol_a (0.267), mol_c (0.250), mol_b (0.180)
    assert ranked[0][0] == "mol_a"
    assert ranked[1][0] == "mol_c"
    assert ranked[2][0] == "mol_b"

def test_proxy_scorer_with_antitarget():
    from elite_miner.scorer import ProxyScorer

    molecules = [
        # (name, smiles, target_score, antitarget_score, heavy_atoms)
        ("mol_a", "CCO", 0.8, 0.2, 3),   # combo = (0.8 - 0.9*0.2)/3 = 0.207
        ("mol_b", "CCCCC", 0.9, 0.8, 5), # combo = (0.9 - 0.9*0.8)/5 = 0.036
    ]

    ranked = ProxyScorer.rank_with_antitarget(molecules, antitarget_weight=0.9)
    assert ranked[0][0] == "mol_a"
    assert ranked[1][0] == "mol_b"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest elite_miner/tests/test_searcher.py::test_proxy_scorer_ranks_by_affinity_per_ha -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# elite_miner/scorer.py
from rdkit import Chem


class ProxyScorer:
    """Ranks molecules by a proxy for the Boltz2 scoring formula.

    Real Boltz2: (affinity_probability_binary - affinity_pred_value) / heavy_atom_count
    Our proxy:   (target_psichic - antitarget_weight * antitarget_psichic) / heavy_atom_count

    Both reward high affinity in small molecules.
    """

    @staticmethod
    def rank_by_proxy(
        scored_molecules: list[tuple[str, str, float, int]],
    ) -> list[tuple[str, str, float, int, float]]:
        """Rank molecules by affinity / heavy_atom_count.

        Args:
            scored_molecules: list of (name, smiles, affinity, heavy_atoms)

        Returns:
            Same tuples with proxy score appended, sorted descending.
        """
        with_proxy = []
        for name, smiles, affinity, ha in scored_molecules:
            proxy = affinity / ha if ha > 0 else float("-inf")
            with_proxy.append((name, smiles, affinity, ha, proxy))

        with_proxy.sort(key=lambda x: x[4], reverse=True)
        return with_proxy

    @staticmethod
    def rank_with_antitarget(
        molecules: list[tuple[str, str, float, float, int]],
        antitarget_weight: float = 0.9,
    ) -> list[tuple[str, str, float, float, int, float]]:
        """Rank molecules by (target - weight*antitarget) / heavy_atoms.

        Args:
            molecules: list of (name, smiles, target_score, antitarget_score, heavy_atoms)
            antitarget_weight: penalty multiplier for antitarget affinity

        Returns:
            Same tuples with proxy score appended, sorted descending.
        """
        with_proxy = []
        for name, smiles, target, antitarget, ha in molecules:
            combined = target - antitarget_weight * antitarget
            proxy = combined / ha if ha > 0 else float("-inf")
            with_proxy.append((name, smiles, target, antitarget, ha, proxy))

        with_proxy.sort(key=lambda x: x[5], reverse=True)
        return with_proxy
```

**Step 4: Run tests**

Run: `python3 -m pytest elite_miner/tests/test_searcher.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add elite_miner/scorer.py elite_miner/tests/test_searcher.py
git commit -m "feat: add ProxyScorer for Boltz2 proxy ranking"
```

---

### Task 5: Main Miner Script - `elite_miner/run.py`

**Files:**
- Create: `elite_miner/run.py`

This is the main integration task. It reuses the existing BT infrastructure from `neurons/miner.py` (argument parsing, wallet setup, subtensor connection, DRAND encryption, GitHub submission) but replaces the search loop with our combinatorial strategy.

**Step 1: Write the main miner script**

```python
# elite_miner/run.py
"""
Competitive NOVA SN68 Miner.

Replaces the default random SAVI-2020 search with systematic rxn:5
combinatorial search optimized for Boltz2 scoring.

Usage:
    python3 elite_miner/run.py \
        --wallet.name my_wallet \
        --wallet.hotkey my_hotkey \
        --netuid 68 \
        --logging.info
"""

import os
import sys
import json
import asyncio
import argparse
import datetime
import tempfile
import base64
import hashlib
import traceback

from typing import Any, Dict
from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata
from bittensor.core.errors import MetadataError
from substrateinterface import SubstrateInterface

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from config.config_loader import load_config
from utils import (
    get_sequence_from_protein_code,
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    get_heavy_atom_count,
)
from utils.molecules import molecule_unique_for_protein_hf
from PSICHIC.wrapper import PsichicWrapper
from btdr import QuicknetBittensorDrandTimelock

from elite_miner.searcher import CombinatorialSearcher
from elite_miner.filters import ValidityFilter
from elite_miner.scorer import ProxyScorer

# Path to combinatorial DB
DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")

# File to persist state across restarts
STATE_FILE = os.path.join(BASE_DIR, "elite_miner", "miner_state.json")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default=os.getenv("SUBTENSOR_NETWORK"))
    parser.add_argument("--netuid", type=int, default=68)
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Molecules to generate and score per search cycle")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    config = bt.config(parser)
    config.update(load_config())

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            "elite_miner",
        )
    )
    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    owner = os.environ.get("GITHUB_REPO_OWNER")
    name = os.environ.get("GITHUB_REPO_NAME")
    branch = os.environ.get("GITHUB_REPO_BRANCH")
    path = os.environ.get("GITHUB_REPO_PATH", "")

    if not all([owner, name, branch]):
        raise ValueError("Missing GITHUB_REPO_* environment variables")

    github_path = f"{owner}/{name}/{branch}"
    if path:
        github_path += f"/{path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path exceeds 100 characters")
    return github_path


async def setup_bittensor(config):
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    async with bt.async_subtensor(network=config.network) as subtensor:
        metagraph = await subtensor.metagraph(config.netuid)
        await metagraph.sync()
        miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Miner UID: {miner_uid}")

        node = SubstrateInterface(url=config.network)
        epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value + 1
        bt.logging.info(f"Epoch length: {epoch_length} blocks")

    return wallet, subtensor, metagraph, miner_uid, epoch_length


def load_persistent_state() -> set:
    """Load set of previously submitted molecule names."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                return set(data.get("submitted", []))
        except Exception:
            pass
    return set()


def save_persistent_state(submitted: set):
    """Save submitted molecule names to disk."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"submitted": list(submitted)}, f)
    except Exception as e:
        bt.logging.warning(f"Failed to save state: {e}")


async def submit_response(state: Dict[str, Any]) -> bool:
    """Encrypt and submit the best candidate molecule."""
    candidate = state["candidate_product"]
    if not candidate:
        return False

    bt.logging.info(f"Submitting: {candidate}")

    current_block = await state["subtensor"].get_current_block()
    encrypted = state["bdt"].encrypt(state["miner_uid"], candidate, current_block)

    tmp = tempfile.NamedTemporaryFile(delete=True)
    with open(tmp.name, "w+") as f:
        f.write(str(encrypted))
        f.flush()
        f.seek(0)
        content_str = f.read()
        encoded = base64.b64encode(content_str.encode()).decode()
        filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
        commit_content = f"{state['github_path']}/{filename}.txt"

        try:
            status = await state["subtensor"].set_commitment(
                wallet=state["wallet"],
                netuid=state["config"].netuid,
                data=commit_content,
            )
            bt.logging.info(f"Chain commitment: {status}")
        except MetadataError:
            bt.logging.info("Too soon to commit again.")
            return False

        if status:
            ok = upload_file_to_github(filename, encoded)
            if ok:
                bt.logging.info(f"Submitted successfully: {commit_content}")
                state["last_submitted"] = candidate
                return True
            else:
                bt.logging.error("GitHub upload failed")
    return False


async def search_and_score(state: Dict[str, Any]) -> None:
    """One cycle: generate batch -> filter -> score -> update best."""
    searcher = state["searcher"]
    vfilter = state["vfilter"]
    config = state["config"]
    target_proteins = state["target_proteins"]
    antitarget_proteins = state["antitarget_proteins"]
    psichic_models = state["psichic_models"]

    batch_size = getattr(config, "batch_size", 500)

    # 1. Generate batch of candidate molecules
    raw_batch = searcher.generate_batch(batch_size=batch_size)
    if not raw_batch:
        bt.logging.warning("No more combinations to generate")
        return

    bt.logging.info(f"Generated {len(raw_batch)} raw candidates")

    # 2. Validity filter
    valid_batch = vfilter.filter_batch(raw_batch)
    bt.logging.info(f"After validity filter: {len(valid_batch)}")

    if not valid_batch:
        return

    # 3. Uniqueness filter (check against HF archive)
    unique_batch = []
    for mol_name, smiles in valid_batch:
        if mol_name in state["submitted_names"]:
            continue
        try:
            if molecule_unique_for_protein_hf(config.weekly_target, smiles):
                unique_batch.append((mol_name, smiles))
        except Exception as e:
            bt.logging.warning(f"Uniqueness check failed for {mol_name}: {e}")
            unique_batch.append((mol_name, smiles))

    bt.logging.info(f"After uniqueness filter: {len(unique_batch)}")
    if not unique_batch:
        return

    # 4. PSICHIC scoring against target
    smiles_list = [smi for _, smi in unique_batch]

    target_scores = []
    for protein in target_proteins:
        if protein not in psichic_models:
            seq = get_sequence_from_protein_code(protein)
            model = PsichicWrapper()
            model.run_challenge_start(seq)
            psichic_models[protein] = model

        results_df = psichic_models[protein].run_validation(smiles_list)
        scores = results_df["predicted_binding_affinity"].tolist()
        target_scores.append(scores)

    # Average target scores across all targets
    avg_target = [
        sum(t[i] for t in target_scores) / len(target_scores)
        for i in range(len(smiles_list))
    ]

    # 5. PSICHIC scoring against antitargets
    antitarget_scores = []
    for protein in antitarget_proteins:
        if protein not in psichic_models:
            seq = get_sequence_from_protein_code(protein)
            model = PsichicWrapper()
            model.run_challenge_start(seq)
            psichic_models[protein] = model

        results_df = psichic_models[protein].run_validation(smiles_list)
        scores = results_df["predicted_binding_affinity"].tolist()
        antitarget_scores.append(scores)

    avg_antitarget = [0.0] * len(smiles_list)
    if antitarget_scores:
        avg_antitarget = [
            sum(t[i] for t in antitarget_scores) / len(antitarget_scores)
            for i in range(len(smiles_list))
        ]

    # 6. Rank by proxy score
    molecules_with_scores = []
    for i, (mol_name, smiles) in enumerate(unique_batch):
        ha = get_heavy_atom_count(smiles)
        molecules_with_scores.append(
            (mol_name, smiles, avg_target[i], avg_antitarget[i], ha)
        )

    ranked = ProxyScorer.rank_with_antitarget(
        molecules_with_scores,
        antitarget_weight=config.antitarget_weight,
    )

    # 7. Update best candidate if we found something better
    if ranked:
        best = ranked[0]
        best_name, best_smiles, best_target, best_anti, best_ha, best_proxy = best
        bt.logging.info(
            f"Best this batch: {best_name} | "
            f"target={best_target:.4f} anti={best_anti:.4f} "
            f"HA={best_ha} proxy={best_proxy:.6f}"
        )

        if best_proxy > state["best_proxy_score"]:
            state["best_proxy_score"] = best_proxy
            state["candidate_product"] = best_name
            bt.logging.info(
                f"NEW BEST: {best_name} proxy={best_proxy:.6f}"
            )


async def run_miner(config):
    """Main miner loop."""
    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor(config)

    # Reconnect subtensor for ongoing use
    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()

    state = {
        "config": config,
        "wallet": wallet,
        "subtensor": subtensor,
        "metagraph": metagraph,
        "miner_uid": miner_uid,
        "epoch_length": epoch_length,
        "github_path": load_github_path(),
        "bdt": QuicknetBittensorDrandTimelock(),
        "psichic_models": {},
        "searcher": CombinatorialSearcher(DB_PATH),
        "vfilter": ValidityFilter(
            min_heavy_atoms=config.min_heavy_atoms,
            min_rotatable_bonds=config.min_rotatable_bonds,
            max_rotatable_bonds=config.max_rotatable_bonds,
            banned_atoms=config.banned_atom_types,
        ),
        "target_proteins": [],
        "antitarget_proteins": [],
        "candidate_product": None,
        "best_proxy_score": float("-inf"),
        "last_submitted": None,
        "submitted_names": load_persistent_state(),
    }

    # Load building blocks
    state["searcher"].load_rxn5_building_blocks()
    bt.logging.info(
        f"Loaded {len(state['searcher'].scaffolds)} scaffolds, "
        f"{len(state['searcher'].boronic_acids)} boronic acids"
    )

    # Get initial challenge
    current_block = await subtensor.get_current_block()
    last_boundary = (current_block // epoch_length) * epoch_length
    next_boundary = last_boundary + epoch_length

    if next_boundary - current_block < config.no_submission_blocks:
        bt.logging.info("Too close to epoch end, waiting...")
        await asyncio.sleep(12 * config.no_submission_blocks)
        current_block = await subtensor.get_current_block()
        last_boundary = (current_block // epoch_length) * epoch_length

    block_hash = await subtensor.determine_block_hash(last_boundary)
    challenge = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        weekly_target=config.weekly_target,
        num_antitargets=config.num_antitargets,
    )
    state["target_proteins"] = challenge["targets"]
    state["antitarget_proteins"] = challenge["antitargets"]
    bt.logging.info(
        f"Targets: {state['target_proteins']}, "
        f"Antitargets: {state['antitarget_proteins']}"
    )

    bt.logging.info("Starting main miner loop...")
    last_epoch_block = last_boundary

    while True:
        try:
            current_block = await subtensor.get_current_block()
            current_epoch_start = (current_block // epoch_length) * epoch_length
            next_epoch = current_epoch_start + epoch_length
            blocks_remaining = next_epoch - current_block

            # New epoch detected
            if current_epoch_start != last_epoch_block:
                bt.logging.info(f"New epoch at block {current_epoch_start}")
                last_epoch_block = current_epoch_start

                # Refresh config
                config.update(load_config())
                state["vfilter"] = ValidityFilter(
                    min_heavy_atoms=config.min_heavy_atoms,
                    min_rotatable_bonds=config.min_rotatable_bonds,
                    max_rotatable_bonds=config.max_rotatable_bonds,
                    banned_atoms=config.banned_atom_types,
                )

                # Get new challenge
                block_hash = await subtensor.determine_block_hash(current_epoch_start)
                challenge = get_challenge_params_from_blockhash(
                    block_hash=block_hash,
                    weekly_target=config.weekly_target,
                    num_antitargets=config.num_antitargets,
                )
                state["target_proteins"] = challenge["targets"]
                state["antitarget_proteins"] = challenge["antitargets"]
                bt.logging.info(
                    f"New targets: {state['target_proteins']}, "
                    f"antitargets: {state['antitarget_proteins']}"
                )

                # Reset per-epoch state
                state["candidate_product"] = None
                state["best_proxy_score"] = float("-inf")
                state["last_submitted"] = None

                # Reset searcher iterator so it reshuffles
                state["searcher"]._combo_iter = None

                # Sync metagraph
                metagraph = await subtensor.metagraph(config.netuid)
                state["metagraph"] = metagraph

            # Search phase: generate, filter, score batches until near epoch end
            if blocks_remaining > config.no_submission_blocks + 5:
                await search_and_score(state)
            # Submission phase: submit best candidate near epoch end
            elif (
                blocks_remaining <= config.no_submission_blocks + 5
                and blocks_remaining > config.no_submission_blocks
                and state["candidate_product"]
                and state["candidate_product"] != state["last_submitted"]
            ):
                bt.logging.info(
                    f"Near epoch end ({blocks_remaining} blocks), submitting..."
                )
                success = await submit_response(state)
                if success:
                    state["submitted_names"].add(state["candidate_product"])
                    save_persistent_state(state["submitted_names"])
            else:
                # Waiting for next epoch
                if blocks_remaining <= config.no_submission_blocks:
                    await asyncio.sleep(6)
                else:
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            bt.logging.info("Reconnecting subtensor...")
            subtensor = bt.async_subtensor(network=config.network)
            await subtensor.initialize()
            state["subtensor"] = subtensor
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            bt.logging.info("Shutting down.")
            break
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)


async def main():
    config = parse_arguments()
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Starting elite miner for subnet {config.netuid}")
    await run_miner(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
```

**Step 2: Verify it loads without syntax errors**

Run: `python3 -c "import ast; ast.parse(open('elite_miner/run.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add elite_miner/run.py
git commit -m "feat: add elite miner main loop with combinatorial search strategy"
```

---

### Task 6: Integration Test - End-to-End Search Pipeline

**Files:**
- Modify: `elite_miner/tests/test_searcher.py`

**Step 1: Write integration test**

```python
# Append to elite_miner/tests/test_searcher.py

def test_end_to_end_pipeline():
    """Test the full pipeline: generate -> filter -> score proxy."""
    from elite_miner.searcher import CombinatorialSearcher
    from elite_miner.filters import ValidityFilter
    from elite_miner.scorer import ProxyScorer
    from rdkit import Chem

    searcher = CombinatorialSearcher(DB_PATH)
    searcher.load_rxn5_building_blocks()

    vfilter = ValidityFilter(
        min_heavy_atoms=10,
        min_rotatable_bonds=1,
        max_rotatable_bonds=10,
        banned_atoms=["Se"],
    )

    # Generate
    raw = searcher.generate_batch(batch_size=50)
    assert len(raw) > 0, "No molecules generated"

    # Filter
    valid = vfilter.filter_batch(raw)
    assert len(valid) > 0, "No valid molecules after filtering"

    # Mock scoring (since PSICHIC needs GPU/model)
    import random
    scored = []
    for name, smiles in valid:
        mol = Chem.MolFromSmiles(smiles)
        ha = mol.GetNumHeavyAtoms() if mol else 0
        fake_target = random.uniform(0.3, 0.9)
        fake_anti = random.uniform(0.1, 0.5)
        scored.append((name, smiles, fake_target, fake_anti, ha))

    # Rank
    ranked = ProxyScorer.rank_with_antitarget(scored, antitarget_weight=0.9)
    assert len(ranked) == len(scored)
    # Verify descending proxy score
    for i in range(len(ranked) - 1):
        assert ranked[i][5] >= ranked[i + 1][5], "Not sorted by proxy score"

    best_name = ranked[0][0]
    assert best_name.startswith("rxn:5:"), f"Best molecule has wrong format: {best_name}"
    print(f"Pipeline OK: {len(raw)} generated -> {len(valid)} valid -> best: {best_name}")
```

**Step 2: Run the test**

Run: `python3 -m pytest elite_miner/tests/test_searcher.py::test_end_to_end_pipeline -v`
Expected: PASS

**Step 3: Commit**

```bash
git add elite_miner/tests/test_searcher.py
git commit -m "test: add end-to-end integration test for search pipeline"
```

---

### Task 7: Clean Up Old Elite Miner Files

**Files:**
- Remove stale files from old `elite_miner/` that conflict with new design

**Step 1: Identify files to remove**

The old `elite_miner/` had: `miner.py`, `config.py`, `requirements.txt`, `README.md`,
`data/`, `optimization/`, `scoring/`, `search/`, `submission/`, `wrappers/`.

Our new design replaces all of this with `searcher.py`, `filters.py`, `scorer.py`, `run.py`.

**Step 2: Remove old files**

```bash
# Remove old files that are replaced
rm -f elite_miner/miner.py elite_miner/config.py elite_miner/requirements.txt elite_miner/README.md
rm -rf elite_miner/data elite_miner/optimization elite_miner/scoring \
       elite_miner/search elite_miner/submission elite_miner/wrappers
rm -rf elite_miner/__pycache__ elite_miner/tests/__pycache__
rm -f elite_miner/tests/test_all.py
```

**Step 3: Update `elite_miner/__init__.py`**

```python
# elite_miner/__init__.py
```
(Empty — keep it simple.)

**Step 4: Commit**

```bash
git add -A elite_miner/
git commit -m "chore: remove old elite_miner files, replaced by new design"
```

---

Plan complete and saved to `docs/plans/2026-03-08-competitive-miner.md`.

Two execution options:

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?