"""Tests for the reaction catalog wrapper."""

import os
import pytest

from elite_miner.molecule.reactions import (
    ReactionCatalog,
    ReactionInfo,
    parse_allowed_reaction,
)

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "combinatorial_db", "molecules.sqlite"))


def test_db_exists():
    assert os.path.exists(DB_PATH), f"combinatorial DB not found at {DB_PATH}"


def test_catalog_loads_all_reactions():
    cat = ReactionCatalog(DB_PATH)
    ids = cat.all_ids()
    assert ids == [1, 2, 3, 4, 5], f"expected reactions 1-5, got {ids}"


def test_reaction_info_has_mol_ids():
    cat = ReactionCatalog(DB_PATH)
    for rid in cat.all_ids():
        info = cat[rid]
        assert isinstance(info, ReactionInfo)
        assert len(info.mol_ids_a) > 0
        assert len(info.mol_ids_b) > 0
        if info.is_three_component:
            assert len(info.mol_ids_c) > 0
        assert info.space_size > 0


def test_singleton_get():
    cat1 = ReactionCatalog.get(DB_PATH)
    cat2 = ReactionCatalog.get(DB_PATH)
    assert cat1 is cat2, "ReactionCatalog.get should return singleton"


def test_parse_allowed_reaction():
    assert parse_allowed_reaction("rxn:3") == 3
    assert parse_allowed_reaction("rxn:5") == 5
    assert parse_allowed_reaction("3") == 3
    assert parse_allowed_reaction("") is None
    assert parse_allowed_reaction(None) is None
    assert parse_allowed_reaction("not a number") is None
