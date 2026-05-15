"""Tests for submission primitives (no network calls)."""

import os
import pytest

from elite_miner.submission import build_message, encode_for_commit, build_github_path


def test_build_message_both_tracks():
    assert build_message("rxn:1:1:2", "EVQL") == "rxn:1:1:2|EVQL"


def test_build_message_molecule_only():
    assert build_message("rxn:1:1:2", None) == "rxn:1:1:2|"


def test_build_message_nanobody_only():
    assert build_message(None, "EVQL") == "|EVQL"


def test_encode_returns_filename_and_b64():
    # In production, the input is a drand-encrypted object that gets str()-ified.
    # We pass a plain string here to test the encode plumbing.
    fn, encoded = encode_for_commit("hello world")
    assert isinstance(fn, str) and len(fn) == 20
    import base64
    decoded = base64.b64decode(encoded).decode()
    # encode_for_commit does str(input) → "hello world" stays as-is for string input
    assert decoded == "hello world"


def test_encode_deterministic_filename():
    fn1, _ = encode_for_commit(b"same content")
    fn2, _ = encode_for_commit(b"same content")
    assert fn1 == fn2


def test_github_path_missing_env_raises(monkeypatch):
    for k in ("GITHUB_REPO_OWNER", "GITHUB_REPO_NAME", "GITHUB_REPO_BRANCH", "GITHUB_REPO_PATH"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError):
        build_github_path()


def test_github_path_basic(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO_OWNER", "synapz-org")
    monkeypatch.setenv("GITHUB_REPO_NAME", "nova")
    monkeypatch.setenv("GITHUB_REPO_BRANCH", "main")
    monkeypatch.delenv("GITHUB_REPO_PATH", raising=False)
    assert build_github_path() == "synapz-org/nova/main"


def test_github_path_with_subpath(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO_OWNER", "owner")
    monkeypatch.setenv("GITHUB_REPO_NAME", "repo")
    monkeypatch.setenv("GITHUB_REPO_BRANCH", "main")
    monkeypatch.setenv("GITHUB_REPO_PATH", "results")
    assert build_github_path() == "owner/repo/main/results"


def test_github_path_too_long_raises(monkeypatch):
    monkeypatch.setenv("GITHUB_REPO_OWNER", "x" * 50)
    monkeypatch.setenv("GITHUB_REPO_NAME", "y" * 50)
    monkeypatch.setenv("GITHUB_REPO_BRANCH", "z" * 50)
    monkeypatch.setenv("GITHUB_REPO_PATH", "p")
    with pytest.raises(ValueError, match="too long"):
        build_github_path()
