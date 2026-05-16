"""Protein target featurization for the Boltz2 surrogate.

Primary: ESM2-T12 mean-pooled embedding (320d) — lazy-loaded, cached per target.
Fallback: lightweight sequence-based features (AA composition + properties, ~25d)
so the codebase works without ESM2 installed (useful for tests, dev, label collection
on small machines).

Cache layout:
    cache/protein_features/{esm2|seqstat}/{uniprot}.npy
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


# AA composition + physicochemical bucket counts. ~25 features, fully sequence-derived.
HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQH")
CHARGED_POS = set("KR")
CHARGED_NEG = set("DE")
AROMATIC = set("FYW")
SMALL = set("AGS")
ALL_AAS = "ACDEFGHIKLMNPQRSTVWY"


def _seq_features(sequence: str) -> np.ndarray:
    """Lightweight 25d protein features: per-AA frequency (20) + bucket fractions (5)."""
    if not sequence:
        return np.zeros(25, dtype=np.float32)
    n = len(sequence)
    counts = {aa: 0 for aa in ALL_AAS}
    for aa in sequence:
        if aa in counts:
            counts[aa] += 1
    aa_freq = np.array([counts[aa] / n for aa in ALL_AAS], dtype=np.float32)

    hydro = sum(1 for c in sequence if c in HYDROPHOBIC) / n
    polar = sum(1 for c in sequence if c in POLAR) / n
    pos = sum(1 for c in sequence if c in CHARGED_POS) / n
    neg = sum(1 for c in sequence if c in CHARGED_NEG) / n
    arom = sum(1 for c in sequence if c in AROMATIC) / n
    buckets = np.array([hydro, polar, pos, neg, arom], dtype=np.float32)

    return np.concatenate([aa_freq, buckets])


def _esm2_features(sequence: str) -> np.ndarray:
    """ESM2-T12 mean-pooled 320d embedding. Lazy import — only triggered when called."""
    import torch
    from transformers import AutoTokenizer, AutoModel  # noqa: lazy import

    if not hasattr(_esm2_features, "_model"):
        tok = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        mdl = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
        mdl.eval()
        _esm2_features._tok = tok
        _esm2_features._model = mdl

    tok = _esm2_features._tok
    mdl = _esm2_features._model
    with torch.no_grad():
        # Truncate to 1024 — ESM2 limit, mean-pool over real residues
        seq = sequence[:1024]
        toks = tok(seq, return_tensors="pt")
        out = mdl(**toks)
        # last_hidden_state: (1, seqlen, 320); drop CLS/SEP, mean-pool over residues
        h = out.last_hidden_state[0, 1:-1].mean(dim=0).cpu().numpy().astype(np.float32)
    return h


def target_features(
    uniprot: str,
    sequence: Optional[str] = None,
    *,
    method: str = "seqstat",
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """Featurize a protein target.

    method='seqstat' (default): 25d sequence statistics. Always available.
    method='esm2': 320d ESM2-T12 mean pool. Requires transformers + torch.

    `sequence` may be omitted if cached.
    """
    cache_dir = cache_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "cache", "protein_features", method,
    )
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{uniprot}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    if sequence is None:
        raise ValueError(f"sequence required for uncached target {uniprot}")

    if method == "seqstat":
        v = _seq_features(sequence)
    elif method == "esm2":
        v = _esm2_features(sequence)
    else:
        raise ValueError(f"unknown method {method!r}; use 'seqstat' or 'esm2'")

    np.save(cache_path, v)
    return v


def feature_dim(method: str = "seqstat") -> int:
    return {"seqstat": 25, "esm2": 320}[method]
