"""
MSA auto-fetch utility for Boltz-2 predictions.

Fetches multiple-sequence alignments (MSAs) from the ColabFold/MMseqs2 API for a
given protein sequence and saves them to boltz/msa_files/{protein_code}.a3m.

Without a pre-computed MSA, Boltz-2 runs in single-sequence mode, which produces
weaker affinity predictions. This module eliminates the manual step required whenever
the weekly target protein rotates.

Typical usage in miner startup:
    from utils.msa import ensure_msa
    seq = get_sequence_from_protein_code(config.weekly_target)
    ensure_msa(config.weekly_target, seq)   # no-op if file already exists
"""

import io
import os
import tarfile
import time
import zipfile

import requests
import bittensor as bt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLABFOLD_API = "https://api.colabfold.com"
_POLL_INTERVAL_S = 15   # seconds between status polls
_SUBMIT_TIMEOUT_S = 30  # HTTP timeout for job submission
_DOWNLOAD_TIMEOUT_S = 120  # HTTP timeout for result download
_MAX_WAIT_S = 900       # 15 minutes — long proteins can take a while

MSA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "boltz", "msa_files")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _msa_path(protein_code: str) -> str:
    return os.path.join(MSA_DIR, f"{protein_code}.a3m")


def _extract_a3m(raw_bytes: bytes) -> bytes | None:
    """Extract the first .a3m file from a tar.gz or zip archive."""
    # Try gzip tarball first (ColabFold API default)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:gz") as tf:
            a3m_members = [m for m in tf.getmembers() if m.name.endswith(".a3m")]
            if a3m_members:
                return tf.extractfile(a3m_members[0]).read()
    except Exception:
        pass

    # Fallback: zip archive
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            a3m_names = [n for n in zf.namelist() if n.endswith(".a3m")]
            if a3m_names:
                return zf.read(a3m_names[0])
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def msa_exists(protein_code: str) -> bool:
    """Return True if the MSA file for protein_code is already on disk."""
    return os.path.isfile(_msa_path(protein_code))


def fetch_msa(protein_code: str, sequence: str) -> bool:
    """
    Fetch an MSA for *sequence* from the ColabFold API and write it to
    boltz/msa_files/{protein_code}.a3m.

    Returns True on success, False on any failure (network, timeout, parse).
    All errors are logged; the caller should warn the user if this returns False.
    """
    out_path = _msa_path(protein_code)
    os.makedirs(MSA_DIR, exist_ok=True)

    bt.logging.info(f"[MSA] Fetching MSA for {protein_code} via ColabFold API ...")

    # 1. Submit job
    try:
        resp = requests.post(
            f"{COLABFOLD_API}/ticket/msa",
            json={"q": f">query\n{sequence}", "mode": "env"},
            timeout=_SUBMIT_TIMEOUT_S,
        )
        resp.raise_for_status()
        job_id = resp.json()["id"]
        bt.logging.info(f"[MSA] Job submitted: {job_id}")
    except Exception as exc:
        bt.logging.error(f"[MSA] Failed to submit job for {protein_code}: {exc}")
        return False

    # 2. Poll until COMPLETE, ERROR, or timeout
    deadline = time.time() + _MAX_WAIT_S
    while time.time() < deadline:
        try:
            poll_resp = requests.get(
                f"{COLABFOLD_API}/ticket/{job_id}", timeout=_SUBMIT_TIMEOUT_S
            )
            poll_resp.raise_for_status()
            status = poll_resp.json().get("status", "")
            if status == "COMPLETE":
                bt.logging.info(f"[MSA] Job {job_id} complete.")
                break
            if status in ("ERROR", "FAILED"):
                bt.logging.error(f"[MSA] Job {job_id} failed with status: {status}")
                return False
            bt.logging.debug(f"[MSA] Job {job_id} status={status!r}, waiting {_POLL_INTERVAL_S}s ...")
        except Exception as exc:
            bt.logging.warning(f"[MSA] Poll error (will retry): {exc}")
        time.sleep(_POLL_INTERVAL_S)
    else:
        bt.logging.error(f"[MSA] Job {job_id} timed out after {_MAX_WAIT_S}s")
        return False

    # 3. Download result archive
    try:
        dl_resp = requests.get(
            f"{COLABFOLD_API}/result/download/{job_id}",
            timeout=_DOWNLOAD_TIMEOUT_S,
        )
        dl_resp.raise_for_status()
        raw = dl_resp.content
    except Exception as exc:
        bt.logging.error(f"[MSA] Download failed for job {job_id}: {exc}")
        return False

    # 4. Extract .a3m from archive
    a3m_bytes = _extract_a3m(raw)
    if not a3m_bytes:
        bt.logging.error(
            f"[MSA] Could not find a .a3m file in the result archive for {protein_code}"
        )
        return False

    # 5. Write to disk
    try:
        with open(out_path, "wb") as fh:
            fh.write(a3m_bytes)
        bt.logging.info(
            f"[MSA] Saved {out_path} ({len(a3m_bytes):,} bytes)"
        )
        return True
    except Exception as exc:
        bt.logging.error(f"[MSA] Failed to write {out_path}: {exc}")
        return False


def ensure_msa(protein_code: str, sequence: str) -> bool:
    """
    Ensure an MSA file exists for protein_code.

    If it is already on disk, return True immediately (no network call).
    Otherwise attempt to fetch it via fetch_msa().
    Returns True if the MSA is available after this call, False otherwise.
    """
    if msa_exists(protein_code):
        bt.logging.debug(f"[MSA] {protein_code}.a3m already on disk — skipping fetch.")
        return True
    return fetch_msa(protein_code, sequence)
