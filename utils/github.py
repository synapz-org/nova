import base64
import json
import os
import sqlite3
from typing import Optional

import requests
import bittensor as bt
from dotenv import load_dotenv

load_dotenv(override=True)


def upload_file_to_github(filename: str, encoded_content: str):
    # Github configs
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')   # example: nova
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH') # example: main
    github_token = os.environ.get('GITHUB_TOKEN')
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER') # example: metanova-labs
    github_repo_path = os.environ.get('GITHUB_REPO_PATH') # example: /data/results or ""

    if not github_repo_name or not github_repo_branch or not github_token or not github_repo_owner:
        raise ValueError("Github environment variables not set. Please set them in your .env file.")

    target_file_path = os.path.join(github_repo_path, f'{filename}.txt')
    url = f"https://api.github.com/repos/{github_repo_owner}/{github_repo_name}/contents/{target_file_path}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        }

    # Check if the file already exists (need its SHA to update)
    existing_file = requests.get(url, headers=headers, params={"ref": github_repo_branch})
    sha = existing_file.json().get("sha") if existing_file.status_code == 200 else None

    payload = {
        "message": f"Encrypted response for {filename}",
        "content": encoded_content,
        "branch": github_repo_branch,
    }
    if sha:
        payload["sha"] = sha  # updating existing file

    response = requests.put(url, headers=headers, json=payload)
    if response.status_code in [200, 201]:
        return True
    else:
        bt.logging.error(f"Failed to upload file for {filename}: {response.status_code} {response.text}")
        return False


# ----------------------------------------------------------------------------
# §PPPPPP — Remote Boltz Cache Persistence
# Upload / download the Boltz score cache as a compact JSON blob to the
# miner's existing GitHub submission repo so it survives container restarts.
# boltz_score_cache.db is gitignored and local-only; every fresh container
# loses the entire cache (surrogate training data, adaptive timing, reaction-
# class weights).  §PPPPPP exports the top-500 entries + miner_state at each
# successful submission and re-imports them on startup so epoch 1 is warm.
# ----------------------------------------------------------------------------

def _github_env() -> tuple:
    """Return (owner, name, branch, token, path) from env vars, or None tuple on miss."""
    return (
        os.environ.get('GITHUB_REPO_OWNER'),
        os.environ.get('GITHUB_REPO_NAME'),
        os.environ.get('GITHUB_REPO_BRANCH'),
        os.environ.get('GITHUB_TOKEN'),
        os.environ.get('GITHUB_REPO_PATH', ''),
    )


def upload_boltz_cache_export(db_path: str, protein: str) -> bool:
    """§PPPPPP: Export top-1000 Boltz cache entries + miner_state to the miner's
    GitHub repo.  Returns True on successful upload, False on any error or missing env.
    §EEEEEEEEEE: JSON is gzip-compressed before base64 encoding, shrinking payload
    by ~65% and allowing 1000 entries (vs 500) within GitHub's 1 MB limit."""
    import gzip as _gzip
    import time as _time

    owner, name, branch, token, path = _github_env()
    if not all([owner, name, branch, token]):
        return False

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute(
            "SELECT smiles, score, affinity_prob_binary, affinity_pred_val, "
            "ligand_iptm, product_name, boltz_le_std, boltz_ww_std, "
            "COALESCE(confidence_score, 1.0) "
            "FROM boltz_cache WHERE protein=? ORDER BY score DESC LIMIT 1000",
            (protein,),
        )
        entries = [
            {"smiles": r[0], "score": r[1], "apb": r[2], "apv": r[3],
             "ligand_iptm": r[4], "product_name": r[5], "le_std": r[6],
             "ww_std": r[7], "conf_score": r[8]}
            for r in c.fetchall()
        ]

        state_out: dict = {}
        for key in ("boltz_time_per_mol", "boltz_trigger_blocks"):
            row = c.execute("SELECT value FROM miner_state WHERE key=?", (key,)).fetchone()
            if row:
                try:
                    state_out[key] = float(row[0])
                except Exception:
                    pass
        for key in ("best_boltz_rxn_class", "rxn_class_scores_json",
                     "salsa_operator_wins_json"):  # §SSSSSSSSSSSS
            row = c.execute(
                "SELECT value_text FROM miner_state WHERE key=?", (key,)
            ).fetchone()
            if row and row[0] is not None:
                state_out[key] = row[0]

        # §RRRRRR: Include top-20 entries for up to 2 prior proteins so that when
        # the weekly target rotates, a fresh container can still seed §WWWWW
        # cross-target search from GitHub history (local SQLite is empty on restart).
        history: dict = {}
        try:
            prior_rows = c.execute(
                "SELECT DISTINCT protein FROM boltz_cache WHERE protein!=? "
                "ORDER BY ts DESC LIMIT 2",
                (protein,),
            ).fetchall()
            for (pp,) in prior_rows:
                c.execute(
                    "SELECT smiles, score, affinity_prob_binary, affinity_pred_val, "
                    "ligand_iptm, product_name, boltz_le_std, boltz_ww_std, "
                    "COALESCE(confidence_score, 1.0) "
                    "FROM boltz_cache WHERE protein=? ORDER BY score DESC LIMIT 20",
                    (pp,),
                )
                history[pp] = [
                    {"smiles": r[0], "score": r[1], "apb": r[2], "apv": r[3],
                     "ligand_iptm": r[4], "product_name": r[5], "le_std": r[6],
                     "ww_std": r[7], "conf_score": r[8]}
                    for r in c.fetchall()
                ]
        except Exception:
            pass

        # §PPPPPPPPPP: Export top-20 Boltz-2 embedding blobs (384D float32 vectors).
        # Enables the embedding-augmented surrogate (§HHHHHHHHHH) to warm-start from
        # epoch 1 on any container restart.  Each vector is 1536 bytes raw; 20 vectors
        # ≈ 30 KB, ~12 KB after gzip — well within the per-export size budget.
        # Backward compatible: old downloaders ignore unknown keys gracefully.
        embeddings = []
        try:
            import numpy as _np_emb
            c.execute(
                "SELECT smiles, boltz_embedding FROM boltz_cache "
                "WHERE protein=? AND boltz_embedding IS NOT NULL "
                "ORDER BY score DESC LIMIT 20",
                (protein,),
            )
            for _esm, _eblob in c.fetchall():
                if not _eblob:
                    continue
                try:
                    _ea = _np_emb.frombuffer(_eblob, dtype=_np_emb.float32)
                    if _ea.shape == (384,):
                        embeddings.append({
                            "smiles": _esm,
                            "emb_b64": base64.b64encode(_eblob).decode(),
                        })
                except Exception:
                    pass
        except Exception:
            pass

        conn.close()

        export = {
            "protein": protein,
            "ts": int(_time.time()),
            "entries": entries,
            "state": state_out,
            "history": history,
            "embeddings": embeddings,  # §PPPPPPPPPP
        }
        # §EEEEEEEEEE: gzip-compress JSON before base64 encoding.
        # JSON is highly compressible (~65-75% reduction); 1000 entries compress to
        # ~40-60 KB — well within GitHub's 1 MB Contents API limit.
        # Downloader detects the gzip magic header for backward compatibility.
        content = base64.b64encode(
            _gzip.compress(json.dumps(export).encode(), compresslevel=6)
        ).decode()

        file_path = "boltz_cache_export.json"
        if path:
            file_path = f"{path}/{file_path}"

        url = f"https://api.github.com/repos/{owner}/{name}/contents/{file_path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }
        existing = requests.get(url, headers=headers, params={"ref": branch})
        sha = existing.json().get("sha") if existing.status_code == 200 else None

        payload = {
            "message": f"cache: Boltz export {protein} ({len(entries)} entries, gzip)",
            "content": content,
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        resp = requests.put(url, headers=headers, json=payload)
        if resp.status_code in (200, 201):
            return True
        bt.logging.warning(
            f"[§PPPPPP] Upload failed: {resp.status_code} {resp.text[:200]}"
        )
        return False
    except Exception as e:
        bt.logging.warning(f"[§PPPPPP] Cache export error: {e}")
        return False


def download_boltz_cache_export(protein: str) -> Optional[dict]:
    """§PPPPPP: Download the Boltz cache export from the miner's GitHub repo.
    Returns the parsed dict on success, or None if not found / protein mismatch / error.
    §EEEEEEEEEE: Detects gzip magic header (\\x1f\\x8b) for new compressed exports;
    falls back to plain JSON decode for legacy uncompressed uploads."""
    import gzip as _gzip

    owner, name, branch, token, path = _github_env()
    if not all([owner, name, branch, token]):
        return None

    try:
        file_path = "boltz_cache_export.json"
        if path:
            file_path = f"{path}/{file_path}"

        url = f"https://api.github.com/repos/{owner}/{name}/contents/{file_path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }
        resp = requests.get(url, headers=headers, params={"ref": branch})
        if resp.status_code != 200:
            return None

        content_b64 = resp.json().get("content", "").replace("\n", "")
        raw = base64.b64decode(content_b64)
        # §EEEEEEEEEE: gzip magic bytes \x1f\x8b indicate compressed payload (new format).
        # Fall back to direct UTF-8 decode for legacy uncompressed exports.
        if raw[:2] == b'\x1f\x8b':
            raw = _gzip.decompress(raw)
        data = json.loads(raw.decode())

        if data.get("protein") != protein:
            bt.logging.info(
                f"[§PPPPPP] Export protein={data.get('protein')!r} "
                f"!= current={protein!r} — main entries skipped; history available."
            )
            # §RRRRRR: return data anyway so caller can process cross-target history.
            data["_protein_matched"] = False
            return data

        data["_protein_matched"] = True
        return data
    except Exception as e:
        bt.logging.warning(f"[§PPPPPP] Cache download error: {e}")
        return None


# ----------------------------------------------------------------------------
# §DDDDDDDDDD — MSA GitHub Cache
# Upload / download compressed MSA (.a3m.gz) files to the miner's GitHub repo
# so that on container restart the ColabFold API round-trip (5–15 min) is
# skipped for already-seen proteins.  Files are gzip-compressed before upload
# to stay within the GitHub Contents API 1 MB payload limit.
# ----------------------------------------------------------------------------

def upload_msa_to_github(protein_code: str, a3m_path: str) -> bool:
    """§DDDDDDDDDD: Upload a gzip-compressed MSA file to GitHub.
    No-ops silently when the file is already present or too large (>700 KB compressed)."""
    import gzip as _gzip

    owner, name, branch, token, path = _github_env()
    if not all([owner, name, branch, token]):
        return False

    try:
        with open(a3m_path, "rb") as fh:
            a3m_bytes = fh.read()

        compressed = _gzip.compress(a3m_bytes, compresslevel=6)
        if len(compressed) > 700_000:
            bt.logging.warning(
                f"[§DDDDDDDDDD] MSA for {protein_code} too large after compression "
                f"({len(compressed):,} bytes) — skipping GitHub upload"
            )
            return False

        file_path = f"msa_cache/{protein_code}.a3m.gz"
        if path:
            file_path = f"{path}/{file_path}"

        url = f"https://api.github.com/repos/{owner}/{name}/contents/{file_path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }

        existing = requests.get(url, headers=headers, params={"ref": branch})
        if existing.status_code == 200:
            bt.logging.debug(
                f"[§DDDDDDDDDD] MSA for {protein_code} already on GitHub — skipping upload"
            )
            return True

        payload = {
            "message": f"cache: MSA for {protein_code}",
            "content": base64.b64encode(compressed).decode(),
            "branch": branch,
        }
        resp = requests.put(url, headers=headers, json=payload)
        if resp.status_code in (200, 201):
            bt.logging.info(
                f"[§DDDDDDDDDD] Uploaded MSA for {protein_code} "
                f"({len(compressed):,} bytes compressed)"
            )
            return True
        bt.logging.warning(
            f"[§DDDDDDDDDD] MSA upload failed: {resp.status_code} {resp.text[:200]}"
        )
        return False
    except Exception as exc:
        bt.logging.warning(f"[§DDDDDDDDDD] MSA upload error: {exc}")
        return False


def download_msa_from_github(protein_code: str, local_path: str) -> bool:
    """§DDDDDDDDDD: Download and decompress a cached MSA from GitHub.
    Writes the .a3m file to local_path and returns True on success."""
    import gzip as _gzip

    owner, name, branch, token, path = _github_env()
    if not all([owner, name, branch, token]):
        return False

    try:
        file_path = f"msa_cache/{protein_code}.a3m.gz"
        if path:
            file_path = f"{path}/{file_path}"

        url = f"https://api.github.com/repos/{owner}/{name}/contents/{file_path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }
        resp = requests.get(url, headers=headers, params={"ref": branch})
        if resp.status_code != 200:
            return False

        content_b64 = resp.json().get("content", "").replace("\n", "")
        a3m_bytes = _gzip.decompress(base64.b64decode(content_b64))

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as fh:
            fh.write(a3m_bytes)

        bt.logging.info(
            f"[§DDDDDDDDDD] Downloaded MSA for {protein_code} from GitHub "
            f"({len(a3m_bytes):,} bytes)"
        )
        return True
    except Exception as exc:
        bt.logging.warning(f"[§DDDDDDDDDD] MSA download error: {exc}")
        return False
