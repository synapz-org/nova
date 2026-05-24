import requests
from typing import List, Optional

_CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
_TIMEOUT = 30


def uniprot_to_chembl_target(uniprot_id: str) -> Optional[str]:
    """Return the ChEMBL target ID for a UniProt accession, or None if not found."""
    try:
        resp = requests.get(
            f"{_CHEMBL_BASE}/target.json",
            params={"target_components__accession": uniprot_id, "limit": 1},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        targets = resp.json().get("targets", [])
        return targets[0]["target_chembl_id"] if targets else None
    except Exception:
        return None


def fetch_chembl_actives(
    chembl_target_id: str,
    pchembl_threshold: float = 7.0,
    limit: int = 200,
) -> List[str]:
    """Return unique canonical SMILES of binding actives (pChEMBL >= threshold)."""
    try:
        resp = requests.get(
            f"{_CHEMBL_BASE}/activity.json",
            params={
                "target_chembl_id": chembl_target_id,
                "pchembl_value__gte": pchembl_threshold,
                "assay_type": "B",
                "limit": limit,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        seen: set = set()
        result: List[str] = []
        for act in resp.json().get("activities", []):
            smi = act.get("canonical_smiles")
            if smi and smi not in seen:
                seen.add(smi)
                result.append(smi)
        return result
    except Exception:
        return []


def get_chembl_seeds(uniprot_id: str, limit: int = 200) -> List[str]:
    """Fetch known actives from ChEMBL for a UniProt target.

    Returns an empty list gracefully on any network or parsing failure.
    Intended for use with asyncio.to_thread at miner startup so it doesn't
    block the event loop.
    """
    target_id = uniprot_to_chembl_target(uniprot_id)
    if not target_id:
        return []
    return fetch_chembl_actives(target_id, limit=limit)
