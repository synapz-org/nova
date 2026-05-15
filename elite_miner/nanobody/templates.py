"""Curated VHH templates for nanobody generation.

Each template is a known camelid VHH that passes our local validity filter
(length, AA whitelist, no homopolymers, FR2 hallmark heuristic, etc).

Templates from literature & PDB. CDRs (highlighted by approximate raw
positions) are the natural mutation hotspots; the framework regions should
stay close to template to preserve VHH-ness.

Approximate CDR positions (raw, not Kabat) — varies by template:
    CDR1: ~26-35
    CDR2: ~50-65
    CDR3: ~95-115 (most variable, most consequential for binding)

Adding more templates is cheap — we keep them as a list for round-robin
sampling in the generator.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VHHTemplate:
    name: str
    sequence: str
    cdr1_range: tuple[int, int]  # (start, end_exclusive) raw positions
    cdr2_range: tuple[int, int]
    cdr3_range: tuple[int, int]
    notes: str = ""


# CDR ranges are conservative estimates; we mutate within them.
# Coordinates were aligned by visual inspection of the WVRQAPG / WGQG framework anchors.
TEMPLATES: list[VHHTemplate] = [
    VHHTemplate(
        name="reference_baseline",
        sequence=(
            "EVELLASGGDLVQPGGSLRLSCAASGFTFSTYAMSWVRQAPGKGLERVSRVNQNGGTTTYADAMKGR"
            "FTISRDNAKNTLYLQMINVKPEDTAIYYCARWDGGSWSTDPWGRGTLVTVS"
        ),
        cdr1_range=(26, 35),
        cdr2_range=(50, 65),
        cdr3_range=(95, 115),
        notes="Reference miner's base template. 118aa, passes validator filters.",
    ),
    VHHTemplate(
        name="anti_EGFR_7D12",
        sequence=(
            "QVQLVESGGGLVQPGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAGISWSGGSTYYTDSVKGR"
            "FTISRDNAKNTLYLQMNSLKPEDTAVYYCAAAGSAWYGTLYEYDYWGQGTQVTVSS"
        ),
        cdr1_range=(26, 35),
        cdr2_range=(50, 65),
        cdr3_range=(95, 120),
        notes="Anti-EGFR 7D12 (PDB 4KRL). Classic camelid VHH framework.",
    ),
    VHHTemplate(
        name="humanized_VHH_C32",
        sequence=(
            "QVQLVESGGGLVQAGGSLRLSCAASERTFRRYNMGWFRQAPGKEREFVATISWSGGITYYADSVKGR"
            "FTISRDNAKNTLYLQMNSLKPEDTAVYYCAAQYGSRYPGRLYDYWGQGTQVTVSS"
        ),
        cdr1_range=(26, 35),
        cdr2_range=(50, 65),
        cdr3_range=(95, 119),
        notes="Humanized VHH C32-style framework. Strong human framework score.",
    ),
    VHHTemplate(
        name="anti_CD38_iVHH",
        sequence=(
            "EVQLQASGGGLVQPGGSLRLSCAASGRTLSDYAMGWFRQAPGKEREFVAAIYWSGGYAYYADSVKGR"
            "FTISRDNAKNTVYLQMNSLKPEDTAVYYCAAATTPYTGSARFFNRYRSDYDYWGQGTQVTVSS"
        ),
        cdr1_range=(26, 35),
        cdr2_range=(50, 65),
        cdr3_range=(95, 127),
        notes="Anti-CD38 iVHH. Longer CDR3 — good for novel targets.",
    ),
]


def list_templates() -> list[VHHTemplate]:
    return list(TEMPLATES)


def get_template(name: str) -> VHHTemplate:
    for t in TEMPLATES:
        if t.name == name:
            return t
    raise KeyError(f"Unknown template: {name}")
