"""VHH templates for nanobody generation.

After empirical analysis of Metanova/Submission-Archive (May 2026):
- All top-10% winners for Q9NZQ7 are length 120
- They cluster around a single parent VHH scaffold with CDR mutations
- design_iiptm range for top: 0.81-0.83 vs our local generated max of 0.30

So we use the actual archive winners as templates. The generator's mutation
strategy stays the same (template + CDR mutation), but now the starting
point is in the winning neighborhood instead of literature VHHs from unrelated
targets.

The first ~10 templates are the top archive winners for Q9NZQ7. CDR ranges
are derived from alignment against framework anchors (WVRQ, WGQG/HGQG).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VHHTemplate:
    name: str
    sequence: str
    cdr1_range: tuple[int, int]
    cdr2_range: tuple[int, int]
    cdr3_range: tuple[int, int]
    notes: str = ""


# Top winners pulled from Metanova/Submission-Archive on 2026-05-16.
# All are length 120, design_iiptm 0.81-0.83 (top 1% of historical submissions).
# CDR ranges aligned by framework anchors:
#   CDR1 ≈ pos 26-35  (between SCAA and WVRQ)
#   CDR2 ≈ pos 50-58  (between WV and ADF/AKG)
#   CDR3 ≈ pos 96-108 (between YYC and HGQG)
ARCHIVE_TOP_WINNERS = [
    ("archive_iiptm_8262",
     "DVTLSESGGGLVQAGGSLRLSCAASGFTYSSYAMSWVRQRAGKEREWVSDVTSSGDQTDYADFVKGRFTISRDNAKQTLYLQMTSLRPEDTAVYYCSKWPYRFMKQYASHGQGTTVTVSS"),
    ("archive_iiptm_8251",
     "DVTLAESGGGLVQAGGSLKLSCAATGFTFSSYAMSWVRQRPGKQREWVSDITSQGDQTDYADFVKGRFTISRDNAKNTLYLQMTSLRPEDTAVYYCSKFPYRFMPQYAQHSQGTTVTVSA"),
    ("archive_iiptm_8218",
     "DVTLVESGGGLVQAGGSLRLSCAATGFTYSSWAMSWVRQAPGKEREWVSDISSQGDQTDYADWVKGRFTISRDQAKNTLYLQMTSLRPEDTAVYYCSKFPYRFLPQFAQHGQGTTVTVSA"),
    ("archive_iiptm_8211",
     "DVTIAESGGGLVQAGGSLRMSCAATGFTFSSYSMSWVRQRPGKQREWVSDISSQGDQTDYADFVKGRFTISRDNAKNTLYLQMSSLRPEDTAVYYCSKFPWRFMPQFAQHGQGTTVTVTA"),
    ("archive_iiptm_8206",
     "DVSLAESGGGLVQAGGSLKLSCAASGFTYSSYAMSWVRQRPGKEREWVSDITSQGDQTDYADFVKGRFTISRDSAKNTLYLQMTSLRPEDTAVYYCSKWPYRFMPQYAQHGQGTTVTVSA"),
    ("archive_iiptm_8204",
     "DVVISESGGGLVQAGGSLRLSCAATGFTYSSYAMSWVRQRAGKEREWVSDITSQGDQTDYADFVKGRFTISRDNAKNTLYLQMTSLRPEDTAVYYCSKYPYRFDQARAQHGQGNTVTVSA"),
    ("archive_iiptm_8192",
     "DVTLVESGGGLVTAGGSLRLSCAATGFTYSSYAMSWVRQRPGKEREWVSDISSQGDQTDYADFVKGRFTISRDNAKNTLYLQMTSLRPEDTAVYYCSKFPWRFMPQFAQHGQGTTVTVSA"),
    ("archive_iiptm_8190",
     "DVVMAESGGGLVQAGGSLRLSCAASGFTYTSYAMSWVRQRPGKEREWVSDVTSQGDQTDYADFVKGRFTISRDNAKNTLYLQMTSLRPEDTAVYYCSKFPWRFMQTFAQHGQGTTVTVSA"),
    ("archive_iiptm_8185",
     "DVVMAESGGGLVQAGGSLRLSCAASGFTFTSYAMSWVRQRPGKEREWVSDLTSQGDQTDYADFVKGRFTISRDNAKNTLYLQMTSLRPEDTAVYYCSKFPWRFMQTFAQHGQGTTVTVSA"),
    ("archive_iiptm_8181",
     "DVLMAESGGGLVQAGGSLRLSCAATGFTYSSYAMSWVRQRAGKEREWVSDITSQGDQTDYADFVKGRFTISRDNAKNTLYLQMTSLRPEDTAVYYCSKFPWRFMQTFAQHGQGTTVTVSA"),
]


TEMPLATES: list[VHHTemplate] = [
    VHHTemplate(
        name=name,
        sequence=seq,
        cdr1_range=(26, 35),
        cdr2_range=(50, 58),
        cdr3_range=(96, 108),
        notes="Top archive winner for Q9NZQ7. Length 120, mutate CDRs.",
    )
    for name, seq in ARCHIVE_TOP_WINNERS
]


def list_templates() -> list[VHHTemplate]:
    return list(TEMPLATES)


def get_template(name: str) -> VHHTemplate:
    for t in TEMPLATES:
        if t.name == name:
            return t
    raise KeyError(f"Unknown template: {name}")
