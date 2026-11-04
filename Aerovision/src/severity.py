"""
Severity scoring for detected aircraft-skin defects.

The rubric combines three signals into a single severity band:
  1. Relative size of the defect (area of defect / area of image)
  2. Detection confidence
  3. Criticality of the defect type (a crack is treated as more
     urgent than a surface scratch/dent of the same size, corrosion
     sits in between)

This is intentionally simple and explainable — a judge or reviewer
can see exactly why a defect got the severity it did, which matters
more for a hackathon PoC than a black-box score.
"""

from dataclasses import dataclass
from typing import Literal

DefectType = Literal["crack", "corrosion", "dent", "rivet_damage"]

# Relative weight of each defect type in the severity formula.
# Cracks are structurally the most dangerous per unit area.
TYPE_CRITICALITY = {
    "crack": 1.0,
    "rivet_damage": 0.85,
    "corrosion": 0.65,
    "dent": 0.45,
}

RECOMMENDED_ACTION = {
    "HIGH": "Ground inspection required within 24 hrs",
    "MEDIUM": "Schedule detailed inspection within 7 days",
    "LOW": "Log for monitoring at next scheduled check",
}


@dataclass
class SeverityResult:
    score: float          # 0-100
    band: str              # LOW / MEDIUM / HIGH
    recommended_action: str


def compute_severity(
    defect_type: DefectType,
    relative_area: float,
    confidence: float,
) -> SeverityResult:
    """
    relative_area: defect bounding-box area / total image area (0-1)
    confidence: model/detector confidence (0-1)
    """
    criticality = TYPE_CRITICALITY.get(defect_type, 0.5)

    # Normalize relative_area with a soft cap so a single huge
    # detection doesn't blow the scale past 100.
    size_component = min(relative_area * 400, 1.0)  # tuned for typical crop sizes

    raw_score = (0.55 * size_component + 0.45 * confidence) * criticality * 100
    score = round(min(raw_score, 100.0), 1)

    if score >= 65:
        band = "HIGH"
    elif score >= 35:
        band = "MEDIUM"
    else:
        band = "LOW"

    return SeverityResult(
        score=score,
        band=band,
        recommended_action=RECOMMENDED_ACTION[band],
    )
