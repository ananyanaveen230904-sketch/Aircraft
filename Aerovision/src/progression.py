"""
Defect progression tracking.

Compares detections from two scans of (roughly) the same panel taken
at different times and reports whether matched defects have grown,
shrunk, or stayed stable. Matching is done by proximity of bounding
box centers + same defect type — good enough for a PoC where the
camera angle is assumed to be roughly consistent between scans.
"""

from dataclasses import dataclass
from typing import List, Optional

from .detector import DefectDetection


@dataclass
class ProgressionResult:
    defect_type: str
    old_area: float
    new_area: float
    change_pct: float
    trend: str  # "growing" | "shrinking" | "stable" | "new" | "resolved"


def _center(bbox):
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def _dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def compare_scans(
    old_detections: List[DefectDetection],
    new_detections: List[DefectDetection],
    match_radius_px: float = 60.0,
) -> List[ProgressionResult]:
    results: List[ProgressionResult] = []
    used_new = set()

    for old in old_detections:
        best_match: Optional[DefectDetection] = None
        best_dist = match_radius_px
        for i, new in enumerate(new_detections):
            if i in used_new or new.defect_type != old.defect_type:
                continue
            d = _dist(_center(old.bbox), _center(new.bbox))
            if d < best_dist:
                best_dist = d
                best_match = new
                best_match_idx = i

        if best_match is not None:
            used_new.add(best_match_idx)
            old_area = old.bbox[2] * old.bbox[3]
            new_area = best_match.bbox[2] * best_match.bbox[3]
            change_pct = round(((new_area - old_area) / max(old_area, 1)) * 100, 1)
            if change_pct > 15:
                trend = "growing"
            elif change_pct < -15:
                trend = "shrinking"
            else:
                trend = "stable"
            results.append(ProgressionResult(
                defect_type=old.defect_type,
                old_area=old_area,
                new_area=new_area,
                change_pct=change_pct,
                trend=trend,
            ))
        else:
            results.append(ProgressionResult(
                defect_type=old.defect_type,
                old_area=old.bbox[2] * old.bbox[3],
                new_area=0,
                change_pct=-100.0,
                trend="resolved",
            ))

    for i, new in enumerate(new_detections):
        if i not in used_new:
            results.append(ProgressionResult(
                defect_type=new.defect_type,
                old_area=0,
                new_area=new.bbox[2] * new.bbox[3],
                change_pct=100.0,
                trend="new",
            ))

    return results
