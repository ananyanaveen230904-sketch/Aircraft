"""
Builds the *grounding record* that the LLM is allowed to talk about.

It extends src.fusion.fuse_inspection_results() (RUL + defects) with the
two things fusion does not carry yet: per-defect severity (src.severity)
and scan-to-scan progression (src.progression). Every defect gets a stable
id (D1, D2, ...) so a generated finding can be traced back to the record.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.fusion import _get_field, fuse_inspection_results
from src.severity import compute_severity

RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
BAND_BY_RANK = {v: k for k, v in RANK.items()}


def build_context(
    detections: Optional[list] = None,
    image_shape: Optional[tuple] = None,
    rul_cycles: Optional[float] = None,
    health_score: Optional[float] = None,
    failure_risk: Optional[str] = None,
    progression: Optional[list] = None,
    scan_id: Optional[str] = None,
    previous_scan_id: Optional[str] = None,
    detection_quality: Optional[Dict[str, Any]] = None,
    vision_review: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    detections: DefectDetection objects or equivalent dicts
                (needs defect_type / confidence / bbox / relative_area).
    progression: list of src.progression.ProgressionResult (or dicts).
    """
    detections = detections or []
    fused = fuse_inspection_results(
        rul_cycles=rul_cycles, health_score=health_score, failure_risk=failure_risk,
        detections=detections, image_shape=image_shape,
    )

    defects: List[Dict[str, Any]] = []
    for i, (det, base) in enumerate(zip(detections, fused["defects"]), start=1):
        d = {"id": f"D{i}", **base}
        rel = _get_field(det, "relative_area")
        conf = _get_field(det, "confidence")
        dtype = _get_field(det, "defect_type")
        if rel is not None and conf is not None and dtype:
            sev = compute_severity(dtype, float(rel), float(conf))
            d["severity_score"] = sev.score
            d["severity_band"] = sev.band
            d["rule_based_action"] = sev.recommended_action
        length = _get_field(det, "length_cm")
        if length:
            d["length_cm"] = length
        defects.append(d)

    prog_rows = []
    for j, p in enumerate(progression or [], start=1):
        g = (lambda n: p.get(n) if isinstance(p, dict) else getattr(p, n))
        prog_rows.append({
            "id": f"P{j}", "type": g("defect_type"), "trend": g("trend"),
            "change_pct": round(float(g("change_pct")), 1),
            "old_area_px": int(g("old_area")), "new_area_px": int(g("new_area")),
        })

    ctx: Dict[str, Any] = {k: v for k, v in fused.items() if k != "defects"}
    ctx["defects"] = defects
    if prog_rows:
        ctx["progression"] = prog_rows
    if scan_id:
        ctx["scan_id"] = scan_id
    if previous_scan_id:
        ctx["previous_scan_id"] = previous_scan_id
    if detection_quality is not None:
        ctx["detection_quality"] = detection_quality
        if detection_quality.get("likely_unreliable"):
            from .config import UNRELIABLE_DETECTION_WARNING
            ctx["warning"] = UNRELIABLE_DETECTION_WARNING
    if vision_review is not None:
        ctx["vision_review"] = vision_review

    bands = [d.get("severity_band") for d in defects if d.get("severity_band")]
    ctx["counts"] = {
        "defects_total": len(defects),
        "high": bands.count("HIGH"), "medium": bands.count("MEDIUM"), "low": bands.count("LOW"),
        **{f"n_{t}": sum(1 for d in defects if d.get("type") == t)
           for t in sorted({d.get("type") for d in defects if d.get("type")})},
        "growing": sum(1 for p in prog_rows if p["trend"] == "growing"),
        "new": sum(1 for p in prog_rows if p["trend"] == "new"),
        "resolved": sum(1 for p in prog_rows if p["trend"] == "resolved"),
    }
    return ctx


def context_to_text(ctx: Dict[str, Any]) -> str:
    """Compact human/LLM-readable rendering of a context (used in prompts)."""
    import json
    return json.dumps(ctx, indent=2, ensure_ascii=False)
