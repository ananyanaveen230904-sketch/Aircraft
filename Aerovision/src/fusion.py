"""
Decision Fusion Module (Stage 3).

Combines Stage 1 (LSTM RUL prediction, src/rul/predict.py) and Stage 2
(YOLO/classical defect detection, src/detector.py) outputs into one
structured record, which becomes the input to Stage 4 (T5/BART report
generation) later.

Pure data-reshaping — no new dependencies, no changes to detector.py,
severity.py, progression.py, report.py, app.py, or src/rul/.

Design notes (confirmed before implementation):
- "location" is a positional description derived from the detection's
  bounding box (3x3 grid over the image: e.g. "upper-left region",
  "center"), NOT an invented anatomical part name — detector.py has no
  way to know which physical aircraft part is in frame, so naming one
  would be false precision.
- Partial records are supported: RUL/health/defect-detection are two
  independent AeroVision stages, so fusion doesn't force both to be
  present. Missing fields are simply OMITTED from the output dict
  (never emitted as null/fake defaults), and to_prompt_text() skips
  the corresponding sentence gracefully. Whether to require both
  before showing results is left as a dashboard/UI decision, not
  enforced here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

# A detection can be a real detector.py DefectDetection object, or a
# plain dict with the same field names (e.g. for tests, or if a
# detection came from JSON/serialization) — both are accepted.
DetectionLike = Union[Any, Dict[str, Any]]

_RISK_LABELS = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}

# 3x3 positional grid labels, indexed [row][col], row/col in {0, 1, 2}
_GRID_LABELS = [
    ["upper-left region", "upper-center region", "upper-right region"],
    ["center-left region", "center region", "center-right region"],
    ["lower-left region", "lower-center region", "lower-right region"],
]


def _get_field(detection: DetectionLike, name: str, default=None):
    """Read a field off either a DefectDetection object or a plain dict."""
    if isinstance(detection, dict):
        return detection.get(name, default)
    return getattr(detection, name, default)


def _bbox_to_grid_location(bbox, image_shape) -> Optional[str]:
    """
    Map a detection's bbox center to a 3x3 positional grid label, e.g.
    "upper-left region". Returns None if bbox or image_shape isn't
    available — no location is better than a wrong/guessed one.

    bbox: (x, y, w, h) in pixels (top-left corner + size)
    image_shape: (height, width) of the source image
    """
    if not bbox or not image_shape:
        return None
    try:
        x, y, w, h = bbox
        img_h, img_w = image_shape[0], image_shape[1]
        if img_w <= 0 or img_h <= 0:
            return None
        cx, cy = x + w / 2.0, y + h / 2.0
        col = min(2, max(0, int((cx / img_w) * 3)))
        row = min(2, max(0, int((cy / img_h) * 3)))
        return _GRID_LABELS[row][col]
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _format_defect(detection: DetectionLike, image_shape) -> Dict[str, Any]:
    defect_type = _get_field(detection, "defect_type")
    confidence = _get_field(detection, "confidence")
    bbox = _get_field(detection, "bbox")

    out: Dict[str, Any] = {}
    if defect_type is not None:
        out["type"] = defect_type
    if confidence is not None:
        out["confidence"] = round(float(confidence), 2)

    location = _bbox_to_grid_location(bbox, image_shape)
    if location is not None:
        out["location"] = location
    # No location key at all if it can't be derived — avoids implying
    # false precision, per the "don't fabricate" decision above.

    return out


def fuse_inspection_results(
    rul_cycles: Optional[float] = None,
    health_score: Optional[float] = None,
    failure_risk: Optional[str] = None,
    detections: Optional[List[DetectionLike]] = None,
    image_shape: Optional[tuple] = None,
) -> Dict[str, Any]:
    """
    Combine Stage 1 (RUL) and Stage 2 (defect detection) outputs into
    one structured record.

    All Stage 1 args are optional and independent of each other and of
    `detections` — pass only what you have. Any that are None are
    simply omitted from the output dict (never emitted as null/0/fake
    defaults).

    Args:
        rul_cycles: predicted RUL in cycles, from src.rul.predict.predict_rul()
        health_score: 0-100, from src.rul.predict.health_score()
        failure_risk: "HIGH" | "MEDIUM" | "LOW", from
            src.rul.predict.failure_risk_band() (case-insensitive on input;
            output is normalized to "High"/"Medium"/"Low" to match the
            target format's capitalization)
        detections: list of DefectDetection objects (or equivalent dicts)
            from src.detector.detect(). An empty list or None both mean
            "no defects" and produce defects: [] — never an error.
        image_shape: (height, width) of the inspected image, used only
            to derive each defect's positional "location". If omitted,
            defects are still included, just without a "location" key.

    Returns:
        dict matching the target format, e.g.:
        {
            "rul_cycles": 18,
            "health_score": 63,
            "failure_risk": "Medium",
            "defects": [
                {"type": "crack", "confidence": 0.94, "location": "center region"},
            ],
        }
        (rul_cycles/health_score/failure_risk keys are simply absent if
        their inputs were None; "defects" is always present, possibly [].)
    """
    record: Dict[str, Any] = {}

    if rul_cycles is not None:
        record["rul_cycles"] = round(float(rul_cycles), 1) if not float(rul_cycles).is_integer() else int(rul_cycles)
    if health_score is not None:
        record["health_score"] = round(float(health_score), 1) if not float(health_score).is_integer() else int(health_score)
    if failure_risk is not None:
        record["failure_risk"] = _RISK_LABELS.get(str(failure_risk).upper(), str(failure_risk))

    detections = detections or []
    record["defects"] = [_format_defect(d, image_shape) for d in detections]

    return record


def to_prompt_text(fused_record: Dict[str, Any]) -> str:
    """
    Convert a fused record (from fuse_inspection_results) into a
    natural-language prompt string for Stage 4 (T5/BART report
    generation). Deliberately simple/template-based for now — Stage 4
    isn't built yet, so this just needs to be a stable, parseable
    format that won't need to be rewritten later.

    Gracefully skips any sentence whose underlying data is missing
    (partial records), rather than saying e.g. "health: unknown".
    """
    parts: List[str] = []

    if "health_score" in fused_record:
        parts.append(f"Engine health: {fused_record['health_score']}%.")
    if "rul_cycles" in fused_record:
        parts.append(f"Estimated remaining useful life: {fused_record['rul_cycles']} cycles.")
    if "failure_risk" in fused_record:
        parts.append(f"Failure risk: {fused_record['failure_risk']}.")

    defects = fused_record.get("defects", [])
    if defects:
        defect_strs = []
        for d in defects:
            dtype = d.get("type", "unknown defect")
            conf = d.get("confidence")
            loc = d.get("location")

            s = dtype
            if conf is not None:
                s += f" ({round(conf * 100)}% confidence)"
            if loc:
                s += f" in {loc}"
            defect_strs.append(s)
        parts.append("Detected defects: " + ", ".join(defect_strs) + ".")
    else:
        parts.append("Detected defects: none.")

    return " ".join(parts)
