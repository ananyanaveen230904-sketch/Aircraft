"""
Optional second opinion from a vision LLM (Gemini) on a flagged crop.

The model is asked *blind* (it is not told what the detector found) so the
two opinions are independent; agreement is computed afterwards. The second
opinion is advisory: it can raise a "needs human review" flag but never
lowers severity or changes the rule-based risk decision.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .schemas import VisionOpinion, VisionScanReport

PROMPT = """You are inspecting a cropped photo of an aircraft skin panel region.
Decide independently whether it shows surface damage. Choose defect_type from:
crack, corrosion, dent, rivet_damage, other, none.
Return ONLY JSON: {"defect_present": bool, "defect_type": str, "confidence": 0-1, "rationale": "one or two sentences describing only what is visible"}"""

SCAN_PROMPT = """Inspect this image independently. Return JSON only matching this shape:
{"scene_type":"closeup_panel|wide_aircraft|other_object|unclear",
"summary":"brief description of what is visible",
"defects":[{"type":"crack|corrosion|dent|rivet_damage|other","region":"one of upper-left region, upper-center region, upper-right region, center-left region, center region, center-right region, lower-left region, lower-center region, lower-right region","confidence":0.0,"note":"visible evidence only"}],
"overall_concern":"none|low|medium|high","needs_human_review":false}
Report only what is visible in the image. Say unclear when unsure. Do not infer hidden damage or airworthiness."""


def analyze_scan(image: np.ndarray, gemini) -> Dict[str, Any]:
    """Blind whole-image review; advisory only and independent of detector output."""
    try:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {"error": "vision scan failed: image is empty or invalid"}
        h, w = image.shape[:2]
        scale = min(1.0, 1024.0 / max(h, w))
        if scale < 1:
            image = cv2.resize(
                image, (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        quality = 88
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        while ok and len(buf) > 3 * 1024 * 1024 and quality > 40:
            quality -= 12
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return {"error": "vision scan failed: could not encode image"}
        if len(buf) > 3 * 1024 * 1024:
            return {"error": "vision scan failed: encoded image exceeds 3 MB"}
        raw = gemini.generate_with_image(SCAN_PROMPT, buf.tobytes(), json_mode=True)
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
        report = VisionScanReport.model_validate(json.loads(text))
        return report.model_dump()
    except Exception as exc:
        return {"error": f"vision scan failed: {exc}"}

def crop_region(image: np.ndarray, bbox, pad_frac: float = 0.25, min_side: int = 96) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, bw, bh = [int(v) for v in bbox]
    px, py = int(bw * pad_frac) + 8, int(bh * pad_frac) + 8
    x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(w, x + bw + px), min(h, y + bh + py)
    crop = image[y0:y1, x0:x1]
    s = max(1.0, min_side / max(1, min(crop.shape[:2])))
    if s > 1:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return crop


def second_opinion(image: np.ndarray, detection: Any, gemini) -> Dict[str, Any]:
    """`gemini` is a GeminiClient (needs .generate_with_image). Returns a plain dict."""
    bbox = detection["bbox"] if isinstance(detection, dict) else detection.bbox
    dtype = detection["defect_type"] if isinstance(detection, dict) else detection.defect_type
    crop = crop_region(image, bbox)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return {"error": "could not encode crop"}
    try:
        raw = gemini.generate_with_image(PROMPT, buf.tobytes())
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        op = VisionOpinion.model_validate(json.loads(t))
    except Exception as e:
        return {"error": f"vision second opinion failed: {e}"}
    if not op.defect_present or op.defect_type == "none":
        agreement = "disagree"
    elif op.defect_type == dtype:
        agreement = "agree"
    else:
        agreement = "type_disagree"
    return {
        "detector_type": dtype, "vision_defect_present": op.defect_present,
        "vision_type": op.defect_type, "vision_confidence": round(op.confidence, 2),
        "vision_rationale": op.rationale, "agreement": agreement,
        "needs_human_review": agreement != "agree",
        "model": getattr(gemini, "name", "gemini"),
    }
