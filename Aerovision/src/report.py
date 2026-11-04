"""
Rendering helpers: draw detection overlays + a simple explainability
heatmap, and build a tabular inspection report.
"""

from typing import List

import cv2
import numpy as np
import pandas as pd

from .detector import DefectDetection
from .severity import compute_severity

COLOR_MAP = {
    "crack": (60, 60, 240),        # red-ish (BGR)
    "corrosion": (30, 140, 230),    # orange
    "dent": (230, 190, 40),         # blue-ish
    "rivet_damage": (170, 60, 220), # purple
}


def draw_overlays(bgr_image: np.ndarray, detections: List[DefectDetection]) -> np.ndarray:
    """Draw bounding boxes + labels for each detection."""
    out = bgr_image.copy()
    for d in detections:
        x, y, w, h = d.bbox
        color = COLOR_MAP.get(d.defect_type, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        sev = compute_severity(d.defect_type, d.relative_area, d.confidence)
        label = f"{d.defect_type} {int(d.confidence * 100)}% [{sev.band}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, max(0, y - th - 8)), (x + tw + 6, y), color, -1)
        cv2.putText(out, label, (x + 3, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_heatmap(bgr_image: np.ndarray, detections: List[DefectDetection]) -> np.ndarray:
    """
    Simple explainability overlay: a heatmap centered on each detection
    box, intensity scaled by severity score. This is not Grad-CAM (that
    requires a trained deep net) but plays the same explanatory role
    for the classical-CV backend, and is deliberately swappable for a
    real Grad-CAM implementation once a YOLO/CNN model is trained.
    """
    h_img, w_img = bgr_image.shape[:2]
    heat = np.zeros((h_img, w_img), dtype=np.float32)

    for d in detections:
        x, y, w, h = d.bbox
        cx, cy = x + w // 2, y + h // 2
        sev = compute_severity(d.defect_type, d.relative_area, d.confidence)
        intensity = sev.score / 100.0
        axes = (max(w, 10), max(h, 10))
        mask = np.zeros((h_img, w_img), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        heat = np.maximum(heat, mask * intensity)

    heat_uint8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr_image, 0.65, heat_color, 0.35, 0)
    return overlay


def build_report_dataframe(detections: List[DefectDetection]) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(detections, start=1):
        sev = compute_severity(d.defect_type, d.relative_area, d.confidence)
        rows.append({
            "ID": i,
            "Defect Type": d.defect_type.replace("_", " ").title(),
            "Confidence": f"{int(d.confidence * 100)}%",
            "Est. Length (cm)": d.length_cm if d.length_cm else "-",
            "Severity Score": sev.score,
            "Severity Band": sev.band,
            "Recommended Action": sev.recommended_action,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "ID", "Defect Type", "Confidence", "Est. Length (cm)",
            "Severity Score", "Severity Band", "Recommended Action",
        ])
    df = pd.DataFrame(rows)
    return df.sort_values("Severity Score", ascending=False).reset_index(drop=True)
