"""
AeroVision defect detector.

Two backends are supported:

1. "classical" (default, always available, zero external downloads)
   A lightweight OpenCV heuristic pipeline: edge/contour analysis +
   HSV color analysis to flag likely cracks, corrosion, dents and
   rivet-damage regions. This is what runs out-of-the-box so the
   whole prototype is demoable with no training, no internet, and
   no GPU.

2. "yolo" (optional, higher accuracy — bring your own trained weights)
   If `ultralytics` is installed AND a trained weights file exists at
   models/best.pt, the detector automatically switches to real YOLOv8
   inference. See train.py + README.md for how to train this.

Both backends return the same DefectDetection objects so the rest of
the app (severity scoring, reporting, dashboard) doesn't care which
one produced them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
YOLO_WEIGHTS_PATH = os.path.join(MODELS_DIR, "best.pt")


@dataclass
class DefectDetection:
    defect_type: str          # crack | corrosion | dent | rivet_damage
    confidence: float          # 0-1
    bbox: tuple                # (x, y, w, h) in pixels
    relative_area: float        # bbox area / image area
    length_cm: float | None = None  # only estimated for classical backend


def _bbox_area_ratio(bbox, img_shape) -> float:
    _, _, w, h = bbox
    img_h, img_w = img_shape[:2]
    return (w * h) / float(img_w * img_h)


# ---------------------------------------------------------------------
# Classical CV backend
# ---------------------------------------------------------------------

def _detect_cracks(raw_gray: np.ndarray, img_shape) -> List[DefectDetection]:
    """Cracks: thin, elongated, high-contrast linear structures.

    Uses probabilistic Hough line detection rather than plain contours,
    since crack edges are often broken into several short segments by
    Canny + dilation and a pure contour/aspect-ratio test misses them.
    """
    edges = cv2.Canny(raw_gray, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=25,
        minLineLength=35, maxLineGap=12,
    )

    results = []
    if lines is None:
        return results

    img_h, img_w = img_shape[:2]
    max_len = 0.35 * max(img_w, img_h)  # long, perfectly straight spans are panel seams, not cracks

    for line in lines:
        # HoughLinesP normally returns shape (N, 1, 4), so line[0] is the
        # 4-tuple. Some OpenCV builds instead return shape (N, 4), where
        # `line` itself is already the 4-tuple. np.array(line).reshape(-1)
        # handles both cases safely.
        x1, y1, x2, y2 = np.array(line).reshape(-1)
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 35 or length > max_len:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        is_axis_aligned = angle < 3 or angle > 177 or (87 < angle < 93)
        if is_axis_aligned and length > max_len * 0.5:
            # long axis-aligned lines are almost always seams/edges, not cracks
            continue
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        pad = 4
        x, y = max(0, x - pad), max(0, y - pad)
        w, h = w + pad * 2, h + pad * 2
        confidence = float(np.clip(0.55 + 0.01 * length, 0.55, 0.93))
        bbox = (x, y, max(w, 6), max(h, 6))
        results.append(DefectDetection(
            defect_type="crack",
            confidence=round(confidence, 2),
            bbox=bbox,
            relative_area=_bbox_area_ratio(bbox, img_shape),
            length_cm=round(length / 40.0, 1),
        ))
    return results


def _detect_corrosion(bgr: np.ndarray, img_shape) -> List[DefectDetection]:
    """Corrosion: rust/brown-orange discoloration patches."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 60, 40])
    upper = np.array([30, 255, 200])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 120:
            continue
        x, y, w, h = cv2.boundingRect(c)
        fill_ratio = area / float(w * h)
        confidence = float(np.clip(0.5 + 0.4 * fill_ratio, 0.5, 0.92))
        bbox = (x, y, w, h)
        results.append(DefectDetection(
            defect_type="corrosion",
            confidence=round(confidence, 2),
            bbox=bbox,
            relative_area=_bbox_area_ratio(bbox, img_shape),
        ))
    return results


def _detect_dents(raw_gray: np.ndarray, img_shape) -> List[DefectDetection]:
    """Dents: smooth circular/blobby local-intensity depressions.

    Compares the raw image against a heavily-blurred version of itself
    (large kernel) to isolate broad, gradual shading changes — this
    picks up dent-like shadow gradients while ignoring thin/sharp
    features like cracks or panel seams. A max-area cap and a
    border-touch check prevent global illumination gradients or seam
    lines from being reported as a single giant "dent".
    """
    img_h, img_w = img_shape[:2]
    total_area = img_h * img_w

    # Black-hat highlights small dark regions relative to their local
    # surroundings, regardless of whether the darkening is a sharp edge
    # or a smooth gradient (as a dent's shading typically is) — a plain
    # blur-and-subtract only lights up at the gradient's edge, not its
    # body.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
    blackhat = cv2.morphologyEx(raw_gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 250 or area > 0.12 * total_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # Reject boxes that touch/span the image edges (seams, borders, global gradients)
        if x <= 2 or y <= 2 or (x + w) >= img_w - 2 or (y + h) >= img_h - 2:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect < 2.2:
            circularity = 4 * np.pi * area / max(1.0, cv2.arcLength(c, True) ** 2)
            confidence = float(np.clip(0.45 + 0.4 * circularity, 0.45, 0.85))
            bbox = (x, y, w, h)
            results.append(DefectDetection(
                defect_type="dent",
                confidence=round(confidence, 2),
                bbox=bbox,
                relative_area=_bbox_area_ratio(bbox, img_shape),
            ))
    return results


def _detect_rivet_damage(raw_gray: np.ndarray, img_shape) -> List[DefectDetection]:
    """Rivet damage: small, anomalously dark circular spots.

    Uses a strict darkness threshold on the *unequalized* grayscale
    image so normal, healthy rivets (mid-gray) are not mistaken for
    damaged/missing ones (near-black). Tune RIVET_DARKNESS_THRESH down
    further if your real camera images have darker healthy rivets.
    """
    RIVET_DARKNESS_THRESH = 32  # 0-255, lower = stricter (only very dark spots)

    _, mask = cv2.threshold(raw_gray, RIVET_DARKNESS_THRESH, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (15 <= area <= 250):
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(1, min(w, h))
        if aspect < 1.6:  # roughly circular, rivet-sized
            bbox = (x, y, w, h)
            results.append(DefectDetection(
                defect_type="rivet_damage",
                confidence=0.65,
                bbox=bbox,
                relative_area=_bbox_area_ratio(bbox, img_shape),
            ))
    return results


def _classical_detect(bgr: np.ndarray) -> List[DefectDetection]:
    raw_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img_shape = bgr.shape

    detections: List[DefectDetection] = []
    detections += _detect_cracks(raw_gray, img_shape)
    detections += _detect_corrosion(bgr, img_shape)
    detections += _detect_dents(raw_gray, img_shape)
    detections += _detect_rivet_damage(raw_gray, img_shape)

    # Simple non-max suppression across all types to avoid duplicate
    # overlapping boxes on the same physical spot.
    detections = _suppress_overlaps(detections)
    return detections


def _suppress_overlaps(detections: List[DefectDetection], iou_thresh: float = 0.3) -> List[DefectDetection]:
    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0

    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[DefectDetection] = []
    for d in detections:
        if all(iou(d.bbox, k.bbox) < iou_thresh for k in kept):
            kept.append(d)
    return kept


# ---------------------------------------------------------------------
# YOLOv8 backend (optional)
# ---------------------------------------------------------------------

_yolo_model = None


def _try_load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    if not os.path.exists(YOLO_WEIGHTS_PATH):
        return None
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_WEIGHTS_PATH)
        return _yolo_model
    except ImportError:
        return None


def _yolo_detect(bgr: np.ndarray) -> List[DefectDetection]:
    model = _try_load_yolo()
    results = model.predict(bgr, verbose=False)[0]
    detections = []
    names = results.names
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x1, y2 - y1
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = (int(x1), int(y1), int(w), int(h))
        detections.append(DefectDetection(
            defect_type=names.get(cls_id, "unknown"),
            confidence=round(conf, 2),
            bbox=bbox,
            relative_area=_bbox_area_ratio(bbox, bgr.shape),
        ))
    return detections


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def get_active_backend() -> str:
    return "yolo" if os.path.exists(YOLO_WEIGHTS_PATH) and _try_load_yolo() else "classical"


def detect(bgr_image: np.ndarray) -> List[DefectDetection]:
    """
    Run defect detection on a BGR (OpenCV-style) image array.
    Automatically uses YOLOv8 if trained weights are present at
    models/best.pt, otherwise falls back to the classical CV pipeline.
    """
    if get_active_backend() == "yolo":
        try:
            return _yolo_detect(bgr_image)
        except Exception:
            # Fail safe: never let a broken YOLO env kill the demo
            return _classical_detect(bgr_image)
    return _classical_detect(bgr_image)
