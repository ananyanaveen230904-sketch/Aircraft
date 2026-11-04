"""
Generates a handful of synthetic "aircraft skin panel" images with
crack / corrosion / dent / rivet-damage patterns drawn onto them, so
the dashboard has something to detect immediately without requiring
you to source a real dataset first.

Run once:
    python generate_samples.py

This creates:
    data/sample_images/panel_clean.jpg
    data/sample_images/panel_defects_scan1.jpg
    data/sample_images/panel_defects_scan2.jpg   (same panel, defects grown -> for progression demo)
    data/sample_images/panel_rivet_line.jpg

NOTE: these are synthetic stand-ins for demo purposes only. For your
real submission, replace these with actual aircraft-skin defect
images (see README.md for public dataset suggestions).
"""

import os

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_images")
os.makedirs(OUT_DIR, exist_ok=True)


def _base_panel(w=800, h=500, seed=0):
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), 190, dtype=np.uint8)
    # brushed-metal texture
    noise = rng.normal(0, 6, (h, w)).astype(np.int16)
    for c in range(3):
        chan = base[:, :, c].astype(np.int16) + noise
        base[:, :, c] = np.clip(chan, 0, 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=0.6)
    # horizontal panel seam lines
    for y in (0, h // 2, h - 1):
        cv2.line(base, (0, y), (w, y), (140, 140, 140), 2)
    return base


def _draw_rivets(img, rows, cols, margin=40):
    h, w = img.shape[:2]
    xs = np.linspace(margin, w - margin, cols)
    ys = np.linspace(margin, h - margin, rows)
    for y in ys:
        for x in xs:
            cv2.circle(img, (int(x), int(y)), 5, (110, 110, 110), -1)
            cv2.circle(img, (int(x), int(y)), 5, (60, 60, 60), 1)
    return img


def _draw_crack(img, start, end, thickness=2, jaggedness=6):
    pts = [start]
    n = 6
    for i in range(1, n):
        t = i / n
        x = int(start[0] + (end[0] - start[0]) * t + np.random.randint(-jaggedness, jaggedness))
        y = int(start[1] + (end[1] - start[1]) * t + np.random.randint(-jaggedness, jaggedness))
        pts.append((x, y))
    pts.append(end)
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (30, 30, 30), thickness)
    return img


def _draw_corrosion(img, center, radius):
    overlay = img.copy()
    cv2.circle(overlay, center, radius, (20, 90, 160), -1)  # rust/orange in BGR
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    mask3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    blended = (img.astype(np.float32) * (1 - mask3 * 0.6) + overlay.astype(np.float32) * (mask3 * 0.6))
    return blended.astype(np.uint8)


def _draw_dent(img, center, radius):
    out = img.copy()
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    shade = np.clip(1 - dist / radius, 0, 1) ** 2 * 80
    for c in range(3):
        out[:, :, c] = np.clip(out[:, :, c].astype(np.int16) - shade.astype(np.int16), 0, 255).astype(np.uint8)
    return out


def _draw_missing_rivet(img, center):
    cv2.circle(img, center, 6, (25, 25, 25), -1)
    return img


def make_clean_panel():
    img = _base_panel(seed=1)
    img = _draw_rivets(img, rows=4, cols=10)
    cv2.imwrite(os.path.join(OUT_DIR, "panel_clean.jpg"), img)


def make_defect_scan(path, crack_len_scale=1.0, corrosion_r=45, dent_r=40, seed=2):
    np.random.seed(seed)
    img = _base_panel(seed=seed)
    img = _draw_rivets(img, rows=4, cols=10)

    img = _draw_crack(img, (150, 120), (150 + int(180 * crack_len_scale), 160), thickness=2)
    img = _draw_corrosion(img, (560, 340), radius=corrosion_r)
    img = _draw_dent(img, (300, 380), radius=dent_r)
    img = _draw_missing_rivet(img, (680, 120))

    cv2.imwrite(os.path.join(OUT_DIR, path), img)


def make_rivet_line_panel():
    img = _base_panel(seed=3)
    img = _draw_rivets(img, rows=3, cols=14)
    # damage a few rivets
    img = _draw_missing_rivet(img, (120, 250))
    img = _draw_missing_rivet(img, (400, 250))
    img = _draw_missing_rivet(img, (640, 90))
    cv2.imwrite(os.path.join(OUT_DIR, "panel_rivet_line.jpg"), img)


if __name__ == "__main__":
    make_clean_panel()
    make_defect_scan("panel_defects_scan1.jpg", crack_len_scale=1.0, corrosion_r=35, dent_r=30, seed=2)
    make_defect_scan("panel_defects_scan2.jpg", crack_len_scale=1.6, corrosion_r=55, dent_r=45, seed=2)
    make_rivet_line_panel()
    print(f"Sample images written to: {OUT_DIR}")
