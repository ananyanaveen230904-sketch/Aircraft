"""
AeroVision — Edge AI Aircraft Skin Inspection Dashboard
Tata Technologies InnoVent | Aerospace Vertical

Run with:
    streamlit run app.py
"""

import os
import sys

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.detector import detect, get_active_backend
from src.report import build_heatmap, build_report_dataframe, draw_overlays
from src.progression import compare_scans

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_images")

st.set_page_config(
    page_title="AeroVision | Edge AI Defect Inspection",
    page_icon="✈️",
    layout="wide",
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_image(file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def list_sample_images():
    if not os.path.isdir(SAMPLE_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg")))


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.title("✈️ AeroVision")
st.sidebar.caption("Edge AI for Aircraft Skin Inspection")

backend = get_active_backend()
backend_label = "YOLOv8 (trained weights found)" if backend == "yolo" else "Classical CV (no trained weights yet)"
st.sidebar.info(f"**Active detection backend:**\n\n{backend_label}")

mode = st.sidebar.radio(
    "Mode",
    ["Single Scan Inspection", "Progression Tracking (2 scans)", "Maintenance Copilot (LLM + RAG)"],
    help="Single Scan detects defects in one image. Progression Tracking compares two scans of the same panel over time.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this backend**\n\n"
    "By default AeroVision runs on a lightweight, explainable OpenCV "
    "heuristic pipeline — zero downloads, zero GPU required, so the "
    "whole demo works offline. Drop a trained YOLOv8 model at "
    "`models/best.pt` to automatically switch to real ML inference "
    "(see `train.py` and `README.md`)."
)

# ---------------------------------------------------------------------
# Maintenance Copilot (LLM + RAG + tools) renders its own page
# ---------------------------------------------------------------------

if mode == "Maintenance Copilot (LLM + RAG)":
    from src.copilot.ui import render_copilot_page
    render_copilot_page()
    st.stop()

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("AeroVision — Intelligent Aircraft Skin Inspection")
st.caption(
    "Multi-defect detection · Severity scoring · Explainable AI overlay · "
    "Defect progression tracking — built for Tata Technologies InnoVent"
)

sample_files = list_sample_images()

# ---------------------------------------------------------------------
# MODE 1: Single scan inspection
# ---------------------------------------------------------------------

if mode == "Single Scan Inspection":
    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload an aircraft panel image", type=["jpg", "jpeg", "png"]
        )
    with col_sample:
        sample_choice = st.selectbox(
            "...or use a sample image", ["-- none --"] + sample_files
        )

    image = None
    if uploaded is not None:
        image = load_image(uploaded)
    elif sample_choice != "-- none --":
        image = cv2.imread(os.path.join(SAMPLE_DIR, sample_choice))

    if image is None:
        st.info("Upload an image or pick a sample from the dropdown to run inspection.")
        st.stop()

    with st.spinner("Running defect detection..."):
        detections = detect(image)
        overlay = draw_overlays(image, detections)
        heatmap = build_heatmap(image, detections)
        report_df = build_report_dataframe(detections)

    st.success(f"Inspection complete — {len(detections)} region(s) flagged.")

    tab1, tab2, tab3 = st.tabs(["Detection Overlay", "Explainability Heatmap", "Original"])
    with tab1:
        st.image(bgr_to_rgb(overlay), use_container_width=True,
                  caption="Bounding boxes: type, confidence, severity band")
    with tab2:
        st.image(bgr_to_rgb(heatmap), use_container_width=True,
                  caption="Severity-weighted heatmap — why each region was flagged")
    with tab3:
        st.image(bgr_to_rgb(image), use_container_width=True, caption="Original input")

    st.subheader("Inspection Report")
    if report_df.empty:
        st.write("No defects detected in this image.")
    else:
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        high = (report_df["Severity Band"] == "HIGH").sum()
        med = (report_df["Severity Band"] == "MEDIUM").sum()
        low = (report_df["Severity Band"] == "LOW").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("HIGH severity", high)
        c2.metric("MEDIUM severity", med)
        c3.metric("LOW severity", low)

        csv = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download report as CSV", data=csv,
            file_name="aerovision_inspection_report.csv", mime="text/csv",
        )

# ---------------------------------------------------------------------
# MODE 2: Progression tracking
# ---------------------------------------------------------------------

else:
    st.subheader("Compare two scans of the same panel")
    st.caption(
        "Upload (or pick sample) images from two different inspection dates. "
        "AeroVision matches defects by type + position and reports whether "
        "they are growing, shrinking, new, or resolved."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Scan 1 (earlier)**")
        up1 = st.file_uploader("Upload scan 1", type=["jpg", "jpeg", "png"], key="scan1")
        sample1 = st.selectbox("...or sample", ["-- none --"] + sample_files, key="sample1")
    with col2:
        st.markdown("**Scan 2 (later)**")
        up2 = st.file_uploader("Upload scan 2", type=["jpg", "jpeg", "png"], key="scan2")
        sample2 = st.selectbox("...or sample", ["-- none --"] + sample_files, key="sample2",
                                index=(sample_files.index("panel_defects_scan2.jpg") + 1)
                                if "panel_defects_scan2.jpg" in sample_files else 0)

    img1 = load_image(up1) if up1 is not None else (
        cv2.imread(os.path.join(SAMPLE_DIR, sample1)) if sample1 != "-- none --" else None)
    img2 = load_image(up2) if up2 is not None else (
        cv2.imread(os.path.join(SAMPLE_DIR, sample2)) if sample2 != "-- none --" else None)

    if img1 is None or img2 is None:
        st.info("Provide both scans (upload or sample) to run progression tracking.")
        st.stop()

    with st.spinner("Analyzing both scans..."):
        d1 = detect(img1)
        d2 = detect(img2)
        overlay1 = draw_overlays(img1, d1)
        overlay2 = draw_overlays(img2, d2)
        progression = compare_scans(d1, d2)

    c1, c2 = st.columns(2)
    with c1:
        st.image(bgr_to_rgb(overlay1), use_container_width=True, caption="Scan 1")
    with c2:
        st.image(bgr_to_rgb(overlay2), use_container_width=True, caption="Scan 2")

    st.subheader("Progression Report")
    if not progression:
        st.write("No defects detected in either scan.")
    else:
        import pandas as pd
        rows = []
        for p in progression:
            trend_icon = {
                "growing": "🔴 Growing",
                "shrinking": "🟢 Shrinking",
                "stable": "🟡 Stable",
                "new": "🆕 New",
                "resolved": "✅ Resolved",
            }[p.trend]
            rows.append({
                "Defect Type": p.defect_type.replace("_", " ").title(),
                "Old Area (px²)": int(p.old_area),
                "New Area (px²)": int(p.new_area),
                "Change": f"{p.change_pct:+.1f}%",
                "Trend": trend_icon,
            })
        prog_df = pd.DataFrame(rows)
        st.dataframe(prog_df, use_container_width=True, hide_index=True)

        growing = sum(1 for p in progression if p.trend == "growing")
        new_defects = sum(1 for p in progression if p.trend == "new")
        if growing or new_defects:
            st.warning(
                f"⚠️ {growing} defect(s) are growing and {new_defects} new defect(s) "
                "appeared since the last scan — recommend prioritizing these for inspection."
            )
        else:
            st.success("No defects are growing since the last scan.")

st.markdown("---")
st.caption("AeroVision Prototype · Tata Technologies InnoVent · Aerospace Vertical")
