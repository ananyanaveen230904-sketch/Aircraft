# AeroVision

**Edge AI for Intelligent Aircraft Skin Inspection & Defect Detection**
Built for Tata Technologies InnoVent — Aerospace Vertical

AeroVision is a working software prototype that inspects aircraft
panel images for multiple defect types (cracks, corrosion, dents,
rivet damage), scores each one's severity, explains *why* it was
flagged, and tracks whether defects are growing across repeated scans.

It runs completely offline out of the box — no dataset download, no
GPU, no internet required for the demo — using a lightweight,
explainable OpenCV detection pipeline. It also supports dropping in a
real trained YOLOv8 model for higher accuracy (see "Upgrading to a
trained ML model" below), which is exactly the upgrade path you'd
narrate in your presentation as "future enhancement / scalability".

---

## Features

- **Multi-defect detection** — crack, corrosion, dent, and rivet
  damage identified in a single pass
- **Severity + urgency scoring** — not just yes/no; each detection
  gets a 0–100 score, a LOW/MEDIUM/HIGH band, and a recommended action
- **Explainable overlay** — a heatmap shows *why* a region was
  flagged, not just a box
- **Defect progression tracking** — compare two scans of the same
  panel and see which defects are growing, shrinking, new, or resolved
- **Streamlit dashboard** — upload an image (or use bundled samples),
  see detections, download a CSV inspection report
- **Edge-ready architecture** — the classical backend needs no GPU at
  all; the optional YOLOv8-nano backend is small enough for
  Jetson/Raspberry-Pi-class hardware

---

## Quick Start (VS Code / local machine)

### 1. Clone / unzip and open in VS Code

Open the `aerovision/` folder in VS Code.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If you only want the classical CV detector (no deep learning), you
> can comment out `ultralytics` and `torch` in `requirements.txt` —
> the app works fully without them.

### 4. (Optional) Regenerate the sample images

Sample synthetic panel images are already included in
`data/sample_images/`, but you can regenerate/customize them:

```bash
python generate_samples.py
```

### 5. Run the dashboard

```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`).
Pick a sample image from the dropdown or upload your own to see
detection in action. Switch to "Progression Tracking" mode in the
sidebar to compare two scans.

---

## Dashboard preview

The Streamlit dashboard (`app.py`) has two modes — Single Scan Inspection
and Progression Tracking — shown below with a screenshot of each view.

### Single Scan Inspection

An uploaded panel image gets flagged regions drawn directly on it, with
the defect type, confidence, and severity band:

![Detection overlay on an uploaded panel image](docs/screenshots/detection_overlay.png)

The "Explainability Heatmap" tab shows *why* a region was flagged, instead
of just a bounding box:

![Explainability heatmap for a flagged region](docs/screenshots/explainability_heatmap.png)

Every flagged region also lands in a downloadable inspection report, with
severity score, band, and a recommended action per defect:

![Inspection report table with severity scoring](docs/screenshots/inspection_report.png)

### Progression Tracking

Two scans of the same panel — upload both (or use bundled samples) — get
matched and compared side by side:

![Setting up a two-scan progression comparison](docs/screenshots/progression_tracking_setup.png)

The progression report shows which defects grew, shrank, appeared, or
resolved between the two scans, with an urgency callout:

![Progression report comparing two scans](docs/screenshots/progression_report.png)

---

## Project Structure

```
aerovision/
├── app.py                     # Streamlit dashboard (main entry point)
├── generate_samples.py        # Creates synthetic demo images
├── train.py                   # Optional: train a real YOLOv8 model
├── requirements.txt
├── README.md
├── src/
│   ├── detector.py            # Core detection logic (classical CV + optional YOLOv8)
│   ├── severity.py            # Severity scoring rubric
│   ├── progression.py         # Scan-to-scan defect comparison
│   ├── report.py              # Overlay drawing, heatmap, report table
│   └── copilot/               # LLM maintenance copilot (RAG, agent, verifier, eval)
├── models/
│   └── README.md              # Where to place trained YOLOv8 weights (best.pt)
├── docs/
│   └── screenshots/           # Dashboard screenshots used in this README
└── data/
    └── sample_images/         # Synthetic demo panel images
```

---

## How detection works

`src/detector.py` exposes one function, `detect(image)`, that returns
a list of `DefectDetection` objects. Internally it picks one of two
backends automatically:

| Backend | When it's used | How it works |
|---|---|---|
| **Classical CV** (default) | Always, unless a trained model is present | OpenCV heuristics: Hough line transform for cracks, HSV color masking for corrosion, black-hat morphology for dents, dark-blob detection for rivet damage |
| **YOLOv8** (optional) | Automatically, if `models/best.pt` exists | Real trained object-detection inference via `ultralytics` |

This means the project is demoable immediately, and upgrading to a
"real ML model" later is a drop-in change — you don't need to modify
`app.py` at all.

### Honest limitation of the classical backend

The classical detector is a heuristic baseline tuned against the
included synthetic sample images. On real aircraft photos it will
need re-tuning (thresholds in `src/detector.py` are clearly commented
and isolated per defect type) or, better, replacement with a trained
YOLOv8 model — which is exactly the story to tell judges: **Phase 1
(now) = explainable, dependency-free baseline; Phase 2 = trained
model for production accuracy.**

---

## Upgrading to a trained ML model

1. Get a labeled dataset (public defect-detection datasets — search
   Roboflow Universe / Kaggle for "aircraft defect", "metal surface
   defect", "corrosion detection", "crack detection")
2. Format it as a YOLO dataset (`train.py` docstring shows the exact
   folder structure + `data.yaml` format)
3. Train:
   ```bash
   pip install ultralytics torch
   python train.py --data path/to/data.yaml --epochs 60
   ```
4. Copy the resulting weights into the app:
   ```bash
   cp runs/detect/aerovision_defect_detector/weights/best.pt models/best.pt
   ```
5. Restart `streamlit run app.py` — it will now use YOLOv8 automatically.

---


---

## Maintenance Copilot (LLM + RAG + tools + guardrails)

Adds an LLM layer on top of the detector, severity, progression, RUL and fusion modules.
Open it from the sidebar: **Mode -> Maintenance Copilot (LLM + RAG)**.

| Component | Where | What it does |
|---|---|---|
| Grounded report generator | `src/copilot/report_generator.py`, `schemas.py` | Fused record (defects, severity, RUL, progression) -> LLM -> JSON validated by Pydantic -> verifier -> one repair retry -> deterministic template fallback |
| RAG knowledge base | `src/copilot/rag.py`, `src/copilot/knowledge/*.md` | Hybrid retrieval (sentence-transformers + FAISS, TF-IDF fallback) with defect-type / severity metadata boost. Reports cite chunk ids |
| Tool-calling agent | `src/copilot/agent.py`, `tools.py` | Llama 3.3 (Groq) calls `detect_defects`, `score_severity`, `compare_scans`, `predict_rul`, `fuse_inspection`, `assess_flight_readiness`, `retrieve_guidance`, `generate_report`, `vision_second_opinion` |
| Guardrails | `src/copilot/verifier.py`, `decision.py` | Every number, interval, location, defect type and citation must trace to the record or a retrieved chunk. Risk and flight readiness may not be *less* cautious than the rule-based decision |
| Evaluation | `src/copilot/evaluation/` | template vs LLM vs LLM+RAG (raw and guarded): faithfulness, hallucination rate, completeness, citation validity, unsafe-decision rate, optional LLM-as-judge |
| Vision second opinion | `src/copilot/vision.py` | Flagged crop -> Gemini (asked blind) -> agree / disagree flag. Never lowers severity |

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then add GROQ_API_KEY and GEMINI_API_KEY (both optional)
streamlit run app.py
```

Without any key the copilot still runs: rule-based agent router + template reports + RAG retrieval.
For engine RUL in the agent, train the LSTM once: `python src/rul/train.py` (creates `models/rul_lstm.pt`).

### Evaluate

```bash
python -m src.copilot.evaluation.run_eval --n 30 --judge gemini   # real run (uses your API keys)
python -m src.copilot.evaluation.run_eval --n 10 --simulate        # offline plumbing test only
python -m pytest tests -q                                           # unit tests, no keys needed
```

Results land in `output/eval/` (`results.jsonl`, `summary.csv`, `summary.md`).
The same verifier is both the runtime guardrail and an evaluation checker, so report the
LLM-as-judge columns next to the deterministic ones as the independent signal.

### Knowledge base

The files in `src/copilot/knowledge/` are **starter summaries written for this project**, paraphrasing
general maintenance themes (FAA AC 43.13-1B Ch. 4 and 6). The inspection intervals ("within 24 hours",
"within 7 days") are AeroVision demo policy aligned with `src/severity.py`, not regulatory limits.
Replace/extend them with passages from the real documents you are licensed to use. Each chunk is one `.md`
file with this header (restart the app after adding files):

```
---
id: "corrosion-pitting"
defect_type: "corrosion"      # crack | corrosion | dent | rivet_damage | general
severity: "ANY"               # HIGH | MEDIUM | LOW | ANY
source: "Document name, section"
action: "One-line recommended action"
interval: ""
---
Body text of the passage...
```

### Safety note

This is a research prototype. Outputs are advisory; a certified person makes every airworthiness decision.

---

## Notes on the sample images

`data/sample_images/*.jpg` are **synthetically generated** (via
`generate_samples.py`) stand-ins for real aircraft skin photos, so the
demo works without needing a licensed dataset. Replace them with real
inspection photos for your actual submission where possible — the
detector and dashboard work identically either way.
