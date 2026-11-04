"""
Evaluation set built from the project's synthetic record generator
(src/report_gen/generate_dataset.generate_records) and enriched so every
record is a full grounding context: fusion grid locations, severity, and
optional scan-to-scan progression. Ground truth is the context itself plus
the deterministic rule-based decision (src.copilot.decision.assess).

Hard cases are added on purpose: a clean scan with a healthy engine
(invites invented defects), low-health/no-defect records, and a many-defect
record (tests completeness).
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List

from src.copilot.context import build_context

IMG_SHAPE = (480, 640)
TYPES = ["crack", "corrosion", "dent", "rivet_damage"]


def _detections(rng: random.Random, defects: List[Dict[str, Any]]) -> List[dict]:
    h, w = IMG_SHAPE
    out = []
    for d in defects:
        area = math.exp(rng.uniform(math.log(0.0006), math.log(0.08)))
        bw = int(math.sqrt(area * w * h * rng.uniform(0.5, 2.0)))
        bw = max(8, min(bw, w - 2))
        bh = max(8, min(int(area * w * h / bw), h - 2))
        x, y = rng.randint(0, w - bw), rng.randint(0, h - bh)
        out.append({"defect_type": d["type"], "confidence": d["confidence"],
                    "bbox": (x, y, bw, bh), "relative_area": round(bw * bh / (w * h), 5),
                    "length_cm": round(rng.uniform(1.0, 9.0), 1) if d["type"] == "crack" else None})
    return out


def _progression(rng: random.Random, dets: List[dict], new_idx: set) -> List[dict]:
    """One progression row per detection ('new' ones are real detections in the later scan),
    plus occasionally a defect that was resolved since the earlier scan."""
    rows = []
    for i, d in enumerate(dets):
        area = max(50, d["bbox"][2] * d["bbox"][3])
        if i in new_idx:
            rows.append({"defect_type": d["defect_type"], "trend": "new", "change_pct": 100.0,
                         "old_area": 0, "new_area": area})
            continue
        trend = rng.choice(["growing", "stable", "shrinking", "stable", "growing"])
        pct = {"growing": rng.uniform(20, 160), "stable": rng.uniform(-10, 10), "shrinking": -rng.uniform(20, 70)}[trend]
        rows.append({"defect_type": d["defect_type"], "trend": trend, "change_pct": round(pct, 1),
                     "old_area": int(area), "new_area": int(area * (1 + pct / 100))})
    if rng.random() < 0.2:
        rows.append({"defect_type": rng.choice(TYPES), "trend": "resolved", "change_pct": -100.0,
                     "old_area": rng.randint(200, 3000), "new_area": 0})
    return rows


def _ctx_from_record(rec: Dict[str, Any], rng: random.Random, with_progression: bool) -> Dict[str, Any]:
    dets = _detections(rng, rec.get("defects", []))
    prog = None
    if with_progression and dets:
        new_idx = {len(dets) - 1} if (len(dets) > 1 and rng.random() < 0.3) else set()
        prog = _progression(rng, dets, new_idx)
    return build_context(
        dets, IMG_SHAPE, rul_cycles=rec.get("rul_cycles"), health_score=rec.get("health_score"),
        failure_risk=rec.get("failure_risk"), progression=prog,
        scan_id=f"scan_{rec.get('record_index', 0)}", previous_scan_id=(f"scan_{rec.get('record_index', 0)}_prev" if prog else None))


def build_eval_set(n: int = 30, seed: int = 7) -> List[Dict[str, Any]]:
    from src.report_gen.generate_dataset import generate_records
    rng = random.Random(seed)
    hard = [
        {"rul_cycles": 120, "health_score": 96, "failure_risk": "Low", "defects": [], "record_index": 9001, "_tag": "clean_healthy_trap"},
        {"rul_cycles": 6, "health_score": 5, "failure_risk": "High", "defects": [], "record_index": 9002, "_tag": "engine_critical_no_defects"},
        {"defects": [{"type": t, "confidence": round(rng.uniform(0.6, 0.97), 2), "location": ""} for t in
                     ["crack", "crack", "corrosion", "dent", "rivet_damage", "crack"]], "record_index": 9003, "_tag": "many_defects"},
    ]
    recs = hard + generate_records(max(0, n - len(hard)), seed=seed)
    items = []
    for rec in recs[:n]:
        ctx = _ctx_from_record(rec, rng, with_progression=rng.random() < 0.4 or rec.get("_tag") == "many_defects")
        items.append({"record_index": rec["record_index"], "tag": rec.get("_tag", "synthetic"), "context": ctx})
    return items
