"""
Tool layer for the agent: thin, JSON-in / JSON-out wrappers around the
existing AeroVision functions (detect, severity, progression, RUL, fusion)
plus RAG, report generation and the vision second opinion.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from src.detector import detect, get_active_backend
from src.progression import compare_scans

from . import config, rul_service
from .context import build_context
from .decision import assess
from .llm import LLM
from .rag import KnowledgeBase, get_kb
from .report_generator import generate_report
from .vision import analyze_scan, second_opinion


# ---------------------------------------------------------------- scan store
@dataclass
class Scan:
    id: str
    image: np.ndarray
    added_at: int
    detections: Optional[list] = None
    detection_quality: Optional[Dict[str, Any]] = None
    vision_review: Optional[Dict[str, Any]] = None


class ScanStore:
    """Holds uploaded/sample scans in insertion order so 'latest'/'previous' resolve."""

    def __init__(self):
        self.scans: Dict[str, Scan] = {}
        self._added_count = 0

    def _ordered_ids(self) -> List[str]:
        return [s.id for s in sorted(self.scans.values(), key=lambda scan: scan.added_at)]

    def add(self, scan_id: str, image: np.ndarray) -> str:
        base, n = scan_id, 2
        while scan_id in self.scans and self.scans[scan_id].image.shape != image.shape:
            scan_id, n = f"{base}_{n}", n + 1
        self._added_count += 1
        self.scans.pop(scan_id, None)
        self.scans[scan_id] = Scan(scan_id, image, self._added_count)
        return scan_id

    DEFAULT_SAMPLE_ORDER = ["panel_clean.jpg", "panel_rivet_line.jpg",
                            "panel_defects_scan1.jpg", "panel_defects_scan2.jpg"]

    def load_samples(self, directory: str | None = None, names: List[str] | None = None) -> List[str]:
        """Load sample images in a meaningful order (so 'previous'/'latest' = scan1/scan2)."""
        directory = directory or config.SAMPLE_DIR
        if not os.path.isdir(directory):
            return []
        present = [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        wanted = names or ([n for n in self.DEFAULT_SAMPLE_ORDER if n in present]
                           + sorted(f for f in present if f not in self.DEFAULT_SAMPLE_ORDER))
        out = []
        for f in wanted:
            img = cv2.imread(os.path.join(directory, f))
            if img is not None:
                out.append(self.add(os.path.splitext(f)[0], img))
        return out

    def resolve(self, ref: Optional[str]) -> Scan:
        ids = self._ordered_ids()
        if not ids:
            raise KeyError("no scans loaded")
        r = (ref or "latest").strip()
        if r in ("latest", "current", "last"):
            return self.scans[ids[-1]]
        if r in ("previous", "prior", "earlier"):
            if len(ids) < 2:
                raise KeyError("only one scan loaded; no previous scan")
            return self.scans[ids[-2]]
        if r in self.scans:
            return self.scans[r]
        matches = [i for i in ids if r.lower() in i.lower()]
        if len(matches) == 1:
            return self.scans[matches[0]]
        raise KeyError(f"unknown scan_id '{ref}'. Available: {ids}")

    def detections(self, scan: Scan) -> list:
        if scan.detections is None:
            scan.detections = detect(scan.image)
        return scan.detections


# -------------------------------------------------------------------- tools
class CopilotTools:
    def __init__(self, store: ScanStore, kb: Optional[KnowledgeBase] = None,
                 report_llm: Optional[LLM] = None, vision_llm=None):
        self.store = store
        self._kb = kb
        self.report_llm = report_llm
        self.vision_llm = vision_llm
        self._registry: Dict[str, Callable[..., Dict[str, Any]]] = {
            "list_scans": self.list_scans,
            "detect_defects": self.detect_defects,
            "score_severity": self.score_severity,
            "compare_scans": self.compare_scans,
            "predict_rul": self.predict_rul,
            "fuse_inspection": self.fuse_inspection,
            "assess_flight_readiness": self.assess_flight_readiness,
            "retrieve_guidance": self.retrieve_guidance,
            "generate_report": self.generate_report,
            "vision_second_opinion": self.vision_second_opinion,
            "analyze_image_with_vision": self.analyze_image_with_vision,
        }

    @property
    def kb(self) -> KnowledgeBase:
        if self._kb is None:
            self._kb = get_kb()
        return self._kb

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        fn = self._registry.get(name)
        if fn is None:
            return {"error": f"unknown tool '{name}'"}
        try:
            return fn(**(arguments or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except KeyError as e:
            return {"error": str(e.args[0]) if e.args else str(e)}
        except Exception as e:
            return {"error": f"{name} failed: {type(e).__name__}: {e}"}

    # ---- individual tools -------------------------------------------------
    def list_scans(self) -> Dict[str, Any]:
        ids = self.store._ordered_ids()
        return {"scans": [{"scan_id": i, "order": n + 1, "added_at": self.store.scans[i].added_at,
                           "alias": "latest" if n == len(ids) - 1 else ("previous" if n == len(ids) - 2 else None)}
                          for n, i in enumerate(ids)]}

    def _ctx(self, scan_id="latest", previous_scan_id: Optional[str] = None,
             engine_id: Optional[int] = None) -> Dict[str, Any]:
        scan = self.store.resolve(scan_id)
        dets = self.store.detections(scan)
        quality = self._detection_quality(scan, dets)
        prog = None
        prev_id = None
        if previous_scan_id:
            prev = self.store.resolve(previous_scan_id)
            prog = compare_scans(self.store.detections(prev), dets)
            prev_id = prev.id
        rul = {}
        if engine_id is not None:
            rul = rul_service.predict_engine(int(engine_id))
            if "error" in rul:
                rul = {"_rul_error": rul["error"]}
        ctx = build_context(
            dets, scan.image.shape, progression=prog, scan_id=scan.id, previous_scan_id=prev_id,
            rul_cycles=rul.get("rul_cycles"), health_score=rul.get("health_score"),
            failure_risk=rul.get("failure_risk"), detection_quality=quality,
            vision_review=scan.vision_review)
        if "_rul_error" in rul:
            ctx["rul_unavailable_reason"] = rul["_rul_error"]
        return ctx

    @staticmethod
    def _detection_quality(scan: Scan, detections: list) -> Dict[str, Any]:
        if scan.detection_quality is None:
            height, width = scan.image.shape[:2]
            raw_count = len(detections)
            megapixels = (height * width) / 1_000_000
            density = raw_count / megapixels if megapixels > 0 else float(raw_count)
            scan.detection_quality = {
                "raw_count": raw_count,
                "detections_per_megapixel": round(density, 2),
                "likely_unreliable": (
                    raw_count > config.DETECTION_QUALITY_MAX_RAW_COUNT
                    or density > config.DETECTION_QUALITY_MAX_PER_MEGAPIXEL
                ),
            }
        return scan.detection_quality

    def build_record(self, scan_id="latest", previous_scan_id: Optional[str] = None,
                     engine_id: Optional[int] = None) -> Dict[str, Any]:
        """Build the full inspection record for a scan and its optional comparisons."""
        return self._ctx(scan_id, previous_scan_id, engine_id)

    def detect_defects(self, scan_id: str = "latest") -> Dict[str, Any]:
        scan = self.store.resolve(scan_id)
        dets = self.store.detections(scan)
        ctx = build_context(
            dets, scan.image.shape, scan_id=scan.id,
            detection_quality=self._detection_quality(scan, dets),
            vision_review=scan.vision_review,
        )
        return {
            "scan_id": scan.id, "backend": get_active_backend(),
            "n_defects": len(ctx["defects"]), "defects": ctx["defects"],
            "detection_quality": ctx["detection_quality"],
            **({"warning": ctx["warning"]} if "warning" in ctx else {}),
            **({"vision_review": scan.vision_review} if scan.vision_review is not None else {}),
        }

    def score_severity(self, scan_id: str = "latest") -> Dict[str, Any]:
        ctx = self._ctx(scan_id)
        return {"scan_id": ctx["scan_id"], "severity": [
            {k: d.get(k) for k in ("id", "type", "location", "severity_score", "severity_band", "rule_based_action")}
            for d in ctx["defects"]], "counts": ctx["counts"]}

    def compare_scans(self, previous_scan_id: str, current_scan_id: str = "latest") -> Dict[str, Any]:
        ctx = self._ctx(current_scan_id, previous_scan_id)
        return {"previous_scan_id": ctx.get("previous_scan_id"), "current_scan_id": ctx["scan_id"],
                "progression": ctx.get("progression", []),
                "counts": {k: ctx["counts"][k] for k in ("growing", "new", "resolved")}}

    def predict_rul(self, engine_id: int) -> Dict[str, Any]:
        return rul_service.predict_engine(int(engine_id))

    def fuse_inspection(self, scan_id: str = "latest", previous_scan_id: Optional[str] = None,
                        engine_id: Optional[int] = None) -> Dict[str, Any]:
        return self._ctx(scan_id, previous_scan_id, engine_id)

    def assess_flight_readiness(self, scan_id: str = "latest", previous_scan_id: Optional[str] = None,
                                engine_id: Optional[int] = None) -> Dict[str, Any]:
        ctx = self._ctx(scan_id, previous_scan_id, engine_id)
        return {"scan_id": ctx["scan_id"], "previous_scan_id": ctx.get("previous_scan_id"),
                **assess(ctx), "counts": ctx["counts"], "defects": ctx["defects"],
                "detection_quality": ctx["detection_quality"],
                "progression": ctx.get("progression", []),
                **({"warning": ctx["warning"]} if "warning" in ctx else {}),
                **({"vision_review": ctx["vision_review"]} if "vision_review" in ctx else {}),
                **({k: ctx[k] for k in ("rul_cycles", "health_score", "failure_risk", "rul_unavailable_reason") if k in ctx}),
                "note": "Rule-based advisory decision; a certified person makes the release decision."}

    def retrieve_guidance(self, query: str, defect_type: Optional[str] = None,
                          severity: Optional[str] = None, k: int = 3,
                          chunk_id: Optional[str] = None) -> Dict[str, Any]:
        if chunk_id:
            chunk = self.kb.get(chunk_id)
            if chunk is None:
                return {"backend": self.kb.backend, "chunks": []}
            return {"backend": self.kb.backend, "chunks": [{
                **chunk.to_dict(), "score": 1.0,
            }]}
        hits = self.kb.search(query, k=int(k), defect_type=defect_type, severity=severity)
        return {"backend": self.kb.backend, "chunks": [h.to_dict() for h in hits]}

    def generate_report(self, scan_id: str = "latest", previous_scan_id: Optional[str] = None,
                        engine_id: Optional[int] = None, mode: str = "llm_rag") -> Dict[str, Any]:
        ctx = self._ctx(scan_id, previous_scan_id, engine_id)
        if mode != "template" and self.report_llm is None:
            mode = "template"
        gen = generate_report(ctx, mode, self.report_llm, self.kb)
        return {"mode": gen.mode, "fallback_used": gen.fallback_used,
                "verified": bool(gen.verification and gen.verification.ok),
                "report": gen.report.model_dump(),
                "guidance_chunk_ids": [r["id"] for r in gen.retrieved]}

    def vision_second_opinion(self, scan_id: str = "latest", defect_id: str = "D1") -> Dict[str, Any]:
        if self.vision_llm is None:
            return {"error": "vision second opinion unavailable (set GEMINI_API_KEY)"}
        scan = self.store.resolve(scan_id)
        ctx = build_context(self.store.detections(scan), scan.image.shape, scan_id=scan.id)
        ids = [d["id"] for d in ctx["defects"]]
        if defect_id not in ids:
            return {"error": f"defect_id {defect_id} not in scan {scan.id}; available: {ids}"}
        det = self.store.detections(scan)[ids.index(defect_id)]
        return {"scan_id": scan.id, "defect_id": defect_id, **second_opinion(scan.image, det, self.vision_llm)}

    def analyze_image_with_vision(self, scan_id: str = "latest") -> Dict[str, Any]:
        scan = self.store.resolve(scan_id)
        dets = self.store.detections(scan)
        quality = self._detection_quality(scan, dets)
        if self.vision_llm is None:
            return {"error": "Vision review unavailable: configure GEMINI_API_KEY and the Gemini SDK."}
        if scan.vision_review is None:
            result = analyze_scan(scan.image, self.vision_llm)
            if "error" in result:
                return result
            scan.vision_review = result

        detector_counts = dict(sorted(Counter(
            str(d.get("defect_type", "unknown") if isinstance(d, dict)
                else getattr(d, "defect_type", "unknown"))
            for d in dets
        ).items()))
        vision_counts = dict(sorted(Counter(d["type"] for d in scan.vision_review["defects"]).items()))
        all_types = set(detector_counts) | set(vision_counts)
        deltas = [vision_counts.get(t, 0) - detector_counts.get(t, 0) for t in all_types]
        if all(delta == 0 for delta in deltas):
            agreement = "agree"
        elif any(delta < 0 for delta in deltas) and not any(delta > 0 for delta in deltas):
            agreement = "detector_overcounts"
        elif any(delta > 0 for delta in deltas) and not any(delta < 0 for delta in deltas):
            agreement = "vision_found_more"
        else:
            agreement = "mixed"

        result = {
            "scan_id": scan.id,
            "vision_review": scan.vision_review,
            "detector_comparison": {
                "detector_counts_by_type": detector_counts,
                "vision_counts_by_type": vision_counts,
                "agreement": agreement,
            },
            "advisory_only": True,
            "note": (
                "This image is not classified as a close-up panel; the detector is tuned for close-up panels."
                if scan.vision_review["scene_type"] != "closeup_panel" else
                "Vision review is advisory only and does not change the detector-based risk or flight-readiness decision."
            ),
        }
        if quality["likely_unreliable"]:
            result["warning"] = config.UNRELIABLE_DETECTION_WARNING
        return result

    # ---- OpenAI-style tool specs -------------------------------------------
    def schemas(self) -> List[dict]:
        def fn(name, desc, props=None, required=None):
            return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
                "type": "object", "properties": props or {}, "required": required or []}}}
        sid = {"type": "string", "description": "scan id from list_scans, or 'latest' / 'previous'"}
        eng = {"type": "integer", "description": "NASA C-MAPSS FD001 test engine id (1-100); only if the user asks for engine RUL"}
        return [
            fn("list_scans", "List loaded scans in chronological order with aliases 'latest'/'previous'."),
            fn("detect_defects", "Run the defect detector on a scan; returns defects with ids, type, confidence, location, severity.",
               {"scan_id": sid}),
            fn("score_severity", "Severity score and band (LOW/MEDIUM/HIGH) for each defect in a scan.", {"scan_id": sid}),
            fn("compare_scans", "Compare an earlier scan with a later one: which defects grew, shrank, are new or resolved.",
               {"previous_scan_id": sid, "current_scan_id": sid}, ["previous_scan_id"]),
            fn("predict_rul", "Predict engine remaining useful life, health score and failure risk.", {"engine_id": eng}, ["engine_id"]),
            fn("fuse_inspection", "Fused inspection record (defects+severity+progression+optional engine RUL).",
               {"scan_id": sid, "previous_scan_id": sid, "engine_id": eng}),
            fn("assess_flight_readiness", "Rule-based advisory risk and flight-readiness decision. ALWAYS call this before answering any 'is it safe to fly / release' question.",
               {"scan_id": sid, "previous_scan_id": sid, "engine_id": eng}),
            fn("retrieve_guidance", "Search the maintenance knowledge base. Cite returned chunk ids in your answer. Use chunk_id='no-defects' when no defects are found.",
               {"query": {"type": "string"}, "defect_type": {"type": "string", "enum": ["crack", "corrosion", "dent", "rivet_damage"]},
                "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}, "k": {"type": "integer"},
                "chunk_id": {"type": "string"}}, ["query"]),
            fn("generate_report", "Generate the structured, verified inspection report.",
               {"scan_id": sid, "previous_scan_id": sid, "engine_id": eng,
                "mode": {"type": "string", "enum": ["llm_rag", "llm", "template"]}}),
            fn("vision_second_opinion", "Ask a vision LLM for an independent opinion on one flagged defect crop.",
               {"scan_id": sid, "defect_id": {"type": "string", "description": "e.g. D1"}}, ["defect_id"]),
            fn("analyze_image_with_vision", "Independently inspect the whole uploaded image with Gemini. Use when asked to look at, check, or verify an image; it is advisory and never changes rule-based risk.",
               {"scan_id": sid}),
        ]
