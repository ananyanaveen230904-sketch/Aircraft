"""
Metrics for comparing report-generation configurations.

faithfulness            1 - unsupported_claims / claims          (verifier-based, deterministic)
hallucination_rate      share of reports with >=1 unsupported claim
completeness            share of record facts the report actually covers
citation_validity       share of recommended actions citing a retrieved chunk of the right defect type
unsafe_decision_rate    share of reports whose risk / readiness is *below* the rule-based decision
schema_valid            share of generations that parsed into the pydantic schema

NOTE ON INDEPENDENCE: the same verifier is used as the runtime guardrail and as the
evaluation checker. That is fair for comparing raw outputs (template vs LLM vs LLM+RAG)
but circular for the guardrail itself, so the optional LLM-as-judge (a different model
family) is the independent signal and should be reported next to these numbers.
"""
from __future__ import annotations

import random
import re
import statistics
from typing import Any, Dict, List, Optional

from src.copilot.decision import assess
from src.copilot.schemas import InspectionReport
from src.copilot.verifier import Verification

HALLU_CODES = {"unsupported_number", "unsupported_interval", "unsupported_location", "unsupported_defect",
               "unknown_defect_id", "type_mismatch", "invalid_citation", "unsupported_engine_claim",
               "unsupported_progression_claim"}
UNSAFE_CODES = {"risk_understated", "readiness_understated"}


def _all_text(r: InspectionReport) -> str:
    parts = [r.summary, r.engine_note or "", r.progression_note or "", r.inspection_interval or ""]
    parts += [f.observation + " " + f.recommended_action for f in r.findings]
    return " ".join(parts).lower()


def completeness(r: InspectionReport, ctx: Dict[str, Any]) -> float:
    txt = _all_text(r)
    got = total = 0

    def has_num(v) -> bool:
        cands = {str(v), str(int(v)) if float(v).is_integer() else str(round(float(v), 1))}
        return any(re.search(rf"(?<![\d.]){re.escape(c)}(?![\d])", txt) for c in cands)

    by_id = {f.defect_id: f for f in r.findings}
    for d in ctx.get("defects", []):
        total += 1; got += d["id"] in by_id and by_id[d["id"]].type == d["type"]
        f = by_id.get(d["id"])
        ftxt = (f.observation + " " + f.recommended_action).lower() if f else ""
        if "confidence" in d:
            total += 1
            pct = round(d["confidence"] * 100)
            got += bool(f) and (str(pct) in ftxt or str(d["confidence"]) in ftxt or has_num(pct))
        if "location" in d:
            total += 1; got += d["location"].lower() in txt
        if "severity_band" in d:
            total += 1; got += bool(f) and d["severity_band"].lower() in ftxt
    for key in ("rul_cycles", "health_score"):
        if key in ctx:
            total += 1; got += has_num(ctx[key])
    if "failure_risk" in ctx:
        total += 1; got += ctx["failure_risk"].lower() in txt
    for p in ctx.get("progression", []):
        total += 1
        got += (p["type"].replace("_", " ") in txt or p["type"] in txt) and (p["trend"] in txt or p["trend"].rstrip("ing") in txt)
    ref = assess(ctx)
    total += 2
    got += r.overall_risk == ref["overall_risk"]
    got += r.flight_recommendation == ref["flight_readiness"]
    return got / total if total else 1.0


def citation_validity(r: InspectionReport, retrieved: List[Dict[str, Any]], ctx: Dict[str, Any]) -> Optional[float]:
    if not retrieved:
        return None
    by_id = {c["id"]: c for c in retrieved}
    bands = {d["id"]: d.get("severity_band") for d in ctx.get("defects", [])}
    if not r.findings:
        return 1.0
    ok = 0
    for f in r.findings:
        good = False
        for cid in f.cited_chunks:
            c = by_id.get(cid)
            if c and c["defect_type"] in (f.type, "general") and c["severity"] in (bands.get(f.defect_id), "ANY"):
                good = True
        ok += good
    return ok / len(r.findings)


def score(report: Optional[InspectionReport], ver: Optional[Verification], ctx: Dict[str, Any],
          retrieved: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if report is None or ver is None:
        return {"schema_valid": 0.0, "faithfulness": None, "hallucinated": None, "completeness": None,
                "citation_validity": None, "unsafe_decision": None, "n_claims": 0, "n_unsupported": 0}
    codes = {i.code for i in ver.issues if i.level == "error"}
    n_h = sum(1 for i in ver.issues if i.level == "error" and i.code in HALLU_CODES)
    return {
        "schema_valid": 1.0,
        "faithfulness": 1.0 if ver.n_claims == 0 else max(0.0, 1 - n_h / ver.n_claims),
        "hallucinated": float(n_h > 0),
        "completeness": completeness(report, ctx),
        "citation_validity": citation_validity(report, retrieved or [], ctx),
        "unsafe_decision": float(bool(codes & UNSAFE_CODES)),
        "n_claims": ver.n_claims, "n_unsupported": n_h,
    }


METRIC_KEYS = ["schema_valid", "faithfulness", "hallucinated", "completeness", "citation_validity", "unsafe_decision"]


def bootstrap_ci(values: List[float], iters: int = 1000, seed: int = 0) -> (float, float, float):
    vals = [v for v in values if v is not None]
    if not vals:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(vals, k=len(vals))) for _ in range(iters))
    return statistics.fmean(vals), means[int(0.025 * iters)], means[int(0.975 * iters) - 1]
