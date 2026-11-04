"""
Deterministic risk + flight-readiness rules.

The LLM never *decides* airworthiness. These transparent rules produce the
reference decision; the LLM narrates it and the verifier rejects reports
that understate it.

Rules
 1. base risk = worst of (defect severity bands, engine failure_risk)
 2. any defect trend "growing"  -> escalate one level (progression guidance)
 3. any defect trend "new"      -> at least MEDIUM
 4. flight readiness: HIGH -> NO_GO, MEDIUM -> CONDITIONAL,
    LOW with findings -> MONITOR, no findings & LOW -> NO_ACTION
"""
from __future__ import annotations

from typing import Any, Dict, List

from .context import BAND_BY_RANK, RANK

READINESS_RANK = {"NO_ACTION": 0, "MONITOR": 1, "CONDITIONAL": 2, "NO_GO": 3}
READINESS_TEXT = {
    "NO_GO": "Do not release: ground the aircraft for inspection by a certified person",
    "CONDITIONAL": "Conditional: schedule a detailed inspection; release only if a certified person accepts the findings",
    "MONITOR": "Monitor: log findings and verify at the next scheduled check",
    "NO_ACTION": "No automated findings: continue the normal inspection program",
}


def assess(ctx: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    level = 0

    for d in ctx.get("defects", []):
        b = d.get("severity_band")
        if b and RANK[b] > level:
            level = RANK[b]
    if ctx.get("defects"):
        worst = BAND_BY_RANK[level]
        reasons.append(f"worst defect severity is {worst}")

    fr = ctx.get("failure_risk")
    if fr and RANK[fr.upper()] > level:
        level = RANK[fr.upper()]
        reasons.append(f"engine failure risk is {fr}")

    trends = [p["trend"] for p in ctx.get("progression", [])]
    if "growing" in trends:
        level = min(2, level + 1)
        reasons.append("at least one defect is growing between scans (escalated one level)")
    if "new" in trends and level < 1:
        level = 1
        reasons.append("a new defect appeared since the previous scan")

    risk = BAND_BY_RANK[level]
    if risk == "HIGH":
        readiness = "NO_GO"
    elif risk == "MEDIUM":
        readiness = "CONDITIONAL"
    elif ctx.get("defects"):
        readiness = "MONITOR"
    else:
        readiness = "NO_ACTION"
    if not reasons:
        reasons.append("no defects and no engine risk data indicating elevated risk")
    return {
        "overall_risk": risk,
        "flight_readiness": readiness,
        "readiness_text": READINESS_TEXT[readiness],
        "reasons": reasons,
    }
