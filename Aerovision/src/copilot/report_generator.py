"""
Grounded report generation.

Modes
  template : deterministic, no LLM, no RAG           (baseline)
  llm      : LLM -> JSON, no retrieved guidance      (ablation)
  llm_rag  : LLM -> JSON, grounded on retrieved KB   (full system)

Pipeline for the LLM modes:
  generate -> pydantic validation -> verifier -> (one repair retry with the
  verifier's feedback) -> deterministic template fallback.
Because the fallback is built from the record itself it always passes the
verifier, so the user never sees an unverified report when guardrails are on.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from . import config
from .context import context_to_text
from .decision import assess
from .llm import LLM
from .llm_payload import compact_for_llm
from .rag import KnowledgeBase, get_kb
from .schemas import Finding, InspectionReport
from .verifier import Verification, verify_report

MODES = ("template", "llm", "llm_rag")

DISCLAIMER = "Automated findings are advisory; a certified person must confirm them and make the release decision."
LOC_NOTE = "Locations are positional (3x3 image grid), not anatomical part names."


@dataclass
class Attempt:
    report: Optional[InspectionReport]
    error: Optional[str]
    verification: Optional[Verification]
    raw_text: Optional[str] = None


@dataclass
class GeneratedReport:
    report: InspectionReport
    mode: str
    attempts: List[Attempt] = field(default_factory=list)
    fallback_used: bool = False
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    verification: Optional[Verification] = None
    latency_s: float = 0.0
    llm_name: str = "none"

    @property
    def first_attempt(self) -> Attempt:
        return self.attempts[0]


# ---------------------------------------------------------------- retrieval
def retrieve_for_context(ctx: Dict[str, Any], kb: KnowledgeBase, k_per_defect: int = 3) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    trends = {p["trend"] for p in ctx.get("progression", [])}
    trend = "growing" if "growing" in trends else ("new" if "new" in trends else None)

    def add(hits):
        for h in hits:
            out.setdefault(h.chunk.id, h.to_dict())

    for d in ctx.get("defects", []):
        if d.get("type") and d.get("severity_band"):
            add(kb.for_defect(d["type"], d["severity_band"], trend, k=k_per_defect))
    if not ctx.get("defects"):
        fixed = _fixed(kb, "no-defects")
        if fixed:
            add([fixed])
    if any(k in ctx for k in ("rul_cycles", "health_score", "failure_risk")):
        fixed = _fixed(kb, "engine-health")
        if fixed:
            add([fixed])
    if ctx.get("progression"):
        fixed = _fixed(kb, "progression-growth")
        if fixed:
            add([fixed])
    fixed = _fixed(kb, "advisory-limits")
    if fixed:
        add([fixed])
    return list(out.values())


def _fixed(kb: KnowledgeBase, cid: str):
    from .rag import Retrieved
    chunk = kb.get(cid)
    return Retrieved(chunk, 1.0) if chunk else None


# ---------------------------------------------------------------- template
def build_template_report(ctx: Dict[str, Any], retrieved: Optional[List[Dict[str, Any]]] = None) -> InspectionReport:
    """Deterministic report. With `retrieved`, actions/intervals come from the cited chunks."""
    retrieved = retrieved or []
    ref = assess(ctx)
    by_key = {(r["defect_type"], r["severity"]): r for r in retrieved}
    findings: List[Finding] = []
    cited: List[str] = []
    for d in ctx.get("defects", []):
        parts = [f"{d['type'].replace('_', ' ')} detected"]
        if "confidence" in d:
            parts.append(f"with {round(d['confidence'] * 100)}% confidence")
        if "location" in d:
            parts.append(f"in the {d['location']}")
        obs = " ".join(parts)
        if "severity_band" in d:
            obs += f"; severity {d['severity_band']} (score {d['severity_score']})"
        obs += "."
        chunk = by_key.get((d["type"], d.get("severity_band")))
        if chunk:
            action, ids = chunk["action"], [chunk["id"]]
            cited.append(chunk["id"])
        else:
            action, ids = d.get("rule_based_action", "Verify the finding by inspection"), []
        findings.append(Finding(defect_id=d["id"], type=d["type"], observation=obs,
                                recommended_action=action, cited_chunks=ids))

    c = ctx["counts"]
    if c["defects_total"]:
        types = sorted({d["type"].replace("_", " ") for d in ctx["defects"]})
        summary = (f"{c['defects_total']} defect(s) detected ({', '.join(types)}). "
                   f"Overall risk is {ref['overall_risk']}. {ref['readiness_text']}.")
    else:
        summary = f"No defects were detected in this scan. Overall risk is {ref['overall_risk']}. {ref['readiness_text']}."

    engine_note = None
    if any(k in ctx for k in ("rul_cycles", "health_score", "failure_risk")):
        bits = []
        if "health_score" in ctx:
            bits.append(f"health score {ctx['health_score']}%")
        if "rul_cycles" in ctx:
            bits.append(f"estimated remaining useful life {ctx['rul_cycles']} cycles")
        if "failure_risk" in ctx:
            bits.append(f"failure risk {ctx['failure_risk']}")
        engine_note = "Engine condition: " + ", ".join(bits) + "."

    prog_note = None
    if ctx.get("progression"):
        prog_note = "; ".join(
            f"{p['type'].replace('_', ' ')} {p['trend']}" + (f" ({p['change_pct']:+}%)" if p["trend"] in ("growing", "shrinking", "stable") else "")
            for p in ctx["progression"]) + "."

    interval = None
    if retrieved and ctx.get("defects"):
        worst = max(ctx["defects"], key=lambda d: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(d.get("severity_band"), 0))
        ch = by_key.get((worst["type"], worst.get("severity_band")))
        if ch and ch.get("interval"):
            interval = ch["interval"]

    citations = sorted(set(cited))
    return InspectionReport(
        summary=summary, overall_risk=ref["overall_risk"], flight_recommendation=ref["flight_readiness"],
        findings=findings, engine_note=engine_note, progression_note=prog_note,
        inspection_interval=interval, citations=citations,
        limitations=[DISCLAIMER, LOC_NOTE],
    )


# ---------------------------------------------------------------- prompts
SYSTEM_PROMPT = """You are an aircraft maintenance reporting assistant.
Write a structured inspection report from the INSPECTION RECORD only.

HARD RULES
1. Use only facts, numbers, defect types and locations that appear in the INSPECTION RECORD{rag_clause}.
2. Never invent defects, measurements, part names, dates, intervals or regulations.
3. Report every shown defect id in the record as exactly one finding, with the same type. If defects_omitted is present, omitted defects are summarized only in counts; do not make findings for them.
4. Express confidence as a percentage (0.94 -> 94%). Do not invent other numbers.
5. overall_risk and flight_recommendation must be at least as cautious as the REFERENCE DECISION. You are advising, not releasing the aircraft.
6. Omit engine_note if the record has no engine data; omit progression_note if it has no progression.
{rag_rules}
Return ONLY a JSON object with these keys:
summary (string, 2-4 sentences), overall_risk ("LOW"|"MEDIUM"|"HIGH"),
flight_recommendation ("NO_GO"|"CONDITIONAL"|"MONITOR"|"NO_ACTION"),
findings (list of {{defect_id, type, observation, recommended_action, cited_chunks}}),
engine_note (string|null), progression_note (string|null), inspection_interval (string|null),
citations (list of chunk ids), limitations (list of strings)."""

RAG_CLAUSE = " or in the GUIDANCE CHUNKS"
RAG_RULES = """7. recommended_action and inspection_interval must be taken from the GUIDANCE CHUNKS (paraphrase allowed, no new numbers).
8. Every finding's cited_chunks must list the ids of the chunks you used; list all of them in citations. Cite only ids shown below."""


def build_messages(ctx: Dict[str, Any], mode: str, retrieved: List[Dict[str, Any]],
                   feedback: Optional[str] = None) -> List[Dict[str, str]]:
    rag = mode == "llm_rag"
    system = SYSTEM_PROMPT.format(rag_clause=RAG_CLAUSE if rag else "", rag_rules=RAG_RULES if rag else
                                  "7. cited_chunks and citations must be empty lists; no external guidance is provided.")
    ref = assess(ctx)
    prompt_ctx = compact_for_llm(ctx)
    user = f"INSPECTION RECORD (JSON)\n{context_to_text(prompt_ctx)}\n\nREFERENCE DECISION\n{json.dumps(ref)}\n"
    if rag:
        chunks = "\n".join(f"[{r['id']}] ({r['defect_type']}/{r['severity']}) action: {r['action']}"
                           f" | interval: {r['interval'] or 'n/a'} | {r['text']}" for r in retrieved)
        user += f"\nGUIDANCE CHUNKS\n{chunks}\n"
    if feedback:
        user += ("\nYOUR PREVIOUS ANSWER WAS REJECTED BY THE VERIFIER. Fix these problems and return the full JSON again:\n"
                 + feedback + "\n")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def _parse(text: str) -> InspectionReport:
    t = _FENCE.sub("", (text or "").strip())
    if not t.startswith("{"):
        m = re.search(r"\{.*\}", t, re.DOTALL)
        t = m.group(0) if m else t
    return InspectionReport.model_validate(json.loads(t))


# ---------------------------------------------------------------- main API
def generate_report(ctx: Dict[str, Any], mode: str = "llm_rag", llm: Optional[LLM] = None,
                    kb: Optional[KnowledgeBase] = None, guardrails: bool = True,
                    max_repairs: int = config.MAX_REPAIR_ATTEMPTS) -> GeneratedReport:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    t0 = time.time()
    rag = mode == "llm_rag"
    retrieved: List[Dict[str, Any]] = []
    if rag:
        retrieved = retrieve_for_context(ctx, kb or get_kb())

    if mode == "template":
        rep = build_template_report(ctx)
        v = verify_report(rep, ctx, [], require_citations=False)
        return GeneratedReport(rep, mode, [Attempt(rep, None, v)], False, [], v, time.time() - t0, "template")

    if llm is None:
        raise ValueError(f"mode '{mode}' requires an LLM client (set GROQ_API_KEY).")

    attempts: List[Attempt] = []
    feedback = None
    final: Optional[InspectionReport] = None
    final_v: Optional[Verification] = None
    for _ in range(1 + (max_repairs if guardrails else 0)):
        try:
            resp = llm.chat(build_messages(ctx, mode, retrieved, feedback), json_mode=True)
            raw = resp["content"]
            rep = _parse(raw)
        except (ValidationError, json.JSONDecodeError, RuntimeError, ValueError, KeyError) as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
            attempts.append(Attempt(None, err, None, locals().get("raw")))
            feedback = f"- [schema] your output was not valid for the required JSON schema: {err}"
            continue
        v = verify_report(
            rep,
            compact_for_llm(ctx),
            retrieved,
            require_citations=rag,
            decision_ctx=ctx,
        )
        attempts.append(Attempt(rep, None, v, raw))
        if v.ok:
            final, final_v = rep, v
            break
        feedback = v.feedback()
        final, final_v = rep, v  # keep last parsed report (used when guardrails are off)

    fallback = False
    if guardrails and (final is None or not final_v.ok):
        final = build_template_report(ctx, retrieved if rag else None)
        final_v = verify_report(final, ctx, retrieved, require_citations=rag)
        fallback = True
    elif final is None:  # guardrails off and nothing parsed
        final = build_template_report(ctx, retrieved if rag else None)
        final_v = verify_report(final, ctx, retrieved)
        fallback = True

    return GeneratedReport(final, mode, attempts, fallback, retrieved, final_v,
                           time.time() - t0, getattr(llm, "name", "llm"))


def report_to_markdown(r: InspectionReport) -> str:
    lines = [f"**Risk: {r.overall_risk}**  |  **Recommendation: {r.flight_recommendation.replace('_', ' ')}**", "", r.summary, ""]
    for f in r.findings:
        cite = f" _[{', '.join(f.cited_chunks)}]_" if f.cited_chunks else ""
        lines.append(f"- **{f.defect_id} · {f.type.replace('_', ' ')}**: {f.observation} → {f.recommended_action}{cite}")
    if r.engine_note:
        lines += ["", f"**Engine:** {r.engine_note}"]
    if r.progression_note:
        lines += ["", f"**Progression:** {r.progression_note}"]
    if r.inspection_interval:
        lines += ["", f"**Inspection interval:** {r.inspection_interval}"]
    if r.limitations:
        lines += ["", "_" + " ".join(r.limitations) + "_"]
    return "\n".join(lines)
