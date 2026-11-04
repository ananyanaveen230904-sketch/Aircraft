"""
Tool-calling maintenance agent.

* With a Groq key: Llama 3.3 decides which tools to call (OpenAI-style function
  calling), we execute them, feed results back, and loop until it answers.
* Without a key: a small rule-based router calls the same tools and composes a
  deterministic answer, so the demo still works offline.

Guardrail: the final answer is verified against everything the tools returned
in this turn. Numbers, intervals, locations or defect types that no tool
returned trigger one repair round; if it is still unsupported the answer is
shown with an explicit warning rather than silently trusted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import config
from .llm_payload import compact_for_llm
from .llm import LLM, redact_secrets
from .tools import CopilotTools
from .verifier import Verification, verify_answer_text

SYSTEM = """You are AeroVision Maintenance Copilot, an assistant for aircraft skin inspection.
You can call tools that run the real detector, severity scoring, scan comparison, RUL model, fusion, guidance retrieval and report generation.

RULES
- Never guess facts. Get them from tools. If a needed scan or engine is missing, say so.
- Resolve scans with list_scans when unsure; 'latest' and 'previous' are valid scan ids.
- For comparison requests, use the two most recent scans and state their ids in earlier-to-later order. If fewer than two are loaded, say so explicitly and summarize the available scan.
- For ANY question about safety, airworthiness, release or 'safe to fly', call assess_flight_readiness first (include previous_scan_id when the user mentions comparing or the last scan).
- For "what should I do" questions, call assess_flight_readiness or detect_defects first, then retrieve guidance for the worst-severity defect. If there are no defects, cite the no-defects guidance.
- Call analyze_image_with_vision when asked to look at, check, or verify an image, or when detection_quality.likely_unreliable is true and vision is available. Attribute those findings as "vision review"; it is advisory and must never lower rule-based severity, risk, or flight readiness.
- Repeat the detector reliability warning when a tool result includes it. Never treat vision review as a replacement for the detector-based decision.
- You only advise. Never say an aircraft is certified safe; the release decision belongs to a certified person. State the rule-based recommendation and its reasons.
- Use only numbers, locations, defect types and intervals that appear in tool outputs. Express confidence as percentages.
- When you use guidance from retrieve_guidance, cite the chunk id in square brackets, e.g. [crack-high].
- Be concise: short paragraphs or a few bullets."""


@dataclass
class AgentResult:
    answer: str
    steps: List[Dict[str, Any]] = field(default_factory=list)   # tool trace
    verification: Optional[Verification] = None
    warning: Optional[str] = None
    mode: str = "llm"


def _short(obj: Any) -> str:
    return json.dumps(compact_for_llm(obj), ensure_ascii=False, default=str)


def _wants_comparison(message: str) -> bool:
    low = message.lower()
    return any(term in low for term in (
        "compare", "previous", "last scan", "last one", "since", "grow", "progress",
    ))


def _comparison_line(results: List[Dict[str, Any]]) -> Optional[str]:
    for result in results:
        if not isinstance(result, dict):
            continue
        earlier = result.get("previous_scan_id")
        later = result.get("current_scan_id") or result.get("scan_id")
        if earlier and later:
            return f"Compared scans (earlier -> later): {earlier} -> {later}."
    return None


def _wants_vision_review(message: str) -> bool:
    low = message.lower()
    return any(phrase in low for phrase in (
        "look at this image", "look at the image", "look at image",
        "check this image", "check the image", "check image",
        "verify this image", "verify the image", "verify image",
        "analyze this image", "analyse this image",
    ))


def _quality_from(result: Any) -> Optional[Dict[str, Any]]:
    if isinstance(result, dict):
        quality = result.get("detection_quality")
        if isinstance(quality, dict):
            return quality
        for value in result.values():
            found = _quality_from(value)
            if found:
                return found
    elif isinstance(result, (list, tuple)):
        for value in result:
            found = _quality_from(value)
            if found:
                return found
    return None


def _reliability_warnings(results: List[Any]) -> List[str]:
    found = []

    def visit(value):
        if isinstance(value, dict):
            warning = value.get("warning")
            quality = value.get("detection_quality")
            if warning and isinstance(quality, dict) and quality.get("likely_unreliable"):
                found.append(str(warning))
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for result in results:
        visit(result)
    return list(dict.fromkeys(found))


class CopilotAgent:
    def __init__(
        self,
        tools: CopilotTools,
        llm: Optional[LLM] = None,
        max_steps: int = 6,
        require_llm: Optional[bool] = None,
        llm_reason: Optional[str] = None,
    ):
        self.tools = tools
        self.llm = llm
        self.max_steps = max_steps
        self.require_llm = require_llm
        self.llm_reason = llm_reason

    # ------------------------------------------------------------ public
    def run(self, user_message: str, history: Optional[List[Dict[str, str]]] = None) -> AgentResult:
        require_llm = self.require_llm
        if require_llm is None:
            config.reload_env()
            require_llm = config.AEROVISION_REQUIRE_LLM
        if self.llm is None:
            reason = self.llm_reason or "No LLM client is available."
            if require_llm:
                return AgentResult(
                    f"LLM is required, but unavailable. {reason}",
                    warning=reason,
                    mode="rules",
                )
            return self._run_rules(user_message)
        try:
            return self._run_llm(user_message, history or [])
        except Exception as e:  # network / rate limit / API error -> keep the demo alive
            reason = f"LLM unavailable (request failed: {redact_secrets(e)})"
            if require_llm:
                return AgentResult(
                    f"LLM is required, but the request failed. {reason}",
                    warning=reason,
                    mode="rules",
                )
            res = self._run_rules(user_message)
            res.warning = f"{reason}; answered with the offline rule-based router."
            return res

    # --------------------------------------------------------------- LLM
    def _run_llm(self, user_message: str, history: List[Dict[str, str]]) -> AgentResult:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
        messages += [m for m in history[-8:] if m["role"] in ("user", "assistant")]
        messages.append({"role": "user", "content": user_message})
        steps: List[Dict[str, Any]] = []
        outputs: List[Any] = []
        answer = ""
        repaired = False
        schemas = self.tools.schemas()
        comparison_notice = None
        vision_called = False

        if _wants_vision_review(user_message) and self.tools.vision_llm is not None:
            vision_result = self.tools.call("analyze_image_with_vision", {})
            steps.append({"tool": "analyze_image_with_vision", "arguments": {}, "result": vision_result})
            outputs.append(vision_result)
            vision_called = "error" not in vision_result
            messages.append({
                "role": "user",
                "content": (
                    "Vision review result (advisory; attribute findings as vision review and never lower "
                    "rule-based risk): " + _short(vision_result)
                ),
            })

        if _wants_comparison(user_message):
            scan_result = self.tools.call("list_scans", {})
            steps.append({"tool": "list_scans", "arguments": {}, "result": scan_result})
            outputs.append(scan_result)
            loaded = scan_result.get("scans", [])
            if len(loaded) < 2:
                if loaded:
                    comparison_notice = "Only one scan is loaded, so there is nothing to compare with yet."
                    single = self.tools.call("assess_flight_readiness", {"scan_id": loaded[-1]["scan_id"]})
                    steps.append({
                        "tool": "assess_flight_readiness",
                        "arguments": {"scan_id": loaded[-1]["scan_id"]},
                        "result": single,
                    })
                    outputs.append(single)
                    details = json.dumps(compact_for_llm(single), ensure_ascii=False, default=str)
                    counts = single["counts"]
                    single_summary = (
                        f"Single-scan result for {single['scan_id']}: {counts['defects_total']} defect(s) "
                        f"({counts['high']} high, {counts['medium']} medium, {counts['low']} low); "
                        f"advisory risk {single['overall_risk']}. {single['readiness_text']}."
                    )
                else:
                    comparison_notice = "No scans are loaded, so there is nothing to compare with yet."
                    details = "No scans are currently loaded."
                    single_summary = ""
                messages.append({
                    "role": "user",
                    "content": (
                        f"Comparison preflight: {comparison_notice} Do not claim a comparison. "
                        f"Give the available single-scan result based on this data: {details}"
                    ),
                })
            else:
                earlier = loaded[-2]["scan_id"]
                later = loaded[-1]["scan_id"]
                comparison_result = self.tools.call(
                    "compare_scans",
                    {"previous_scan_id": earlier, "current_scan_id": later},
                )
                steps.append({
                    "tool": "compare_scans",
                    "arguments": {"previous_scan_id": earlier, "current_scan_id": later},
                    "result": comparison_result,
                })
                outputs.append(comparison_result)
                messages.append({
                    "role": "user",
                    "content": (
                        "Comparison requested. The comparison tool result below is chronological; state "
                        f"the scan ids (earlier -> later): {json.dumps(compact_for_llm(comparison_result), ensure_ascii=False)}"
                    ),
                })

        for _ in range(self.max_steps):
            resp = self.llm.chat(messages, tools=schemas)
            calls = resp.get("tool_calls") or []
            if calls:
                messages.append({
                    "role": "assistant", "content": resp.get("content") or "",
                    "tool_calls": [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                   for c in calls]})
                for c in calls:
                    result = self.tools.call(c["name"], c["arguments"])
                    steps.append({"tool": c["name"], "arguments": c["arguments"], "result": result})
                    outputs.append(result)
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": _short(result)})
                    quality = _quality_from(result)
                    if (
                        quality and quality.get("likely_unreliable")
                        and self.tools.vision_llm is not None and not vision_called
                    ):
                        vision_result = self.tools.call(
                            "analyze_image_with_vision",
                            {"scan_id": result.get("scan_id", "latest")},
                        )
                        steps.append({
                            "tool": "analyze_image_with_vision",
                            "arguments": {"scan_id": result.get("scan_id", "latest")},
                            "result": vision_result,
                        })
                        outputs.append(vision_result)
                        vision_called = "error" not in vision_result
                        messages.append({
                            "role": "user",
                            "content": (
                                "Vision review result (advisory; attribute findings as vision review and "
                                "never lower rule-based risk): " + _short(vision_result)
                            ),
                        })
                continue

            answer = (resp.get("content") or "").strip()
            warnings = _reliability_warnings(outputs)
            if warnings:
                answer = (answer + "\n\n" + "\n".join(warnings)).strip()
            comparison = _comparison_line(outputs)
            if comparison and comparison not in answer:
                answer = f"{answer}\n\n{comparison}".strip()
            if comparison_notice:
                prefix = comparison_notice
                if single_summary:
                    prefix += "\n\n" + single_summary
                if prefix not in answer:
                    answer = f"{prefix}\n\n{answer}".strip()
            v = verify_answer_text(answer, outputs) if outputs else Verification(ok=True)
            if v.ok or repaired:
                warning = None if v.ok else (
                    "Some statements could not be traced to tool outputs and should be double-checked:\n"
                    + "\n".join(f"- {i.detail}" for i in v.issues))
                return AgentResult(answer, steps, v, warning, "llm")
            repaired = True
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": (
                "Verifier rejected your answer because these claims are not in the tool outputs:\n"
                + v.feedback() + "\nRewrite the answer using only facts from the tool outputs.")})
        return AgentResult(answer or "I could not finish within the tool-call limit.", steps,
                           warning="Stopped at the tool-call limit.", mode="llm")

    # --------------------------------------------------------- rule-based
    def _run_rules(self, msg: str) -> AgentResult:
        low = msg.lower()
        steps: List[Dict[str, Any]] = []

        def call(name, **kw):
            r = self.tools.call(name, kw)
            steps.append({"tool": name, "arguments": kw, "result": r})
            return r

        scans = call("list_scans")
        ids = [s["scan_id"] for s in scans.get("scans", [])]
        if not ids:
            answer = "No scans are loaded. Upload or select a scan first."
            if _wants_comparison(msg):
                answer = "No scans are loaded, so there is nothing to compare with yet. " + answer
            return AgentResult(answer, steps, mode="rules")

        mentioned = [i for i in ids if i.lower() in low]
        want_compare = _wants_comparison(msg)
        m = re.search(r"engine\s*(?:id\s*)?#?(\d+)", low)
        engine = int(m.group(1)) if m else None
        scan = mentioned[-1] if mentioned else "latest"
        prev = None
        if len(mentioned) >= 2:
            prev, scan = mentioned[-2], mentioned[-1]
        elif want_compare and len(ids) >= 2:
            prev = "previous"

        parts: List[str] = []
        want_action = any(w in low for w in (
            "how to", "repair", "treat", "guidance", "procedure", "what should", "what do",
            "what to do", "recommend", "fix",
        )) and not any(w in low for w in ("safe", "fly"))

        r = call("assess_flight_readiness", scan_id=scan, previous_scan_id=prev, engine_id=engine) if prev or engine is not None \
            else call("assess_flight_readiness", scan_id=scan)
        if "error" in r:
            return AgentResult(f"Could not complete the request: {r['error']}", steps, mode="rules")

        if want_action:
            defects = r.get("defects", [])
            if not defects:
                guidance = call(
                    "retrieve_guidance",
                    query="no defects detected",
                    chunk_id="no-defects",
                    k=1,
                )
                chunks = guidance.get("chunks", [])
                if chunks:
                    chunk = chunks[0]
                    parts.append(f"No defects were detected in scan {r['scan_id']}.")
                    parts.append(f"{chunk['action']} [{chunk['id']}].")
                else:
                    parts.append(f"No defects were detected in scan {r['scan_id']}; no no-defects guidance chunk is available.")
            else:
                worst = max(
                    defects,
                    key=lambda d: (
                        {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(d.get("severity_band"), 0),
                        float(d.get("severity_score") or 0),
                        float(d.get("confidence") or 0),
                    ),
                )
                parts.append(
                    f"The highest-severity finding in scan {r['scan_id']} is {worst['id']}: "
                    f"{worst['type']} ({worst.get('severity_band', 'UNKNOWN')}, "
                    f"score {worst.get('severity_score', 'unknown')})"
                    + (f" in the {worst['location']}" if worst.get("location") else "")
                    + "."
                )
                guidance = call(
                    "retrieve_guidance",
                    query=f"{worst['type']} {worst.get('severity_band', '')} recommended action",
                    defect_type=worst["type"],
                    severity=worst.get("severity_band"),
                    k=1,
                )
                for chunk in guidance.get("chunks", []):
                    parts.append(f"Recommended action: {chunk['action']} [{chunk['id']}].")

            if r.get("previous_scan_id"):
                parts.append(
                    f"Compared scans (earlier -> later): {r['previous_scan_id']} -> {r['scan_id']}."
                )
            if r.get("warning"):
                parts.append(r["warning"])
            if r.get("detection_quality", {}).get("likely_unreliable") and self.tools.vision_llm is not None:
                parts.append("For an independent review, click “Analyze with vision AI”.")
            answer = "\n\n".join(parts)
            v = verify_answer_text(answer, [s["result"] for s in steps])
            if want_compare and len(ids) == 1:
                answer = "Only one scan is loaded, so there is nothing to compare with yet.\n\n" + answer
                v = verify_answer_text(answer, [s["result"] for s in steps])
            return AgentResult(answer, steps, v, None if v.ok else "; ".join(i.detail for i in v.issues), "rules")

        c = r["counts"]
        head = f"Scan {r['scan_id']}: {c['defects_total']} defect(s) detected"
        if c["defects_total"]:
            head += f" ({c['high']} high, {c['medium']} medium, {c['low']} low severity)"
        parts.append(head + ".")
        if r.get("previous_scan_id"):
            parts.append(f"Compared with {r['previous_scan_id']}: {c['growing']} growing, {c['new']} new, {c['resolved']} resolved.")
            parts.append(
                f"Compared scans (earlier -> later): {r['previous_scan_id']} -> {r['scan_id']}."
            )
        if "rul_cycles" in r:
            parts.append(f"Engine: RUL {r['rul_cycles']} cycles, health {r['health_score']}%, failure risk {r['failure_risk']}.")
        elif r.get("rul_unavailable_reason"):
            parts.append(f"Engine RUL unavailable: {r['rul_unavailable_reason']}")
        parts.append(f"Advisory risk: {r['overall_risk']}. {r['readiness_text']}.")
        parts.append("Reasons: " + "; ".join(r["reasons"]) + ".")
        if r.get("warning"):
            parts.append(r["warning"])
        if r.get("detection_quality", {}).get("likely_unreliable") and self.tools.vision_llm is not None:
            parts.append("For an independent review, click “Analyze with vision AI”.")

        # guidance for the worst defect, with citation
        worst = sorted(r["defects"], key=lambda d: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(d.get("severity_band"), 0), reverse=True)[:1]
        if worst:
            g = call("retrieve_guidance", query=f"{worst[0]['type']} {worst[0]['severity_band']} action",
                     defect_type=worst[0]["type"], severity=worst[0]["severity_band"], k=1)
            for ch in g.get("chunks", []):
                parts.append(f"Guidance for the most severe defect: {ch['action']} [{ch['id']}].")
        if want_compare and len(ids) == 1:
            parts.insert(0, "Only one scan is loaded, so there is nothing to compare with yet.")
        parts.append("This is an advisory output; a certified person must confirm it.")
        answer = "\n\n".join(parts)
        v = verify_answer_text(answer, [s["result"] for s in steps])
        return AgentResult(answer, steps, v, None if v.ok else "; ".join(i.detail for i in v.issues), "rules")
