"""LLM-as-judge (independent of the deterministic verifier). Prefer a different model family
than the generator (default: Gemini judges Llama output)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

RUBRIC = """You are a strict reviewer of aircraft maintenance inspection reports.
You get the INSPECTION RECORD (ground truth), optional GUIDANCE CHUNKS (the only allowed source of
maintenance guidance) and a REPORT. Score the REPORT from 1 (very poor) to 5 (excellent):
- faithfulness: every number, defect, location and recommendation is supported by the record or the guidance chunks. Any invented fact caps this at 2.
- completeness: all defects, engine data, progression and the risk decision from the record are covered.
- actionability: recommended actions are specific, appropriate for the severity, and do not overstate certainty about airworthiness.
Also list any unsupported claims you found.
Return ONLY JSON: {"faithfulness": int, "completeness": int, "actionability": int, "unsupported_claims": [str]}"""


def judge_report(judge_llm, ctx: Dict[str, Any], report_json: Dict[str, Any],
                 retrieved: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    chunks = "\n".join(f"[{r['id']}] {r['text']} Action: {r['action']} Interval: {r['interval']}" for r in (retrieved or [])) or "(none provided)"
    user = (f"INSPECTION RECORD\n{json.dumps(ctx, indent=1)}\n\nGUIDANCE CHUNKS\n{chunks}\n\n"
            f"REPORT\n{json.dumps(report_json, indent=1)}")
    try:
        out = judge_llm.chat([{"role": "system", "content": RUBRIC}, {"role": "user", "content": user}], json_mode=True)
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (out["content"] or "").strip())
        d = json.loads(t)
        return {k: int(d[k]) for k in ("faithfulness", "completeness", "actionability")} | {
            "unsupported_claims": d.get("unsupported_claims", [])}
    except Exception:
        return None
