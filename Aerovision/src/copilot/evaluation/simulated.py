"""
SIMULATED LLM for testing the evaluation *plumbing* offline.

It builds a correct template report and then randomly injects typical LLM faults
(wrong number, invented defect type, bad citation, understated risk, dropped finding).
Numbers produced with it say NOTHING about real model quality; run_eval prints a
banner and writes simulated results to a separate folder.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List

from src.copilot.llm import LLM
from src.copilot.report_generator import build_template_report


class SimulatedLLM(LLM):
    name = "SIMULATED"

    def __init__(self, seed: int = 0, p_fault_plain: float = 0.45, p_fault_rag: float = 0.2):
        self.rng = random.Random(seed)
        self.p_plain, self.p_rag = p_fault_plain, p_fault_rag
        self.ctx: Dict[str, Any] = {}
        self.retrieved: List[Dict[str, Any]] = []

    def set_context(self, ctx, retrieved=None):
        self.ctx, self.retrieved = ctx, retrieved or []

    def chat(self, messages, tools=None, json_mode=False, temperature=0.0):
        user = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        rag = "GUIDANCE CHUNKS" in user
        rep = build_template_report(self.ctx, self.retrieved if rag else None).model_dump()
        repaired = "REJECTED" in user
        p = (self.p_rag if rag else self.p_plain) * (0.2 if repaired else 1.0)
        if self.rng.random() < p:
            fault = self.rng.choice(["number", "defect", "citation", "risk", "drop", "interval"])
            if fault == "number":
                rep["summary"] += " Estimated 37 cycles of service margin remain."
            elif fault == "defect":
                have = {d["type"] for d in self.ctx.get("defects", [])}
                extra = next((t for t in ("corrosion", "crack", "dent") if t not in have), None)
                if extra:
                    rep["summary"] += f" Surface {extra} is also visible."
            elif fault == "citation":
                if rep["findings"]:
                    rep["findings"][0]["cited_chunks"] = ["faa-ac-43-13-ch9"]
            elif fault == "risk":
                rep["overall_risk"] = "LOW"; rep["flight_recommendation"] = "NO_ACTION"
            elif fault == "drop":
                rep["findings"] = rep["findings"][:-1]
            elif fault == "interval":
                rep["inspection_interval"] = "within 72 hours"
        return {"content": json.dumps(rep), "tool_calls": []}
