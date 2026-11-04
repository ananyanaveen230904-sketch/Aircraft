"""
Guardrail: every claim in a generated report (or chat answer) must trace
to the inspection record or to a retrieved guidance chunk.

Checks (all deterministic, no LLM involved):
  numbers      every number (and number-word before a defect noun) must exist in the grounding
  time phrases "within 24 hours"-style phrases must literally exist in the grounding text
  defect types a defect type may only be asserted if it is in the record (negated / hedged
               mentions such as "no corrosion" or "check for cracks" are allowed)
  locations    positional labels must be ones in the record
  ids          finding ids must exist and match the record's defect type
  citations    cited chunk ids must be in the retrieved set; in RAG mode every
               recommended action must carry a citation
  coverage     every detected defect must have a finding
  risk         overall_risk / flight_recommendation must not understate the rule-based decision
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Set

from .decision import READINESS_RANK, assess
from .context import RANK
from .schemas import InspectionReport

# ------------------------------------------------------------------ vocab
TYPE_WORDS = {
    "crack": ["crack", "cracks", "cracking", "cracked"],
    "corrosion": ["corrosion", "corroded", "corrode", "rust", "rusting"],
    "dent": ["dent", "dents", "dented", "denting"],
    "rivet_damage": ["rivet", "rivets", "rivet damage"],
}
LOCATION_RE = re.compile(
    r"(?<![-\w])((?:upper|center|lower)-(?:left|center|right) region|center region)\b", re.I)
NUM_RE = re.compile(r"(?<![\w.\-])(\d+(?:\.\d+)?)(?![\w])")
PHRASE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(flight hours?|hours?|hrs?|days?|weeks?|months?|cycles?)\b", re.I)
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}
COUNT_NOUNS = r"(?:defects?|cracks?|corrosion|dents?|rivets?|findings?|regions?|areas?|indications?|scans?)"
NUMWORD_RE = re.compile(rf"\b({'|'.join(NUMBER_WORDS)})\b\s+(?:\w+\s+)?{COUNT_NOUNS}\b", re.I)
# identifiers that contain digits but are not claims
IGNORE_RE = re.compile(
    r"\bAC\s*43\.13-1B\b|\b14\s*CFR\b|\bPart\s*43\b|\bCh\.?\s*\d+\b|\b[DP]\d+\b|\bC-?MAPSS\b|\bFD00\d\b", re.I)
NEGATION_CTX = re.compile(
    r"(no|not|without|absence of|absent|none|never|n't|nor|free of|rule out|rules out|ruling out|"
    r"check for|checks for|checking for|inspect for|inspecting for|look for|looking for|for any|"
    r"hidden|possible|potential|potentially|suspected|may|might|could|if)\W+(?:\w+\W+){0,3}$", re.I)


@dataclass
class Issue:
    code: str
    detail: str
    level: str = "error"  # error | warning

    def to_dict(self):
        return asdict(self)


@dataclass
class Verification:
    ok: bool
    issues: List[Issue] = field(default_factory=list)
    n_claims: int = 0
    n_unsupported: int = 0

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def faithfulness(self) -> float:
        return 1.0 if self.n_claims == 0 else max(0.0, 1 - self.n_unsupported / self.n_claims)

    def feedback(self) -> str:
        return "\n".join(f"- [{i.code}] {i.detail}" for i in self.errors)


# --------------------------------------------------------------- grounding
@dataclass
class Grounding:
    numbers: Set[float] = field(default_factory=set)
    types: Set[str] = field(default_factory=set)
    locations: Set[str] = field(default_factory=set)
    phrases: Set[str] = field(default_factory=set)       # normalized "24 hour"
    chunk_ids: Set[str] = field(default_factory=set)      # retrieved chunk ids
    text: str = ""

    @classmethod
    def from_objects(cls, objs: Iterable[Any], chunk_ids: Iterable[str] = ()) -> "Grounding":
        g = cls()
        g.chunk_ids = set(chunk_ids)
        parts: List[str] = []

        def add_num(v: float):
            g.numbers.add(round(float(v), 4))
            if 0 < abs(v) <= 1:               # confidences appear as percentages in prose
                g.numbers.add(round(abs(v) * 100, 4))

        def walk(o: Any):
            if isinstance(o, bool):
                return
            if isinstance(o, (int, float)):
                add_num(o)
            elif isinstance(o, str):
                parts.append(o)
            elif isinstance(o, dict):
                for k, v in o.items():
                    parts.append(str(k))
                    walk(v)
            elif isinstance(o, (list, tuple, set)):
                for v in o:
                    walk(v)
            elif hasattr(o, "__dict__"):
                walk(vars(o))

        for o in objs:
            walk(o)
        g.text = " ".join(parts)
        clean = IGNORE_RE.sub(" ", g.text)
        for m in NUM_RE.finditer(clean):
            add_num(float(m.group(1)))
        for m in PHRASE_RE.finditer(g.text):
            g.phrases.add(_norm_phrase(m.group(1), m.group(2)))
        for m in LOCATION_RE.finditer(g.text):
            g.locations.add(m.group(1).lower())
        low = g.text.lower()
        for t, words in TYPE_WORDS.items():
            if t in low or any(re.search(rf"\b{re.escape(w)}\b", low) for w in words[:1]):
                g.types.add(t)
        return g

    @classmethod
    def from_context(cls, ctx: Dict[str, Any], retrieved: Iterable[Dict[str, Any]] = ()) -> "Grounding":
        retrieved = list(retrieved)
        g = cls.from_objects([ctx, *retrieved], chunk_ids=[r["id"] for r in retrieved])
        # defect types allowed = types actually *detected*, not words in guidance chunks
        g.types = {d["type"] for d in ctx.get("defects", []) if d.get("type")}
        g.types |= {p["type"] for p in ctx.get("progression", []) if p.get("type")}  # e.g. a resolved defect
        g.locations = {d["location"].lower() for d in ctx.get("defects", []) if d.get("location")}
        vision = ctx.get("vision_review")
        if isinstance(vision, dict):
            g.types |= {
                d["type"] for d in vision.get("defects", [])
                if isinstance(d, dict) and d.get("type") in TYPE_WORDS
            }
            g.locations |= {
                d["region"].lower() for d in vision.get("defects", [])
                if isinstance(d, dict) and d.get("region")
            }
        if "rul_cycles" in ctx:
            g.phrases.add(_norm_phrase(str(ctx["rul_cycles"]), "cycles"))
        # "Scan 1 / Scan 2" are harmless labels when two scans are compared
        if ctx.get("progression") or ctx.get("previous_scan_id"):
            g.numbers.update({1.0, 2.0})
        return g


def _norm_phrase(num: str, unit: str) -> str:
    u = unit.lower()
    u = re.sub(r"^flight\s+", "", u)
    u = {"hrs": "hour", "hr": "hour", "hours": "hour", "hour": "hour", "days": "day", "day": "day",
         "weeks": "week", "week": "week", "months": "month", "month": "month",
         "cycles": "cycle", "cycle": "cycle"}.get(u, u)
    n = float(num)
    return f"{int(n) if n.is_integer() else n} {u}"


def _num_supported(v: float, allowed: Set[float]) -> bool:
    is_int = float(v).is_integer()
    tol = 0.5 if is_int else 0.051
    return any(abs(v - a) <= tol or abs(v - abs(a)) <= tol for a in allowed)


# --------------------------------------------------------------- text checks
def check_text(text: str, g: Grounding, where: str, check_types: bool = True,
               ) -> (List[Issue], int, int):
    """Returns (issues, n_claims, n_unsupported) for a free-text field."""
    issues: List[Issue] = []
    claims = bad = 0
    if not text:
        return issues, 0, 0
    clean = IGNORE_RE.sub(" ", text)

    # numbers
    for m in NUM_RE.finditer(clean):
        claims += 1
        v = float(m.group(1))
        if not _num_supported(v, g.numbers):
            bad += 1
            issues.append(Issue("unsupported_number", f"{where}: number {m.group(1)} is not in the record or retrieved guidance"))
    for m in NUMWORD_RE.finditer(clean):
        claims += 1
        v = NUMBER_WORDS[m.group(1).lower()]
        if not _num_supported(float(v), g.numbers):
            bad += 1
            issues.append(Issue("unsupported_number", f"{where}: '{m.group(0)}' implies a count ({v}) that is not in the record"))

    # time phrases
    for m in PHRASE_RE.finditer(clean):
        claims += 1
        if _norm_phrase(m.group(1), m.group(2)) not in g.phrases:
            bad += 1
            issues.append(Issue("unsupported_interval", f"{where}: '{m.group(0)}' does not appear in the record or retrieved guidance"))

    # locations
    for m in LOCATION_RE.finditer(text):
        claims += 1
        if m.group(1).lower() not in g.locations:
            bad += 1
            issues.append(Issue("unsupported_location", f"{where}: location '{m.group(1)}' is not a location in the record"))

    # defect types
    if check_types:
        low = text.lower()
        for t, words in TYPE_WORDS.items():
            if t in g.types:
                continue
            for w in words:
                for m in re.finditer(rf"\b{re.escape(w)}\b", low):
                    claims += 1
                    before = low[max(0, m.start() - 60): m.start()]
                    if NEGATION_CTX.search(before):
                        continue  # "no corrosion", "check for cracks" -> not an assertion
                    bad += 1
                    issues.append(Issue("unsupported_defect", f"{where}: mentions '{w}' but no {t.replace('_', ' ')} is in the record"))
                    break
                else:
                    continue
                break
    return issues, claims, bad


# --------------------------------------------------------------- report check
def verify_report(report: InspectionReport, ctx: Dict[str, Any],
                  retrieved: Optional[List[Dict[str, Any]]] = None,
                  require_citations: bool = False,
                  decision_ctx: Optional[Dict[str, Any]] = None) -> Verification:
    retrieved = retrieved or []
    g = Grounding.from_context(ctx, retrieved)
    issues: List[Issue] = []
    claims = bad = 0

    def run(text, where, check_types=True):
        nonlocal claims, bad
        i, c, b = check_text(text or "", g, where, check_types)
        issues.extend(i); claims += c; bad += b

    run(report.summary, "summary")
    run(report.engine_note, "engine_note")
    run(report.progression_note, "progression_note")
    run(report.inspection_interval, "inspection_interval", check_types=False)

    defect_by_id = {d["id"]: d for d in ctx.get("defects", [])}
    seen_ids = set()
    for f in report.findings:
        where = f"finding {f.defect_id}"
        claims += 1
        d = defect_by_id.get(f.defect_id)
        if d is None:
            bad += 1
            issues.append(Issue("unknown_defect_id", f"{where}: id not in the record"))
        else:
            seen_ids.add(f.defect_id)
            if d.get("type") != f.type:
                bad += 1
                issues.append(Issue("type_mismatch", f"{where}: reports type {f.type}, record says {d.get('type')}"))
        run(f.observation, f"{where} observation")
        # actions come from guidance, so defect words there are expected (type check off)
        run(f.recommended_action, f"{where} action", check_types=False)
        for cid in f.cited_chunks:
            claims += 1
            if cid not in g.chunk_ids:
                bad += 1
                issues.append(Issue("invalid_citation", f"{where}: cited chunk '{cid}' was not retrieved"))
        if require_citations and not f.cited_chunks:
            claims += 1; bad += 1
            issues.append(Issue("uncited_action", f"{where}: recommended action has no citation to retrieved guidance"))

    for cid in report.citations:
        claims += 1
        if cid not in g.chunk_ids:
            bad += 1
            issues.append(Issue("invalid_citation", f"report: cited chunk '{cid}' was not retrieved"))

    # coverage
    for did in defect_by_id:
        if did not in seen_ids:
            issues.append(Issue("missing_finding", f"defect {did} ({defect_by_id[did].get('type')}) has no finding"))
    # engine facts must not be invented
    if report.engine_note and not any(k in ctx for k in ("rul_cycles", "health_score", "failure_risk")):
        claims += 1; bad += 1
        issues.append(Issue("unsupported_engine_claim", "engine_note present but the record has no engine data"))
    if report.progression_note and not ctx.get("progression"):
        claims += 1; bad += 1
        issues.append(Issue("unsupported_progression_claim", "progression_note present but the record has no progression data"))

    # decision consistency
    ref = assess(decision_ctx or ctx)
    # decision issues are tracked separately (safety), not as hallucinated claims
    if RANK[report.overall_risk] < RANK[ref["overall_risk"]]:
        issues.append(Issue("risk_understated", f"overall_risk {report.overall_risk} is below the rule-based {ref['overall_risk']}"))
    elif RANK[report.overall_risk] > RANK[ref["overall_risk"]]:
        issues.append(Issue("risk_overstated", f"overall_risk {report.overall_risk} is above the rule-based {ref['overall_risk']}", "warning"))
    if READINESS_RANK[report.flight_recommendation] < READINESS_RANK[ref["flight_readiness"]]:
        issues.append(Issue("readiness_understated", f"flight_recommendation {report.flight_recommendation} is less cautious than rule-based {ref['flight_readiness']}"))
    elif READINESS_RANK[report.flight_recommendation] > READINESS_RANK[ref["flight_readiness"]]:
        issues.append(Issue("readiness_overstated", f"flight_recommendation {report.flight_recommendation} is more cautious than rule-based {ref['flight_readiness']}", "warning"))

    ok = not any(i.level == "error" for i in issues)
    return Verification(ok=ok, issues=issues, n_claims=claims, n_unsupported=bad)


def verify_answer_text(text: str, grounding_objs: Iterable[Any]) -> Verification:
    """Guardrail for free-text chat answers, grounded on all tool outputs of the turn."""
    objs = list(grounding_objs)
    g = Grounding.from_objects(objs)
    # for chat, any defect type present in any tool output counts as grounded
    low = g.text.lower()
    g.types = {t for t in TYPE_WORDS if t in low or any(re.search(rf"\b{w}\b", low) for w in TYPE_WORDS[t])}
    vision_reports = _vision_reports(objs)
    vision_types = {
        d["type"] for report in vision_reports for d in report.get("defects", [])
        if isinstance(d, dict) and d.get("type") in TYPE_WORDS
    }
    vision_regions = {
        d["region"].lower() for report in vision_reports for d in report.get("defects", [])
        if isinstance(d, dict) and d.get("region")
    }
    g.types |= vision_types
    g.locations |= vision_regions
    issues, claims, bad = check_text(text, g, "answer")
    for issue in _check_vision_attribution(text, vision_reports, vision_types, vision_regions):
        issues.append(issue)
        claims += 1
        bad += 1
    return Verification(ok=not issues, issues=issues, n_claims=claims, n_unsupported=bad)


def _vision_reports(objects: Iterable[Any]) -> List[Dict[str, Any]]:
    reports = []

    def visit(value):
        if isinstance(value, dict):
            review = value.get("vision_review")
            if isinstance(review, dict) and "scene_type" in review and "defects" in review:
                reports.append(review)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for obj in objects:
        visit(obj)
    return reports


def _check_vision_attribution(text, reports, vision_types, vision_regions):
    issues = []
    low = text.lower()
    if not reports:
        if re.search(
            r"\bvision review\b.{0,100}\b(?:found|identified|shows|reported|detected|"
            r"noted|observed|indicates|classified)\b",
            low,
        ):
            issues.append(Issue(
                "unsupported_vision_fact",
                "the answer describes vision review findings, but no vision_review result is available",
            ))
        return issues

    if "vision review" not in low:
        if any(re.search(rf"\b{re.escape(word)}\b", low)
               for words in TYPE_WORDS.values() for word in words):
            issues.append(Issue(
                "missing_vision_attribution",
                "vision findings must be explicitly attributed as 'vision review'",
            ))
            return issues

    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    for sentence in sentences:
        sentence_low = sentence.lower()
        if "vision review" not in sentence_low:
            if any(re.search(rf"\b{re.escape(word)}\b", sentence_low)
                   for t in vision_types for word in TYPE_WORDS[t]):
                issues.append(Issue(
                    "missing_vision_attribution",
                    "vision defect findings must be attributed as 'vision review'",
                ))
            continue

        mentioned = {
            defect_type
            for defect_type, words in TYPE_WORDS.items()
            if any(re.search(rf"\b{re.escape(word)}\b", sentence_low) for word in words)
        }
        for defect_type in mentioned - vision_types:
            issues.append(Issue(
                "unsupported_vision_defect",
                f"vision review does not report {defect_type.replace('_', ' ')}",
            ))

        for match in LOCATION_RE.finditer(sentence):
            if match.group(1).lower() not in vision_regions:
                issues.append(Issue(
                    "unsupported_vision_region",
                    f"vision review does not report region '{match.group(1)}'",
                ))

        count_match = re.search(r"vision review.{0,80}\b(\d+)\s+(?:visible\s+)?defects?\b", sentence_low)
        if count_match and int(count_match.group(1)) != sum(
            len(report.get("defects", [])) for report in reports
        ):
            issues.append(Issue(
                "unsupported_vision_count",
                "the stated vision review defect count does not match the vision_review result",
            ))
    return issues
