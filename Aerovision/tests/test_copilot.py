"""Run with:  python -m pytest tests -q     (no API keys needed; LLMs are scripted)"""
import json
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.copilot.agent import CopilotAgent, compact_for_llm
from src.copilot.context import build_context
from src.copilot.decision import assess
from src.copilot.llm import ScriptedLLM
from src.copilot.rag import KnowledgeBase
from src.copilot.report_generator import _fixed, build_messages, build_template_report, generate_report, retrieve_for_context
from src.copilot.rag import MissingKnowledgeChunksError
from src.copilot.vision import analyze_scan
from src.copilot.tools import CopilotTools, ScanStore
from src.copilot.verifier import verify_answer_text, verify_report

DETS = [
    dict(defect_type="crack", confidence=0.94, bbox=(100, 100, 60, 20), relative_area=0.004, length_cm=2.1),
    dict(defect_type="corrosion", confidence=0.8, bbox=(300, 300, 120, 100), relative_area=0.05),
]


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(backend="tfidf")


@pytest.fixture()
def ctx():
    return build_context(DETS, (480, 640), rul_cycles=18, health_score=63, failure_risk="medium")


def test_retrieval_matches_defect_and_severity(kb):
    hits = kb.for_defect("crack", "HIGH")
    assert hits[0].chunk.id == "crack-high"
    assert kb.search("rust on wing skin", k=1)[0].chunk.defect_type == "corrosion"


def test_template_passes_verifier(ctx, kb):
    rep = build_template_report(ctx)
    assert verify_report(rep, ctx).ok
    ret = retrieve_for_context(ctx, kb)
    rep = build_template_report(ctx, ret)
    assert verify_report(rep, ctx, ret, require_citations=True).ok


def test_verifier_rejects_hallucinations(ctx, kb):
    ret = retrieve_for_context(ctx, kb)
    good = build_template_report(ctx, ret)
    bad = good.model_copy(deep=True)
    bad.summary = "3 cracks and a dent found, RUL 25 cycles, inspect within 48 hours."
    bad.overall_risk = "LOW"
    bad.findings[0].cited_chunks = ["made-up-chunk"]
    codes = {i.code for i in verify_report(bad, ctx, ret, True).issues}
    assert {"unsupported_number", "unsupported_defect", "unsupported_interval",
            "invalid_citation", "risk_understated"} <= codes


def test_negated_defect_mentions_are_allowed(ctx, kb):
    ret = retrieve_for_context(ctx, kb)
    rep = build_template_report(ctx, ret)
    rep.summary += " No dent was detected; check for hidden cracks near the rivet line."
    assert verify_report(rep, ctx, ret, True).ok


def test_fallback_when_llm_hallucinates(ctx, kb):
    ret = retrieve_for_context(ctx, kb)
    bad = build_template_report(ctx, ret).model_copy(deep=True)
    bad.summary = "Found 9 dents."
    llm = ScriptedLLM([{"content": bad.model_dump_json()}])
    g = generate_report(ctx, "llm_rag", llm, kb)
    assert g.fallback_used and g.verification.ok and len(g.attempts) == 2


def test_repair_succeeds_on_second_try(ctx, kb):
    ret = retrieve_for_context(ctx, kb)
    good = build_template_report(ctx, ret)
    bad = good.model_copy(deep=True); bad.summary = "Found 9 dents."
    llm = ScriptedLLM([{"content": bad.model_dump_json()}, {"content": good.model_dump_json()}])
    g = generate_report(ctx, "llm_rag", llm, kb)
    assert not g.fallback_used and g.verification.ok and len(g.attempts) == 2


def test_invalid_json_falls_back(ctx, kb):
    g = generate_report(ctx, "llm_rag", ScriptedLLM(["not json"]), kb)
    assert g.fallback_used and g.attempts[0].report is None


def test_guardrails_off_keeps_raw_attempt(ctx, kb):
    bad = build_template_report(ctx).model_copy(deep=True); bad.summary = "Found 9 dents."
    g = generate_report(ctx, "llm", ScriptedLLM([{"content": bad.model_dump_json()}]), kb, guardrails=False)
    assert not g.first_attempt.verification.ok and len(g.attempts) == 1


def test_decision_rules():
    clean = build_context([], (480, 640))
    assert assess(clean)["flight_readiness"] == "NO_ACTION"
    healthy_crack = build_context([dict(defect_type="dent", confidence=0.5, bbox=(5, 5, 10, 10), relative_area=0.0005)], (480, 640))
    assert assess(healthy_crack)["overall_risk"] == "LOW"
    prog = [dict(defect_type="dent", trend="growing", change_pct=50.0, old_area=100, new_area=150)]
    grown = build_context([dict(defect_type="dent", confidence=0.5, bbox=(5, 5, 10, 10), relative_area=0.0005)], (480, 640), progression=prog)
    assert assess(grown)["overall_risk"] == "MEDIUM"


def test_chat_answer_verifier():
    tool_out = [{"overall_risk": "HIGH", "counts": {"defects_total": 4}, "defects": [{"type": "crack", "confidence": 0.9}]}]
    assert verify_answer_text("4 defects including a crack at 90% confidence; risk HIGH.", tool_out).ok
    assert not verify_answer_text("12 defects found, plus corrosion.", tool_out).ok


def test_compact_for_llm_preserves_counts_and_keeps_highest_ranked():
    defects = [
        {"id": f"D{i}", "type": "crack" if i % 2 else "dent",
         "severity_band": "HIGH" if i > 5 else "LOW",
         "severity_score": i, "confidence": i / 20}
        for i in range(1, 21)
    ]
    result = {
        "counts": {"defects_total": 20, "high": 15, "low": 5},
        "overall_risk": "HIGH",
        "defects": defects,
    }

    compact = compact_for_llm(result, max_defects=5)
    encoded = json.dumps(compact)
    decoded = json.loads(encoded)

    assert decoded["defects_shown"] == 5
    assert decoded["defects_omitted"] == 15
    assert [d["id"] for d in decoded["defects"]] == ["D20", "D19", "D18", "D17", "D16"]
    assert decoded["counts"] == result["counts"]
    assert decoded["overall_risk"] == "HIGH"
    assert decoded["by_type_and_band"] == {
        "crack": {"HIGH": 7, "LOW": 3},
        "dent": {"HIGH": 8, "LOW": 2},
    }
    tied = compact_for_llm({
        "defects": [
            {"id": "lower-confidence", "severity_score": 90, "confidence": 0.6},
            {"id": "higher-confidence", "severity_score": 90, "confidence": 0.9},
        ],
    }, max_defects=1)
    assert tied["defects"][0]["id"] == "higher-confidence"


def test_report_prompt_and_verifier_compact_only_shown_defects():
    detections = [
        dict(defect_type="crack", confidence=0.9, bbox=(i, i, 50, 20),
             relative_area=0.01, length_cm=1.0)
        for i in range(20)
    ]
    full_ctx = build_context(
        detections, (480, 640), failure_risk="HIGH", scan_id="large_scan",
    )
    messages = build_messages(full_ctx, "llm", [])
    prompt_record = json.loads(messages[1]["content"].split(
        "INSPECTION RECORD (JSON)\n", 1
    )[1].split("\n\nREFERENCE DECISION", 1)[0])

    assert prompt_record["defects_shown"] == 15
    assert prompt_record["defects_omitted"] == 5
    assert prompt_record["counts"] == full_ctx["counts"]
    assert json.loads(messages[1]["content"].split("REFERENCE DECISION\n", 1)[1])["overall_risk"] == "HIGH"

    report = build_template_report(prompt_record)
    verified = verify_report(report, prompt_record, decision_ctx=full_ctx)
    assert verified.ok
    assert len(report.findings) == 15


def test_required_knowledge_chunk_error_names_missing_id_and_folder(tmp_path):
    source = os.path.join(os.path.dirname(__file__), "..", "src", "copilot", "knowledge")
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    (tmp_path / "no-defects.md").unlink()

    with pytest.raises(MissingKnowledgeChunksError) as error:
        KnowledgeBase(directory=str(tmp_path), backend="tfidf")

    assert "no-defects" in str(error.value)
    assert str(tmp_path) in str(error.value)


def test_fixed_guidance_skips_a_missing_chunk():
    class MissingChunkKB:
        def get(self, chunk_id):
            return None

    assert _fixed(MissingChunkKB(), "no-defects") is None


def test_rules_comparison_with_one_scan_is_explicit(kb):
    store = ScanStore()
    store.add("only_scan", np.zeros((64, 64, 3), dtype=np.uint8))

    result = CopilotAgent(CopilotTools(store, kb=kb)).run("compare with the last scan")

    assert "Only one scan is loaded, so there is nothing to compare with yet" in result.answer
    assert "Scan only_scan:" in result.answer
    assert "Single-scan result" not in result.answer
    assert "Compared scans" not in result.answer


def test_llm_comparison_with_one_scan_still_reports_single_scan(kb):
    store = ScanStore()
    store.add("only_scan", np.zeros((64, 64, 3), dtype=np.uint8))

    result = CopilotAgent(
        CopilotTools(store, kb=kb),
        ScriptedLLM(["Single scan summary received."]),
    ).run("compare with the last scan")

    assert "Only one scan is loaded, so there is nothing to compare with yet" in result.answer
    assert "Single-scan result for only_scan:" in result.answer
    assert result.verification.ok


def test_rules_comparison_names_scan_ids_in_order(kb):
    store = ScanStore()
    store.load_samples()

    result = CopilotAgent(CopilotTools(store, kb=kb)).run("compare with the last scan")

    assert "Compared scans (earlier -> later): panel_defects_scan1 -> panel_defects_scan2." in result.answer


def test_rules_what_should_i_do_targets_worst_defect_and_cites_guidance(kb):
    store = ScanStore()
    store.add("inspection", np.zeros((480, 640, 3), dtype=np.uint8))
    store.scans["inspection"].detections = [
        dict(defect_type="dent", confidence=0.6, bbox=(20, 20, 15, 15), relative_area=0.001),
        dict(defect_type="crack", confidence=0.95, bbox=(100, 100, 100, 20), relative_area=0.02),
    ]

    result = CopilotAgent(CopilotTools(store, kb=kb)).run(
        "what should I do about the most severe defect?"
    )

    assert "D2: crack (HIGH" in result.answer
    assert "[crack-high]" in result.answer
    assert result.verification.ok


def test_rules_what_should_i_do_with_no_defects_cites_clean_scan_guidance(kb):
    store = ScanStore()
    store.add("clean_scan", np.zeros((64, 64, 3), dtype=np.uint8))

    result = CopilotAgent(CopilotTools(store, kb=kb)).run("what should I do about this scan?")

    assert "No defects were detected in scan clean_scan." in result.answer
    assert "[no-defects]" in result.answer
    assert result.verification.ok


def test_llm_tool_message_compacts_many_defects_but_keeps_full_totals(kb):
    store = ScanStore()
    store.add("large_scan", np.zeros((480, 640, 3), dtype=np.uint8))
    store.scans["large_scan"].detections = [
        dict(defect_type="crack", confidence=0.9, bbox=(i % 600, i % 450, 50, 20),
             relative_area=0.01, length_cm=1.0)
        for i in range(1000)
    ]
    tools = CopilotTools(store, kb=kb)
    observed = {}

    def script(messages, _schemas):
        tool_messages = [m for m in messages if m["role"] == "tool"]
        if not tool_messages:
            return {"tool_calls": [{
                "id": "assess", "name": "assess_flight_readiness",
                "arguments": {"scan_id": "large_scan"},
            }]}
        payload = json.loads(tool_messages[-1]["content"])
        observed.update({
            "shown": len(payload["defects"]),
            "omitted": payload["defects_omitted"],
            "total": payload["counts"]["defects_total"],
            "risk": payload["overall_risk"],
        })
        return {"content": f"{payload['counts']['defects_total']} defects detected; risk {payload['overall_risk']}."}

    result = CopilotAgent(tools, ScriptedLLM(script)).run("is it safe to fly?")

    assert observed == {"shown": 15, "omitted": 985, "total": 1000, "risk": "HIGH"}
    assert "1000 defects" in result.answer
    assert result.verification.ok
    assert len(result.steps[0]["result"]["defects"]) == 1000


class FakeVision:
    name = "fake-gemini"

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate_with_image(self, prompt, jpeg_bytes, json_mode=True):
        self.calls += 1
        assert json_mode
        assert len(jpeg_bytes) <= 3 * 1024 * 1024
        if callable(self.response):
            return self.response(prompt, jpeg_bytes)
        return self.response


def vision_report(scene_type="closeup_panel", defects=None):
    return json.dumps({
        "scene_type": scene_type,
        "summary": "Visible surface details are limited in this image.",
        "defects": defects or [],
        "overall_concern": "low",
        "needs_human_review": True,
    })


def test_analyze_scan_accepts_scene_type_and_flags_wide_image():
    from src.copilot.vision import SCAN_PROMPT

    gemini = FakeVision(vision_report("wide_aircraft"))
    result = analyze_scan(np.zeros((40, 80, 3), dtype=np.uint8), gemini)

    assert result["scene_type"] == "wide_aircraft"
    assert "detector" not in SCAN_PROMPT.lower()
    assert gemini.calls == 1


def test_analyze_scan_returns_readable_error_for_invalid_schema():
    gemini = FakeVision('{"scene_type":"not-a-scene","defects":[]}')

    result = analyze_scan(np.zeros((40, 80, 3), dtype=np.uint8), gemini)

    assert "error" in result
    assert "vision scan failed:" in result["error"]
    assert "scene_type" in result["error"]


def test_analyze_scan_returns_readable_error_for_malformed_json():
    gemini = FakeVision("{not-json")

    result = analyze_scan(np.zeros((40, 80, 3), dtype=np.uint8), gemini)

    assert "error" in result
    assert "vision scan failed:" in result["error"]
    assert "Expecting property name enclosed in double quotes" in result["error"]


def test_analyze_image_tool_caches_successful_review(kb):
    store = ScanStore()
    store.add("wide_scan", np.zeros((40, 80, 3), dtype=np.uint8))
    gemini = FakeVision(vision_report("wide_aircraft"))
    tools = CopilotTools(store, kb=kb, vision_llm=gemini)

    first = tools.analyze_image_with_vision("wide_scan")
    second = tools.analyze_image_with_vision("wide_scan")

    assert first["detector_comparison"]["agreement"] == "agree"
    assert "detector is tuned for close-up panels" in first["note"]
    assert second == first
    assert gemini.calls == 1


def test_unreliable_detection_warning_keeps_full_decision(kb):
    store = ScanStore()
    store.add("many_regions", np.zeros((1000, 1000, 3), dtype=np.uint8))
    store.scans["many_regions"].detections = [
        dict(defect_type="crack", confidence=0.9, bbox=(i % 900, i % 900, 50, 20),
             relative_area=0.01)
        for i in range(1000)
    ]
    tools = CopilotTools(store, kb=kb, vision_llm=FakeVision(vision_report()))
    record = tools.assess_flight_readiness("many_regions")
    result = CopilotAgent(tools).run("is it safe to fly?")
    full_ctx = build_context(
        store.detections(store.resolve("many_regions")),
        store.resolve("many_regions").image.shape,
    )

    assert record["detection_quality"]["likely_unreliable"]
    assert record["counts"]["defects_total"] == 1000
    assert record["overall_risk"] == assess(full_ctx)["overall_risk"] == "HIGH"
    assert record["warning"] in result.answer
    assert "click “Analyze with vision AI”" in result.answer
    assert len(result.steps[1]["result"]["defects"]) == 1000


def test_vision_facts_require_a_vision_review_and_are_attributed():
    vision_output = {
        "vision_review": {
            "scene_type": "closeup_panel",
            "summary": "A visible crack is in the upper-left region.",
            "defects": [{"type": "crack", "region": "upper-left region", "confidence": 0.8, "note": "line"}],
            "overall_concern": "medium",
            "needs_human_review": True,
        },
    }

    assert not verify_answer_text(
        "The vision review noted a crack in the upper-left region.", [vision_output]
    ).errors
    unsupported = verify_answer_text(
        "The vision review noted corrosion in the upper-left region.", [vision_output]
    )
    assert any(issue.code == "unsupported_vision_defect" for issue in unsupported.issues)

    missing_review = verify_answer_text(
        "The vision review noted a crack in the upper-left region.", []
    )
    assert any(issue.code == "unsupported_vision_fact" for issue in missing_review.issues)


def test_llm_image_request_includes_vision_tool_and_avoids_certification(kb):
    store = ScanStore()
    store.add("panel", np.zeros((64, 64, 3), dtype=np.uint8))
    tools = CopilotTools(store, kb=kb, vision_llm=FakeVision(vision_report()))

    def script(messages, _schemas):
        if not any(message["role"] == "tool" for message in messages):
            return {"tool_calls": [{
                "id": "assess",
                "name": "assess_flight_readiness",
                "arguments": {"scan_id": "latest"},
            }]}
        return {
            "content": (
                "The vision review reports no visible defects. The detector-based result remains "
                "advisory; a qualified person must decide, and this does not certify the aircraft safe."
            )
        }

    result = CopilotAgent(tools, ScriptedLLM(script)).run(
        "Look at this image and tell me if it is safe to fly."
    )

    assert any(step["tool"] == "analyze_image_with_vision" for step in result.steps)
    assert "vision review" in result.answer.lower()
    assert "does not certify" in result.answer
    assert result.verification.ok


def test_vision_review_cannot_change_assessment():
    detections = [
        dict(defect_type="crack", confidence=0.8, bbox=(20, 20, 60, 10), relative_area=0.02),
    ]
    baseline = build_context(detections, (100, 100))
    reviewed = build_context(
        detections, (100, 100),
        vision_review={
            "scene_type": "wide_aircraft",
            "summary": "No obvious damage visible.",
            "defects": [],
            "overall_concern": "none",
            "needs_human_review": False,
        },
    )

    assert assess(reviewed) == assess(baseline)


def test_agent_tool_loop_and_rules_fallback(kb):
    store = ScanStore(); store.load_samples()
    tools = CopilotTools(store, kb=kb)

    def script(messages, _tools):
        if not any(m["role"] == "tool" for m in messages):
            return {"tool_calls": [{"id": "1", "name": "assess_flight_readiness",
                                    "arguments": {"scan_id": "latest", "previous_scan_id": "previous"}}]}
        data = json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])
        return {"content": f"Advisory risk is {data['overall_risk']}."}

    r = CopilotAgent(tools, ScriptedLLM(script)).run("is it safe to fly?")
    assert r.steps[0]["tool"] == "assess_flight_readiness" and r.warning is None

    r2 = CopilotAgent(tools, None).run("Compare this scan with the last one and tell me if it's safe to fly")
    assert r2.mode == "rules" and r2.verification.ok
    assert "Advisory risk" in r2.answer


def test_tool_errors_do_not_crash(kb):
    tools = CopilotTools(ScanStore(), kb=kb)
    assert "error" in tools.call("detect_defects", {"scan_id": "nope"})
    assert "error" in tools.call("no_such_tool", {})


def test_agent_falls_back_when_llm_fails(kb):
    class Boom(ScriptedLLM):
        def chat(self, *a, **k):
            raise RuntimeError("Connection error.")
    store = ScanStore(); store.load_samples()
    r = CopilotAgent(CopilotTools(store, kb=kb), Boom([])).run("is it safe to fly?")
    assert r.mode == "rules" and "LLM unavailable" in r.warning


def test_agent_does_not_fall_back_when_llm_is_required(kb, monkeypatch):
    store = ScanStore(); store.load_samples()
    monkeypatch.setenv("AEROVISION_REQUIRE_LLM", "1")
    tools = CopilotTools(store, kb=kb)
    agent = CopilotAgent(
        tools,
        None,
        llm_reason="GROQ_API_KEY is empty or missing.",
    )
    r = agent.run("is it safe to fly?")
    assert r.mode == "rules"
    assert not r.steps
    assert "LLM is required, but unavailable" in r.answer
    assert "GROQ_API_KEY is empty or missing." in r.answer

    class Boom(ScriptedLLM):
        def chat(self, *args, **kwargs):
            raise RuntimeError("Invalid API key.")

    failed = CopilotAgent(tools, Boom([])).run("is it safe to fly?")
    assert failed.mode == "rules" and not failed.steps
    assert "LLM is required, but the request failed" in failed.answer
    assert "Invalid API key." in failed.answer


def test_vision_second_opinion_with_fake_gemini(kb):
    class FakeGemini:
        name = "fake-gemini"
        def generate_with_image(self, prompt, jpeg_bytes, json_mode=True):
            assert jpeg_bytes[:2] == b"\xff\xd8"  # a real JPEG crop was sent
            return json.dumps({"defect_present": True, "defect_type": "corrosion", "confidence": 0.7, "rationale": "Brown staining."})
    store = ScanStore(); store.load_samples()
    tools = CopilotTools(store, kb=kb, vision_llm=FakeGemini())
    d = tools.detect_defects("panel_defects_scan2")["defects"][0]["id"]
    res = tools.vision_second_opinion("panel_defects_scan2", d)
    assert res["agreement"] in ("agree", "type_disagree") and "needs_human_review" in res
    assert "error" in CopilotTools(store, kb=kb).vision_second_opinion("latest", "D1")
