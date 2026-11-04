"""
Compare report-generation configurations on the synthetic eval set.

  template        deterministic baseline (no LLM, no RAG)
  llm_raw         LLM, no RAG, first attempt, no guardrail
  llm_guard       LLM, no RAG, + verifier/repair/fallback
  llm_rag_raw     LLM + RAG, first attempt, no guardrail
  llm_rag_guard   LLM + RAG + verifier/repair/fallback   (full system)

Usage (from project root):
    python -m src.copilot.evaluation.run_eval --n 30                  # real Groq run
    python -m src.copilot.evaluation.run_eval --n 30 --judge gemini   # + independent LLM judge
    python -m src.copilot.evaluation.run_eval --n 10 --simulate       # plumbing test, NOT real results

Resumable: finished records are skipped when you re-run the same command.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.copilot import config
from src.copilot.llm import get_gemini, get_groq
from src.copilot.rag import KnowledgeBase
from src.copilot.report_generator import generate_report
from src.copilot.evaluation.eval_set import build_eval_set
from src.copilot.evaluation.judge import judge_report
from src.copilot.evaluation.metrics import METRIC_KEYS, bootstrap_ci, score

CONFIGS = ["template", "llm_raw", "llm_guard", "llm_rag_raw", "llm_rag_guard"]


def _load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["record_index"], r["config"]))
            except Exception:
                pass
    return done


def summarize(rows, out_dir, simulated):
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)
    keys = METRIC_KEYS + ["fallback_rate", "judge_faithfulness", "judge_completeness", "judge_actionability", "latency_s"]
    table = []
    for cfg in CONFIGS:
        rs = by_cfg.get(cfg, [])
        if not rs:
            continue
        row = {"config": cfg, "n": len(rs)}
        for k in METRIC_KEYS:
            m, lo, hi = bootstrap_ci([r["metrics"].get(k) for r in rs])
            row[k] = f"{m:.3f} [{lo:.3f},{hi:.3f}]" if m == m else "n/a"
        fb = [float(r.get("fallback", False)) for r in rs]
        row["fallback_rate"] = f"{sum(fb) / len(fb):.3f}"
        for j in ("faithfulness", "completeness", "actionability"):
            vals = [r["judge"][j] for r in rs if r.get("judge")]
            row[f"judge_{j}"] = f"{sum(vals) / len(vals):.2f} (n={len(vals)})" if vals else "n/a"
        lat = [r.get("latency_s", 0) for r in rs]
        row["latency_s"] = f"{sum(lat) / len(lat):.2f}"
        table.append(row)
    if not table:
        return
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "n"] + keys)
        w.writeheader(); w.writerows(table)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        if simulated:
            f.write("> **SIMULATED RUN: plumbing test only. These numbers are NOT model results.**\n\n")
        f.write("| config | n | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 2) + "\n")
        for r in table:
            f.write(f"| {r['config']} | {r['n']} | " + " | ".join(str(r[k]) for k in keys) + " |\n")
        f.write("\nMean with 95% bootstrap CI. hallucinated / unsafe_decision: lower is better; others: higher is better.\n")
    print(open(os.path.join(out_dir, "summary.md"), encoding="utf-8").read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--judge", choices=["none", "gemini", "groq"], default="none")
    ap.add_argument("--judge-n", type=int, default=15, help="judge only the first N records per config (cost control)")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between LLM calls (rate limits)")
    ap.add_argument("--simulate", action="store_true", help="offline plumbing test with a fault-injecting fake LLM")
    args = ap.parse_args()

    items = build_eval_set(args.n, args.seed)
    kb = KnowledgeBase()
    print(f"Retriever backend: {kb.backend} | records: {len(items)}")

    simulated = args.simulate
    if simulated:
        from src.copilot.evaluation.simulated import SimulatedLLM
        llm = SimulatedLLM()
        print("=" * 70 + "\nSIMULATED RUN: numbers below are NOT real model results.\n" + "=" * 70)
    else:
        llm = get_groq()
        if llm is None:
            raise SystemExit("GROQ_API_KEY not set. Put it in .env (see .env.example) or use --simulate.")
    judge_llm = None
    if args.judge == "gemini" and not simulated:
        judge_llm = get_gemini() or SystemExit("GEMINI_API_KEY not set for --judge gemini")
    elif args.judge == "groq" and not simulated:
        judge_llm = llm
    if isinstance(judge_llm, SystemExit):
        raise judge_llm

    out_dir = os.path.join(config.EVAL_OUT_DIR, "simulated" if simulated else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.jsonl")
    done = _load_done(path)

    with open(path, "a", encoding="utf-8") as fout:
        for n, item in enumerate(items):
            idx, ctx = item["record_index"], item["context"]
            if all((idx, c) in done for c in CONFIGS):
                continue
            outputs = {}
            outputs["template"] = (generate_report(ctx, "template"), 0)
            for base, mode in (("llm", "llm"), ("llm_rag", "llm_rag")):
                if simulated:
                    from src.copilot.report_generator import retrieve_for_context
                    llm.set_context(ctx, retrieve_for_context(ctx, kb) if mode == "llm_rag" else [])
                g = generate_report(ctx, mode, llm, kb, guardrails=True)
                outputs[f"{base}_guard"] = (g, 0)
                outputs[f"{base}_raw"] = (g, 1)
                if not simulated:
                    time.sleep(args.sleep)
            for cfg in CONFIGS:
                g, raw = outputs[cfg]
                if raw:
                    att = g.first_attempt
                    rep, ver = att.report, att.verification
                    fallback = False
                else:
                    rep, ver, fallback = g.report, g.verification, g.fallback_used
                m = score(rep, ver, ctx, g.retrieved)
                row = {"record_index": idx, "tag": item["tag"], "config": cfg, "metrics": m,
                       "fallback": fallback, "latency_s": round(g.latency_s, 2),
                       "attempts": len(g.attempts), "report": rep.model_dump() if rep else None,
                       "issues": [i.to_dict() for i in (ver.issues if ver else [])]}
                if judge_llm and rep and n < args.judge_n:
                    row["judge"] = judge_report(judge_llm, ctx, rep.model_dump(), g.retrieved)
                    time.sleep(args.sleep)
                fout.write(json.dumps(row) + "\n"); fout.flush()
            print(f"  record {idx} done ({n + 1}/{len(items)})")

    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    summarize(rows, out_dir, simulated)


if __name__ == "__main__":
    main()
