"""
Generate a synthetic (input_text, target_text) training dataset for
fine-tuning T5-small on AeroVision inspection report generation.

This step ONLY generates data — no model training here.

Pipeline per record:
  1. Programmatically build a structured inspection record (rul_cycles,
     health_score, failure_risk, defects) with deliberate coverage of
     edge cases, not just uniform random combinations.
  2. Convert it to prompt text via the EXISTING src.fusion.to_prompt_text()
     — reused as-is, not reimplemented, so the training input format
     always matches what the real dashboard will actually produce.
  3. Call Gemini once to generate a realistic professional inspection
     summary paragraph grounded strictly in that structured data.
  4. Append one JSONL line: {"record_index", "input_text", "target_text"}.

Resumable: re-running the same command skips any record_index already
present in the output file, so an interrupted run (Ctrl+C, rate limit,
crash) can just be re-run to pick up where it left off.

Usage:
    # Either set the key directly:
    export GEMINI_API_KEY=...
    # ...or put it in a .env file in the project root (GEMINI_API_KEY=...),
    # which this script loads automatically via python-dotenv if installed.
    python src/report_gen/generate_dataset.py --target 500

    # Test the pipeline without spending any API calls:
    python src/report_gen/generate_dataset.py --target 20 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Load a .env file from the project root, if present, so GEMINI_API_KEY
# (or GOOGLE_API_KEY) can live there instead of needing to be set in
# every shell session. Silently does nothing if python-dotenv isn't
# installed or no .env file exists — falls back to whatever's already
# in the environment either way.
try:
    from dotenv import load_dotenv
    _PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass

from src.fusion import to_prompt_text

DEFECT_TYPES = ["crack", "corrosion", "dent", "rivet_damage"]
LOCATIONS = [
    "compressor casing", "engine surface", "wing panel",
    "fuselage section", "upper-left panel", "center panel",
]

# Same thresholds as src/rul/predict.py's failure_risk_band(), but
# returned already normalized to Title Case ("High"/"Medium"/"Low") —
# matching what src.fusion.fuse_inspection_results() actually produces
# in real usage (it maps HIGH/MEDIUM/LOW -> High/Medium/Low before
# to_prompt_text() ever sees it). Training data must match that exact
# real input format, or the fine-tuned model would learn on a
# distribution that never actually occurs at inference time.
def _failure_risk_from_rul(rul_cycles: float) -> str:
    if rul_cycles < 20:
        return "High"
    elif rul_cycles <= 50:
        return "Medium"
    else:
        return "Low"


SYSTEM_INSTRUCTION_TEMPLATES = [
    "You are an aircraft maintenance engineer writing a brief inspection summary. "
    "Based ONLY on the data below, write a professional, factual 3-5 sentence summary. "
    "Do not invent any details not present in the data.",

    "Write a concise aircraft maintenance inspection report paragraph (3-5 sentences) "
    "for the following inspection data. Use a formal, technical tone. "
    "Only reference facts explicitly given below — do not add any information not present.",

    "As an aviation maintenance technician, summarize this inspection data into a "
    "clear 3-5 sentence report for the maintenance log. Be direct and factual. "
    "Stick strictly to the data provided; do not fabricate additional findings.",

    "Draft a short (3-5 sentence) maintenance inspection summary based strictly on the "
    "structured data below, written the way a certified inspector would document it. "
    "Do not include any claims beyond what the data states.",
]


def _random_defect(rng: random.Random) -> Dict[str, Any]:
    return {
        "type": rng.choice(DEFECT_TYPES),
        "confidence": round(rng.uniform(0.5, 0.99), 2),
        "location": rng.choice(LOCATIONS),
    }


def _random_rul_health(rng: random.Random, rul_range=(2, 130)) -> Dict[str, Any]:
    rul_cycles = rng.randint(*rul_range)
    # health_score roughly correlated with rul_cycles, capped 125 like
    # health_score() does, plus noise, clipped to [0, 100]
    base = min(100.0, (rul_cycles / 125.0) * 100.0)
    noise = rng.uniform(-12, 12)
    health_score = round(max(0.0, min(100.0, base + noise)))
    failure_risk = _failure_risk_from_rul(rul_cycles)
    return {"rul_cycles": rul_cycles, "health_score": health_score, "failure_risk": failure_risk}


def generate_records(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate n structured records with deliberate scenario coverage
    (not pure uniform random), so the fine-tuned model sees the full
    range of realistic report styles:
      - full record: RUL/health + 0-4 defects, independently varied
      - high health + defects present (skin damage independent of engine wear)
      - low health + zero defects (engine degrading, nothing visible)
      - single-defect-type-only records
      - RUL-only partial records (no defect scan data)
      - defects-only partial records (no sensor data)
    """
    rng = random.Random(seed)
    records = []

    # Scenario weights — deliberately not uniform, to guarantee edge-case coverage
    # rather than leaving it to chance.
    scenarios = (
        ["full_random"] * 30
        + ["high_health_with_defects"] * 15
        + ["low_health_no_defects"] * 10
        + ["single_defect_type"] * 15
        + ["rul_only_partial"] * 15
        + ["defects_only_partial"] * 15
    )

    for i in range(n):
        scenario = rng.choice(scenarios)
        record: Dict[str, Any] = {}

        if scenario == "full_random":
            record.update(_random_rul_health(rng))
            num_defects = rng.choice([0, 0, 1, 1, 2, 2, 3, 4])  # weighted toward fewer defects
            record["defects"] = [_random_defect(rng) for _ in range(num_defects)]

        elif scenario == "high_health_with_defects":
            record.update(_random_rul_health(rng, rul_range=(70, 130)))
            num_defects = rng.randint(1, 4)
            record["defects"] = [_random_defect(rng) for _ in range(num_defects)]

        elif scenario == "low_health_no_defects":
            record.update(_random_rul_health(rng, rul_range=(2, 18)))
            record["defects"] = []

        elif scenario == "single_defect_type":
            record.update(_random_rul_health(rng))
            defect_type = rng.choice(DEFECT_TYPES)
            num_defects = rng.randint(1, 3)
            record["defects"] = [
                {"type": defect_type, "confidence": round(rng.uniform(0.5, 0.99), 2),
                 "location": rng.choice(LOCATIONS)}
                for _ in range(num_defects)
            ]

        elif scenario == "rul_only_partial":
            # No defect-scan data at all this inspection — RUL/health only.
            record.update(_random_rul_health(rng))
            record["defects"] = []

        elif scenario == "defects_only_partial":
            # No sensor data this inspection — defects only, matching
            # fuse_inspection_results()'s partial-record support.
            num_defects = rng.randint(0, 4)
            record["defects"] = [_random_defect(rng) for _ in range(num_defects)]

        record["record_index"] = i
        records.append(record)

    return records


def _fused_dict_to_prompt_input(record: Dict[str, Any]) -> str:
    """
    Build the exact same shape fuse_inspection_results() would return
    (minus record_index), then reuse to_prompt_text() as-is — so
    training inputs always match the real fusion output format,
    including the "rul_only_partial"/"defects_only_partial" cases
    (keys simply absent, same as a real partial record).
    """
    fused = {k: v for k, v in record.items() if k != "record_index"}
    return to_prompt_text(fused)


def _load_done_indices(output_path: str) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row["record_index"])
            except (json.JSONDecodeError, KeyError):
                continue  # skip a corrupt/partial last line from an interrupted run
    return done


def _call_llm_with_retry(call_fn, prompt: str, max_retries: int = 4, base_delay: float = 2.0) -> str:
    """
    call_fn: a zero-extra-arg callable(prompt) -> str, provider-specific
    (see _make_gemini_caller / _make_groq_caller below). Retries with
    exponential backoff on any failure (rate limits, timeouts, etc.).
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            text = call_fn(prompt).strip()
            if not text:
                raise ValueError("Empty response from LLM")
            return text
        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                break
            delay = base_delay * (2 ** attempt)
            print(f"    LLM call failed ({e}), retrying in {delay:.0f}s...")
            time.sleep(delay)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def _make_gemini_caller(args):
    try:
        import google.generativeai as genai
    except ImportError:
        raise SystemExit(
            "google-generativeai is not installed. Run: pip install google-generativeai\n"
            "Or use --dry-run to test the pipeline without calling an LLM."
        )
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "No Gemini API key found. Set GEMINI_API_KEY in your environment or .env file.\n"
            "Or use --dry-run to test the pipeline without calling an LLM."
        )
    genai.configure(api_key=api_key)

    if args.list_models:
        print("Gemini models available to your API key that support generateContent:\n")
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(f"  {m.name}")
        sys.exit(0)

    model_name = args.model or "gemini-2.5-flash"
    gemini_model = genai.GenerativeModel(model_name)

    # Fail fast with a helpful message if the model/key doesn't work,
    # rather than burning retries x every record before finding out.
    try:
        gemini_model.generate_content("test")
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            raise SystemExit(
                f"Model '{model_name}' is not available for your API key/API version ({e}).\n"
                f"Run with --list-models to see what's currently valid."
            )
        raise SystemExit(f"Gemini test call failed ({e}). Check your API key/project access.")

    def call(prompt: str) -> str:
        response = gemini_model.generate_content(prompt)
        return response.text or ""

    return call


def _make_groq_caller(args):
    try:
        from groq import Groq
    except ImportError:
        raise SystemExit(
            "groq is not installed. Run: pip install groq\n"
            "Or use --dry-run to test the pipeline without calling an LLM."
        )
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "No Groq API key found. Either:\n"
            "  1. Set it directly: $env:GROQ_API_KEY=\"your-key\"  (PowerShell)\n"
            "  2. Put GROQ_API_KEY=your-key in a .env file in the project root\n"
            "Get a free key at: https://console.groq.com/keys\n"
            "Or use --dry-run to test the pipeline without calling an LLM."
        )
    client = Groq(api_key=api_key)

    if args.list_models:
        print("Models available from Groq:\n")
        for m in client.models.list().data:
            print(f"  {m.id}")
        sys.exit(0)

    model_name = args.model or "llama-3.3-70b-versatile"

    # Fail fast if the model/key doesn't work.
    try:
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
    except Exception as e:
        raise SystemExit(
            f"Groq test call failed for model '{model_name}' ({e}).\n"
            f"Run with --list-models to see what's currently valid for your key, "
            f"or check https://console.groq.com/keys for key status."
        )

    def call(prompt: str) -> str:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        return completion.choices[0].message.content or ""

    return call


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic report-generation training pairs")
    parser.add_argument("--target", type=int, default=500, help="Number of records to generate")
    parser.add_argument("--output", default=os.path.join("data", "report_training_pairs.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                         help="Skip LLM calls; write a placeholder target_text instead. "
                              "Use this to test the pipeline/resume logic without spending API calls.")
    parser.add_argument("--provider", default="groq", choices=["groq", "gemini"],
                         help="Which LLM API to use for generating summaries (default: groq).")
    parser.add_argument("--model", default=None,
                         help="Model name to use. Defaults: 'llama-3.3-70b-versatile' for groq, "
                              "'gemini-2.5-flash' for gemini. Model names change over time — "
                              "if you get a 'model not found' error, run with --list-models.")
    parser.add_argument("--list-models", action="store_true",
                         help="List models available to your API key for the chosen --provider, then exit.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    records = generate_records(args.target, seed=args.seed)
    done_indices = _load_done_indices(args.output)
    print(f"Target: {args.target} records. Already done: {len(done_indices)}. "
          f"Remaining: {args.target - len(done_indices)}.")

    call_llm = None
    if not args.dry_run:
        if args.provider == "groq":
            call_llm = _make_groq_caller(args)
        else:
            call_llm = _make_gemini_caller(args)

    processed_since_print = 0
    skipped = []

    with open(args.output, "a") as f:
        for record in records:
            idx = record["record_index"]
            if idx in done_indices:
                continue

            input_text = _fused_dict_to_prompt_input(record)

            if args.dry_run:
                target_text = f"[DRY RUN placeholder summary for record {idx}]"
            else:
                instruction = random.choice(SYSTEM_INSTRUCTION_TEMPLATES)
                prompt = f"{instruction}\n\nInspection data:\n{input_text}\n\nSummary:"
                try:
                    target_text = _call_llm_with_retry(call_llm, prompt)
                except RuntimeError as e:
                    print(f"  SKIPPING record {idx} after repeated failures: {e}")
                    skipped.append(idx)
                    continue

            row = {"record_index": idx, "input_text": input_text, "target_text": target_text}
            f.write(json.dumps(row) + "\n")
            f.flush()

            processed_since_print += 1
            if processed_since_print % 50 == 0:
                print(f"  ...{processed_since_print} new records written this run "
                      f"(record_index {idx}/{args.target - 1})")

    total_done = len(_load_done_indices(args.output))
    print(f"\nDone. {total_done}/{args.target} records now in {args.output}.")
    if skipped:
        print(f"Skipped {len(skipped)} record(s) after repeated failures: {skipped}")
        print("Re-run the same command to retry just these (already-done ones are skipped automatically).")


if __name__ == "__main__":
    main()
