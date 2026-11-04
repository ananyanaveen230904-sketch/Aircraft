"""Helpers for keeping structured LLM inputs bounded without corrupting JSON."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict


def compact_for_llm(obj: Any, max_defects: int = 15) -> Any:
    """Keep the highest-severity defects while preserving totals for all defects."""
    if max_defects < 0:
        raise ValueError("max_defects must be non-negative")
    if not isinstance(obj, dict) or not isinstance(obj.get("defects"), list):
        return obj

    compact = dict(obj)
    defects = obj["defects"]
    if len(defects) <= max_defects:
        return compact

    def score(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    def rank(row):
        if not isinstance(row, dict):
            return (0.0, 0.0)
        return (
            score(row.get("severity_score")),
            score(row.get("confidence")),
        )

    ranked = sorted(enumerate(defects), key=lambda item: (-rank(item[1])[0], -rank(item[1])[1], item[0]))
    shown_rows = [row for _, row in ranked[:max_defects]]
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in defects:
        if not isinstance(row, dict):
            counts["unknown"]["UNKNOWN"] += 1
            continue
        counts[str(row.get("type") or "unknown")][str(row.get("severity_band") or "UNKNOWN")] += 1

    compact["defects"] = shown_rows
    compact["defects_shown"] = len(shown_rows)
    compact["defects_omitted"] = len(defects) - len(shown_rows)
    compact["by_type_and_band"] = {
        defect_type: dict(sorted(bands.items()))
        for defect_type, bands in sorted(counts.items())
    }
    return compact
