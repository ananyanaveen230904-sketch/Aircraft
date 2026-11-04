"""
Central configuration for the AeroVision Maintenance Copilot.

All secrets come from environment variables / a local `.env` file
(never hard-code keys). See `.env.example`.
"""
from __future__ import annotations

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(ROOT, ".env")


def reload_env() -> None:
    """Refresh configuration, preferring process variables over the .env file."""
    try:
        from dotenv import dotenv_values
        file_values = dotenv_values(ENV_PATH) if os.path.isfile(ENV_PATH) else {}
    except ImportError:  # pragma: no cover
        file_values = {}

    def value(name: str, default: str = "") -> str:
        current = os.environ[name] if name in os.environ else file_values.get(name)
        return current if current is not None else default

    global GROQ_API_KEY, GEMINI_API_KEY, GROQ_MODEL, GEMINI_MODEL
    global EMBED_MODEL, AEROVISION_REQUIRE_LLM
    GROQ_API_KEY = value("GROQ_API_KEY")
    GEMINI_API_KEY = value("GEMINI_API_KEY") or value("GOOGLE_API_KEY")
    GROQ_MODEL = value("AEROVISION_GROQ_MODEL") or "llama-3.3-70b-versatile"
    GEMINI_MODEL = value("AEROVISION_GEMINI_MODEL") or "gemini-2.5-flash"
    EMBED_MODEL = value("AEROVISION_EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
    AEROVISION_REQUIRE_LLM = value("AEROVISION_REQUIRE_LLM", "0") == "1"


reload_env()

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
EVAL_OUT_DIR = os.path.join(ROOT, "output", "eval")
SAMPLE_DIR = os.path.join(ROOT, "data", "sample_images")
CMAPSS_DIR = os.path.join(ROOT, "data", "cmapss")

MAX_REPAIR_ATTEMPTS = 1  # LLM re-tries (with verifier feedback) before template fallback

# These thresholds flag likely poor detector inputs without changing the decision.
DETECTION_QUALITY_MAX_RAW_COUNT = 40
DETECTION_QUALITY_MAX_PER_MEGAPIXEL = 25.0
UNRELIABLE_DETECTION_WARNING = (
    "The classical detector flagged an unusually large number of regions; this usually means "
    "the image is not a close-up panel photo or the detector needs tuning. Treat these counts "
    "as unreliable and get a human or vision review."
)
