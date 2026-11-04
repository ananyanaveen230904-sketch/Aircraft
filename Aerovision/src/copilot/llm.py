"""
Thin LLM client layer.

* GroqClient   - Llama 3.3 70B via Groq (OpenAI-compatible chat + tool calling)
* GeminiClient - Gemini text / vision (used for the second opinion + judge)
* ScriptedLLM  - deterministic stand-in used by unit tests / offline demos

Everything downstream depends only on the tiny `LLM` interface:

    chat(messages, tools=None, json_mode=False, temperature=0.0) -> dict
        returns {"content": str|None, "tool_calls": [{"id","name","arguments"}]}
"""
from __future__ import annotations

import importlib
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from . import config


class LLMUnavailable(RuntimeError):
    """Raised when no API key / SDK is available."""


def redact_secrets(error: Any) -> str:
    """Avoid exposing configured API keys if an SDK includes them in an error."""
    text = str(error)
    for secret in (config.GROQ_API_KEY, config.GEMINI_API_KEY):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


class LLM:
    name = "base"

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[dict]] = None,
             json_mode: bool = False, temperature: float = 0.0) -> Dict[str, Any]:
        raise NotImplementedError


def _retry(fn: Callable[[], Any], attempts: int = 3, base_delay: float = 2.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # rate limits, transient network errors
            last = e
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    raise RuntimeError(f"LLM call failed after {attempts} attempts: {last}")


class GroqClient(LLM):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or config.GROQ_API_KEY
        if not api_key:
            raise LLMUnavailable("GROQ_API_KEY is not set (see .env.example).")
        try:
            from groq import Groq
        except ImportError as e:  # pragma: no cover
            raise LLMUnavailable("`pip install groq` is required.") from e
        self._client = Groq(api_key=api_key)
        self.model = model or config.GROQ_MODEL
        self.name = f"groq:{self.model}"

    def chat(self, messages, tools=None, json_mode=False, temperature=0.0):
        kwargs: Dict[str, Any] = dict(model=self.model, messages=messages, temperature=temperature)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = _retry(lambda: self._client.chat.completions.create(**kwargs))
        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return {"content": msg.content, "tool_calls": calls}


class GeminiClient(LLM):
    """Text + optional image input. Tool calling is not used for Gemini here."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or config.GEMINI_API_KEY
        if not api_key:
            raise LLMUnavailable("GEMINI_API_KEY is not set (see .env.example).")
        try:
            import google.generativeai as genai
        except ImportError as e:  # pragma: no cover
            raise LLMUnavailable("`pip install google-generativeai` is required.") from e
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model or config.GEMINI_MODEL
        self._model = genai.GenerativeModel(self.model)
        self.name = f"gemini:{self.model}"

    def chat(self, messages, tools=None, json_mode=False, temperature=0.0):
        prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        cfg: Dict[str, Any] = {"temperature": temperature}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        resp = _retry(lambda: self._model.generate_content(prompt, generation_config=cfg))
        return {"content": resp.text, "tool_calls": []}

    def generate_with_image(self, prompt: str, jpeg_bytes: bytes, json_mode: bool = True) -> str:
        cfg: Dict[str, Any] = {"temperature": 0.0}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        parts = [prompt, {"mime_type": "image/jpeg", "data": jpeg_bytes}]
        # Keep Gemini image calls here so a future SDK migration changes only this wrapper.
        resp = _retry(lambda: self._model.generate_content(parts, generation_config=cfg))
        return resp.text


class ScriptedLLM(LLM):
    """Returns pre-scripted responses in order (or via a callable). Test/offline only."""

    name = "scripted"

    def __init__(self, responses):
        self._responses = responses
        self._i = 0
        self.calls: List[dict] = []

    def chat(self, messages, tools=None, json_mode=False, temperature=0.0):
        self.calls.append({"messages": messages, "tools": tools, "json_mode": json_mode})
        if callable(self._responses):
            out = self._responses(messages, tools)
        else:
            out = self._responses[min(self._i, len(self._responses) - 1)]
            self._i += 1
        if isinstance(out, str):
            out = {"content": out, "tool_calls": []}
        out.setdefault("tool_calls", [])
        out.setdefault("content", None)
        return out


def get_groq() -> Optional[GroqClient]:
    config.reload_env()
    try:
        return GroqClient()
    except LLMUnavailable:
        return None


def get_gemini() -> Optional[GeminiClient]:
    config.reload_env()
    try:
        return GeminiClient()
    except LLMUnavailable:
        return None


def llm_status() -> Dict[str, Dict[str, Any]]:
    """Describe whether each configured provider can construct a client."""
    config.reload_env()

    def status(key: str, aliases: str, sdk: str, module: str, model: str, factory):
        api_key = getattr(config, key)
        base = {"available": False, "reason": "", "model": model}
        if not api_key:
            missing = f"{aliases} is empty or missing."
            if not os.path.isfile(config.ENV_PATH):
                base["reason"] = f".env file not found at {config.ENV_PATH}; {missing}"
            else:
                base["reason"] = f"{missing} Check {config.ENV_PATH} or the process environment."
            return base

        try:
            importlib.import_module(module)
        except ImportError:
            base["reason"] = f"The {sdk} SDK is not installed."
            return base

        try:
            factory()
        except Exception as exc:
            base["reason"] = f"Client construction failed: {redact_secrets(exc)}"
            return base

        base["available"] = True
        base["reason"] = "Client initialized."
        return base

    return {
        "groq": status(
            "GROQ_API_KEY", "GROQ_API_KEY", "Groq", "groq",
            config.GROQ_MODEL, GroqClient,
        ),
        "gemini": status(
            "GEMINI_API_KEY", "GEMINI_API_KEY or GOOGLE_API_KEY",
            "Gemini", "google.generativeai", config.GEMINI_MODEL, GeminiClient,
        ),
    }


def test_llm_connection(client: LLM) -> None:
    """Send a minimal provider request; propagate its actual error to the UI."""
    if isinstance(client, GroqClient):
        client._client.chat.completions.create(
            model=client.model,
            messages=[{"role": "user", "content": "Reply OK."}],
            max_tokens=5,
        )
    elif isinstance(client, GeminiClient):
        client._model.generate_content(
            "Reply OK.",
            generation_config={"temperature": 0.0, "max_output_tokens": 5},
        )
    else:
        client.chat([{"role": "user", "content": "Reply OK."}])
