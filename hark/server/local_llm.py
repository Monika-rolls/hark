"""The model that writes the local spoken answer.

In local mode the speech-to-text and the text-to-speech already run on this
machine; only the answer text has been a network call. Two backends fill that
slot:

  gemini  -- Google's API, the default, and the same model the cloud path uses.
  qwen    -- a local llama.cpp `llama-server` (or anything OpenAI-compatible),
             which takes the last cloud hop out of the loop. Qwen 0.5B/1.5B
             GGUF is the intended target.

Both are streaming: they yield text pieces as the model writes, and the caller
turns those into sentences so the first one can be spoken while the rest is
still being generated. Generation is retried on a transient failure, but only
before anything has been yielded -- a retry after that would repeat what was
already spoken.
"""
import json
import os
import time
import urllib.request
from typing import Iterator, Optional

# llama.cpp reachability, cached so /api/health can call it cheaply.
_probe = {"at": 0.0, "ok": False}
_PROBE_TTL = 10.0

_TRANSIENT = ("503", "429", "unavailable", "overloaded", "timed out",
              "connection refused", "temporarily")


def _transient(exc: Exception) -> bool:
    """Worth retrying? A quota wall is not -- it will not clear in a second."""
    message = str(exc).lower()
    if "quota" in message or "resource_exhausted" in message:
        return False
    return any(marker in message for marker in _TRANSIENT)


def default_llm() -> str:
    name = (os.getenv("HARK_LOCAL_LLM", "") or "gemini").strip().lower()
    return name if name in ("gemini", "qwen") else "gemini"


def _qwen_base() -> str:
    return (os.getenv("HARK_LOCAL_LLM_BASE_URL", "")
            or "http://127.0.0.1:8080/v1").rstrip("/")


def _qwen_model() -> str:
    return os.getenv("HARK_LOCAL_LLM_MODEL", "") or "qwen2.5-1.5b-instruct"


class GeminiAnswer:
    """Google's model, over the network. The default."""

    name = "gemini"
    _client = None

    def __init__(self, model: Optional[str] = None):
        from hark.core.brain import CHAT_MODEL
        self._model = model or os.getenv("HARK_LOCAL_CHAT_MODEL", "") or CHAT_MODEL

    def available(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY", ""))

    def _client_or_none(self):
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            return None
        if GeminiAnswer._client is None:
            from google import genai
            GeminiAnswer._client = genai.Client(api_key=key)
        return GeminiAnswer._client

    def stream(self, prompt: str, system: str) -> Iterator[str]:
        client = self._client_or_none()
        if client is None:
            return
        from hark.core.brain import MAX_REPLY_TOKENS, THINKING_BUDGET
        from google.genai import types
        for attempt in range(3):
            started = False
            try:
                stream = client.models.generate_content_stream(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=MAX_REPLY_TOKENS,
                        temperature=0.3,
                        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
                    ),
                )
                for chunk in stream:
                    if chunk.text:
                        started = True
                        yield chunk.text
                return
            except Exception as exc:
                print(f"[llm] gemini attempt {attempt + 1} failed: {exc}")
                if started or not _transient(exc):
                    return
                time.sleep(0.7 * (attempt + 1))


class QwenAnswer:
    """A local llama.cpp `llama-server`, spoken to over its OpenAI-compatible API.

    `stream=True` gives server-sent events, so the reply can be spoken sentence
    by sentence exactly like Gemini's. Nothing here is llama.cpp-specific -- any
    `/v1/chat/completions` endpoint works.
    """

    name = "qwen"

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None):
        self._base = (base_url or _qwen_base()).rstrip("/")
        self._model = model or _qwen_model()
        self._key = api_key if api_key is not None else os.getenv("HARK_LOCAL_LLM_API_KEY", "")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        return headers

    def available(self) -> bool:
        """Is the server answering? Cached briefly; used by /api/health."""
        now = time.time()
        if now - _probe["at"] < _PROBE_TTL:
            return _probe["ok"]
        ok = False
        try:
            req = urllib.request.Request(self._base + "/models", headers=self._headers())
            urllib.request.urlopen(req, timeout=0.8).read(1)
            ok = True
        except Exception:
            ok = False
        _probe.update(at=now, ok=ok)
        return ok

    def stream(self, prompt: str, system: str) -> Iterator[str]:
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "temperature": 0.3,
            # Small models ramble to whatever the cap allows; a spoken answer is
            # one or two sentences, so keep the leash short.
            "max_tokens": 220,
            "stream": True,
        }).encode("utf-8")
        for attempt in range(2):
            started = False
            try:
                req = urllib.request.Request(self._base + "/chat/completions",
                                             data=body, headers=self._headers())
                with urllib.request.urlopen(req, timeout=180) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            return
                        try:
                            event = json.loads(payload)
                        except ValueError:
                            continue
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        piece = (choices[0].get("delta") or {}).get("content")
                        if piece:
                            started = True
                            yield piece
                return
            except Exception as exc:
                print(f"[llm] qwen attempt {attempt + 1} failed: {exc}")
                if started or not _transient(exc):
                    return
                time.sleep(0.5)


def build_answer_model(name: Optional[str] = None):
    """The answer model to use, or None if the chosen one cannot run here."""
    chosen = (name or default_llm()).strip().lower()
    if chosen == "qwen":
        # Returned even if the probe failed: a server that is merely slow to
        # start should not silently downgrade the answer to the passage.
        return QwenAnswer()
    model = GeminiAnswer()
    return model if model.available() else None


def available_llms() -> dict:
    """What the page can offer. Gemini needs a key; Qwen needs a live server."""
    return {"gemini": GeminiAnswer().available(), "qwen": QwenAnswer().available()}
