"""The model that writes the local spoken answer.

In local mode the speech-to-text and the text-to-speech already run on this
machine; only the answer text has been a network call. Two backends fill that
slot:

  gemini  -- Google's API, the default, and the same model the cloud path uses.
  gemma   -- Google's open Gemma on a local llama.cpp `llama-server` (or
             anything OpenAI-compatible), which takes the last cloud hop out of
             the loop. Gemma 3 1B instruct, Q4_K_M GGUF, is the intended target.
             "qwen" is still accepted as an alias for this backend.

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


# The local backend was Qwen before it was Gemma; old settings and saved page
# choices still say "qwen", and mean the same llama.cpp server.
_ALIASES = {"qwen": "gemma"}


def normalize_llm(name: Optional[str]) -> str:
    name = (name or "").strip().lower()
    name = _ALIASES.get(name, name)
    return name if name in ("gemini", "gemma") else "gemini"


def default_llm() -> str:
    return normalize_llm(os.getenv("HARK_LOCAL_LLM", "") or "gemini")


def _local_base() -> str:
    return (os.getenv("HARK_LOCAL_LLM_BASE_URL", "")
            or "http://127.0.0.1:8081/v1").rstrip("/")


def _local_model() -> str:
    return os.getenv("HARK_LOCAL_LLM_MODEL", "") or "gemma-3-1b-it"


_vertex = None


def vertex_client():
    """Gemini through Vertex AI, when HARK_VERTEX_PROJECT is set; else None.

    An AI Studio free-tier key allows 20 requests a day per model and has no
    Google Search grounding to speak of -- a demo ran out mid-afternoon. Vertex
    AI bills the Google Cloud project instead and, on Cloud Run, authenticates
    as the service account: no key in the container at all.
    """
    global _vertex
    project = os.getenv("HARK_VERTEX_PROJECT", "").strip()
    if not project:
        return None
    if _vertex is None:
        from google import genai
        from google.genai import types
        _vertex = genai.Client(
            vertexai=True, project=project,
            location=os.getenv("HARK_VERTEX_LOCATION", "global").strip() or "global",
            http_options=types.HttpOptions(timeout=30_000))
    return _vertex


class GeminiAnswer:
    """Google's model, over the network. The default."""

    name = "gemini"
    _client = None

    def __init__(self, model: Optional[str] = None):
        from hark.core.brain import CHAT_MODEL
        self._model = model or os.getenv("HARK_LOCAL_CHAT_MODEL", "") or CHAT_MODEL

    def available(self) -> bool:
        return bool(os.getenv("HARK_VERTEX_PROJECT", "").strip() or os.getenv("GEMINI_API_KEY", ""))

    def _client_or_none(self):
        vertex = vertex_client()
        if vertex is not None:
            return vertex
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            return None
        if GeminiAnswer._client is None:
            from google import genai
            from google.genai import types
            # A stalled request must not hold a spoken reply open forever.
            GeminiAnswer._client = genai.Client(
                api_key=key, http_options=types.HttpOptions(timeout=30_000))
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


class GemmaAnswer:
    """A local llama.cpp `llama-server`, spoken to over its OpenAI-compatible API.

    `stream=True` gives server-sent events, so the reply can be spoken sentence
    by sentence exactly like Gemini's. Nothing here is llama.cpp-specific -- any
    `/v1/chat/completions` endpoint works.
    """

    name = "gemma"

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None):
        self._base = (base_url or _local_base()).rstrip("/")
        self._model = model or _local_model()
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
                print(f"[llm] gemma attempt {attempt + 1} failed: {exc}")
                if started or not _transient(exc):
                    return
                time.sleep(0.5)


def build_answer_model(name: Optional[str] = None):
    """The answer model to use, or None if the chosen one cannot run here."""
    chosen = normalize_llm(name) if name else default_llm()
    if chosen == "gemma":
        # Returned even if the probe failed: a server that is merely slow to
        # start should not silently downgrade the answer to the passage.
        return GemmaAnswer()
    model = GeminiAnswer()
    return model if model.available() else None


def available_llms() -> dict:
    """What the page can offer. Gemini needs a key; Gemma needs a live server."""
    return {"gemini": GeminiAnswer().available(), "gemma": GemmaAnswer().available()}


def _grounding_client():
    """Vertex AI when configured (see vertex_client), else the Gemini API key."""
    return GeminiAnswer()._client_or_none()


def google_search(query: str, context: str = "") -> Optional[dict]:
    """Answer a question from the live web, via Gemini grounded on Google Search.

    For what the book cannot answer: a fact it assumes, whether something is
    still true today, the latest on a topic. Returns {"text", "sources"} with
    sources as (title, url) pairs, or None when there is no key or the call
    fails -- the caller says so rather than inventing an answer.
    """
    try:
        client = _grounding_client()
    except Exception as exc:
        print(f"[search] no search client: {exc}")
        return None
    if client is None or not (query or "").strip():
        return None
    from hark.core.brain import CHAT_MODEL, THINKING_BUDGET
    from google.genai import types
    about = f" They are listening to: {context}." if context else ""
    try:
        resp = client.models.generate_content(
            model=os.getenv("HARK_LOCAL_CHAT_MODEL", "") or CHAT_MODEL,
            contents=f"Question:{about} {query}",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Search the web and answer the question in at most four plain "
                    "sentences, suitable to be read aloud. State facts, no lists, "
                    "no markdown. If the sources disagree or are unclear, say so."),
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            ),
        )
    except Exception as exc:
        print(f"[search] google search failed: {exc}")
        return None
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        return None
    sources = []
    try:
        meta = resp.candidates[0].grounding_metadata
        for chunk in (getattr(meta, "grounding_chunks", None) or []):
            web = getattr(chunk, "web", None)
            if web and getattr(web, "uri", None):
                pair = ((web.title or "").strip() or "source", web.uri)
                if pair not in sources:
                    sources.append(pair)
    except (AttributeError, IndexError, TypeError):
        pass
    return {"text": text, "sources": sources[:5]}
