"""Hark backend: mints session tokens and serves the one tool that matters.

The shape here follows AssemblyAI's stored-agent + HTTP-tools pattern, so this
backend holds **no** websocket to AssemblyAI, no tool dispatcher, and no session
loop. AssemblyAI calls `/tools/passage_at_playhead` from its own servers; the
browser holds the only websocket, using a short-lived token minted here so the
API key never leaves the server.

What makes this project more than a wrapper is the tool itself: the listener's
playback position selects the passage. "What did that last part mean?" has no
content to search for, so similarity search over the book returns noise.
Position answers it exactly.
"""

# Support `python hark/server/main.py` as well as `uvicorn hark.server.main:app`
# and `python -m hark.server.main`. Run as a script, sys.path[0] is hark/server/,
# so the repo root is not importable (ModuleNotFoundError: hark) and there is no
# package for the relative imports below to resolve against. Put the root on the
# path and name the package, so both forms behave the same.
if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "hark.server"

import asyncio
import base64
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import urllib.error
import urllib.request
import hmac
import json

import numpy as np

from dotenv import load_dotenv
from fastapi import (FastAPI, HTTPException, Request, WebSocket,
                     WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hark.core.library import Library

from . import books, local_llm
from .local_voice import (CONTENTLESS, HALLUCINATIONS, MIN_UTTERANCE_S, STT_RATE,
                          WIRE_RATE, DEFAULT_TTS_VOICE, TTS_VOICES, Endpointer,
                          LocalVoice, new_speech_detector, pcm16_to_float,
                          resample, tts_voice, vad_enabled)
from .store import build_store

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"


def _setting(name: str, default: str = "") -> str:
    """Read HARK_<name>."""
    return os.getenv(f"HARK_{name}") or default


# The built-in titles, in shelf order; the first is the default. HARK_BOOK is
# the older single-title form and still works.
BOOKS = [b.strip() for b in (_setting("BOOKS") or _setting("BOOK", "calculus")).split(",")
         if b.strip()]
# Read lazily-tolerant: a missing key should surface as a clear 500 on the
# one endpoint that needs it, not as an import-time crash that takes the
# whole function down and reports nothing useful.
AAI_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
AGENT_ID = _setting("AGENT_ID", "")

# How much audio may be sent through this deployment, as settings rather than
# constants -- a bigger host, or a self-hosted one, should be able to raise
# them without touching code. Note the upload ceiling is only ours down to
# whatever the platform itself enforces: Vercel rejects a request body over
# ~4.5 MB at the edge, before any of this runs. Raising MAX_UPLOAD_MB past that
# only helps somewhere without that cap.
MAX_UPLOAD_MB = float(_setting("MAX_UPLOAD_MB", "4"))
# 0 means no ceiling of ours. Kept as a setting rather than deleted so a
# deployment that is not sitting on someone's personal API key can put one
# back without a code change.
MAX_SOURCE_MB = float(_setting("MAX_SOURCE_MB", "0"))
# Books added per hour from one address; 0 disables the check entirely.
RATE_PER_HOUR = int(_setting("RATE_PER_HOUR", "0"))
# Files in one book. A LibriVox novel is 30-60 chapters.
MAX_PARTS = int(_setting("MAX_PARTS", "80"))

# Which voice engine the browser defaults to. "assemblyai" is the cloud Voice
# Agent (one websocket, STT+LLM+TTS on their servers, needs an API key);
# "local" runs Whistle + Needle + PocketTTS on this machine and can only work
# where this server runs, not on Vercel. The page can override it per session;
# this is the default it starts from.
VOICE_PROVIDER = _setting("VOICE_PROVIDER", "assemblyai").strip().lower()
if VOICE_PROVIDER not in ("assemblyai", "local"):
    VOICE_PROVIDER = "assemblyai"

app = FastAPI(title="Hark")


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    """Always answer with JSON.

    An unhandled error used to return Starlette's plain-text "Internal Server
    Error", and the page -- which does res.json() -- surfaced the nonsense
    "Unexpected token 'I', \"Internal S\"... is not valid JSON" instead of the
    real problem.
    """
    print(f"[error] {request.url.path}: {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500,
                        content={"detail": f"{type(exc).__name__}: {exc}"})
# Built-in indexes live in data/ (committed) or media/ (local-only titles whose
# audio and transcript must not go in a public repo; the Cloud Run image still
# carries them).
MEDIA = ROOT / "media"


def _load_catalog() -> Dict[str, dict]:
    """Per-title metadata: data/catalog.json, then media/catalog.local.json over
    it, so a local-only title never has to be named in the repo."""
    out: Dict[str, dict] = {}
    for path in (ROOT / "data" / "catalog.json", MEDIA / "catalog.local.json"):
        if not path.exists():
            continue
        try:
            for entry in json.loads(path.read_text(encoding="utf-8")):
                out[entry["id"]] = entry
        except Exception as exc:
            print(f"[catalog] {path.name} unreadable, skipped: {exc}")
    return out


def _open_builtins(ids) -> Dict[str, Library]:
    """Open each named index. A missing one is skipped with a note rather than
    taking the server down: a shelf one title short beats no shelf."""
    out: Dict[str, Library] = {}
    for book_id in ids:
        db = next((d / f"{book_id}.db" for d in (ROOT / "data", MEDIA)
                   if (d / f"{book_id}.db").exists()), None)
        if db is None:
            print(f"[books] built-in '{book_id}' has no index; skipped")
            continue
        out[book_id] = Library(db)
    return out


CATALOG = _load_catalog()
BUILTINS = _open_builtins(BOOKS) or _open_builtins(["calculus"])
BOOK = next(iter(BUILTINS))
library = BUILTINS[BOOK]
playheads = build_store()
# The local model stack. Constructing it loads nothing -- the engine binary and
# the weights are fetched on first use -- so this is safe on a host (Vercel)
# that will never touch the local path.
local_voice = LocalVoice()


def library_for(session_id: Optional[str]):
    """The book this listener is on, as something with Library's methods.

    The shipped book is a SQLite file in the repo; anything anyone added is in
    Redis. Both answer window() and search(), so neither tool below has to know
    which one it is holding.
    """
    book_id = playheads.get_book(session_id) or BOOK
    if book_id in BUILTINS:
        return BUILTINS[book_id], _builtin_title(book_id)
    rec = books.load(playheads, book_id)
    if not rec or rec.status != "ready":
        return library, _builtin_title(BOOK)
    return books.RedisLibrary(playheads, rec), rec.title


# ---------- browser -> backend ----------

class Playhead(BaseModel):
    session_id: str
    seconds: float
    book: Optional[str] = None
    brevity: Optional[str] = None
    lookback: Optional[float] = None
    tools_off: Optional[List[str]] = None


# Every tool the agent can use, as the Settings page lists them. The passage
# tool is the product itself, so it cannot be switched off; the rest can, per
# listener. A switched-off tool is refused at its endpoint (cloud agent) and
# routed to the passage instead (local pipeline).
TOOL_CATALOG = [
    ("passage_at_playhead", "What did that mean?", "Answers from what you just heard", True),
    ("go_to_topic", "Jump to a topic", "Moves the audio to where something is discussed", False),
    ("book_outline", "Overview", "What the whole title covers", False),
    ("recap", "Recap", "Summarises the last few minutes", False),
    ("explain_simply", "Explain simply", "Plain words and an everyday example", False),
    ("translate", "Translate", "Explains it in another language", False),
    ("quiz_me", "Quiz me", "Asks a question and marks your answer", False),
    ("save_concept", "Remember this", "Saves the idea to your notes", False),
    ("web_search", "Web search", "Looks things up on Google", False),
]
OPTIONAL_TOOLS = {tid for tid, _, _, always in TOOL_CATALOG if not always}


def _set_tools_off(session_id: Optional[str], ids) -> None:
    if not session_id:
        return
    off = sorted({str(i).strip() for i in (ids or [])} & OPTIONAL_TOOLS)
    playheads.kv_set(f"hark:toolsoff:{session_id}", ",".join(off), ttl=6 * 3600)


def _tools_off(session_id: Optional[str]) -> set:
    raw = playheads.kv_get(f"hark:toolsoff:{session_id}") if session_id else None
    return set(raw.split(",")) & OPTIONAL_TOOLS if raw else set()


def _tool_off_reply(name: str) -> dict:
    label = next((lbl for tid, lbl, _, _ in TOOL_CATALOG if tid == name), name)
    return {"ok": False, "message": (
        f"The listener has switched off '{label}' in Settings. Tell them in one "
        f"sentence that it is off and can be turned back on in Settings.")}


# How far back "what did that mean?" may look, chosen by the listener. The cap is
# not arbitrary: AssemblyAI truncates a tool response at 8 KiB, and five minutes
# of narration (~6 passages of ~630 chars) is the most that fits with room to
# spare. The passage text is also trimmed to a budget in case a reader is fast.
LOOKBACK_DEFAULT_S, LOOKBACK_MIN_S, LOOKBACK_MAX_S = 90.0, 30.0, 300.0
PASSAGE_BUDGET_CHARS = 5500


def _clamp_lookback(seconds) -> float:
    try:
        return max(LOOKBACK_MIN_S, min(LOOKBACK_MAX_S, float(seconds)))
    except (TypeError, ValueError):
        return LOOKBACK_DEFAULT_S


def _lookback(session_id: Optional[str]) -> float:
    raw = playheads.kv_get(f"hark:lookback:{session_id}") if session_id else None
    return _clamp_lookback(raw) if raw else LOOKBACK_DEFAULT_S


@app.post("/api/playhead")
def report_playhead(p: Playhead):
    """The browser reports where the audiobook is, roughly once a second.

    AssemblyAI calls the tool server-to-server and has no idea where playback
    is, so the position has to arrive out of band. Cheap, and it means the tool
    stays stateless from the agent's point of view.
    """
    playheads.set(p.session_id, max(0.0, p.seconds))
    # Which book, on the same seam and for the same reason as the position:
    # the browser knows, and the tool call from AssemblyAI cannot see it.
    if p.book:
        playheads.set_book(p.session_id, p.book)
    if p.brevity:
        playheads.set_brevity(p.session_id, p.brevity)
    if p.lookback is not None:
        playheads.kv_set(f"hark:lookback:{p.session_id}", str(_clamp_lookback(p.lookback)), ttl=6 * 3600)
    if p.tools_off is not None:
        _set_tools_off(p.session_id, p.tools_off)
    # The same heartbeat carries jumps back. The agent cannot move the audio
    # itself -- it runs on AssemblyAI's servers -- so go_to_topic leaves a
    # pending position here and the browser collects it within the second.
    seek = playheads.take_seek(p.session_id)
    out = {"ok": True}
    if seek is not None:
        out["seek"] = seek
    # And it carries the tool the agent just called, claimed once, so the page
    # can show what ran even though the call happened server-to-server.
    raw = playheads.kv_get(f"hark:tool:{p.session_id}")
    if raw:
        try:
            out["tool"] = json.loads(raw)
        except ValueError:
            pass
        playheads.kv_set(f"hark:tool:{p.session_id}", "", ttl=1)
    return out


class PriorQuestion(BaseModel):
    t: float = 0.0
    q: str


class ContextPost(BaseModel):
    session_id: Optional[str] = None
    questions: list[PriorQuestion] = []


@app.post("/api/context")
def set_context(c: ContextPost):
    """What this listener asked in earlier sessions.

    Notes live in the listener's browser, not here -- there are no accounts.
    The browser hands over a short digest at connect time so the agent has some
    continuity, and it expires with the session.
    """
    if not c.questions:
        return {"ok": True, "carried": 0}
    lines = [f"- at {int(q.t)//60}:{int(q.t)%60:02d}, they asked: {q.q.strip()[:160]}"
             for q in c.questions[-8:] if q.q.strip()]
    playheads.set_context(c.session_id, chr(10).join(lines))
    return {"ok": True, "carried": len(lines)}


# ---------- the judges' access code ----------
#
# Listening is free and stays open to anyone. The two things that spend
# AssemblyAI credit -- a voice session, and indexing a new book -- need the code
# when HARK_ACCESS_CODE is set (a Vercel setting, never in the repo; Vercel
# applies a changed setting on the next deploy). Unset, everything is open, as before.
ACCESS_CODE = _setting("ACCESS_CODE", "").strip()


def _require_code(request: Request) -> None:
    if not ACCESS_CODE:
        return
    given = request.headers.get("x-access-code", "").strip().upper()
    if not hmac.compare_digest(given.encode(), ACCESS_CODE.upper().encode()):
        raise HTTPException(401, "access code required")


@app.get("/api/session")
def new_session(request: Request, provider: Optional[str] = None,
                llm: Optional[str] = None):
    """Start a voice session.

    Two engines behind one door. `assemblyai` mints a short-lived agent token and
    the browser connects to the vendor directly, exactly as before. `local` runs
    the models on this machine, so the browser connects to our own websocket
    (`/ws/voice`) instead and no vendor key or credit is involved -- which is
    also why the judges' access code is not required for a local session. The
    local answer model (`llm`: gemini or gemma) also rides here.
    """
    chosen = (provider or VOICE_PROVIDER or "assemblyai").strip().lower()
    session_id = f"s{int(time.time()*1000)}{os.urandom(3).hex()}"

    if chosen == "local":
        return {
            "provider": "local",
            "session_id": session_id,
            "ws_path": "/ws/voice",
            "llm": llm or local_llm.default_llm(),
            "llms": local_llm.available_llms(),
            "book": BOOK,
            "duration": library.duration_hint(),
            "models": local_voice.status(),
        }

    _require_code(request)
    if not AAI_KEY:
        raise HTTPException(500, "ASSEMBLYAI_API_KEY is not set on this deployment.")
    if not AGENT_ID:
        raise HTTPException(500, "HARK_AGENT_ID is not set. Run scripts/create_agent.py.")
    url = ("https://agents.assemblyai.com/v1/token"
           "?expires_in_seconds=300&max_session_duration_seconds=1800")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {AAI_KEY}"})
    try:
        token = json.load(urllib.request.urlopen(req))["token"]
    except urllib.error.HTTPError as e:
        raise HTTPException(502, f"token mint failed: {e.read().decode()[:200]}")
    return {
        "provider": "assemblyai",
        "token": token,
        "agent_id": _session_agent(session_id),
        "session_id": session_id,
        "book": BOOK,
        "duration": library.duration_hint(),
    }


# ---------- one stored agent per listener ----------
#
# AssemblyAI calls an HTTP tool with the model's arguments and nothing else: no
# session id, no header that identifies the conversation (checked against a
# live call, 2026-09-23). HTTP tools cannot be set per session either --
# session.update rejects them. So with one shared agent, every tool call is
# anonymous, and the backend could only guess the listener from whoever
# reported a playhead last. Two people listening at once got each other's
# answers.
#
# The fix is to give each session its own copy of the agent, with the session
# id pinned in the tool URLs. AssemblyAI keeps URL query params on the request,
# so every tool call says exactly whose playhead it is about. Creating one takes
# ~0.6 s. If it fails, the shared agent still works for a single listener.

AGENTS_API = "https://agents.assemblyai.com/v1/agents"
PUBLIC_URL = _setting("PUBLIC_URL", "http://localhost:8000").rstrip("/")
SESSION_AGENT_PREFIX = "hark-session-"
SESSION_AGENT_MAX_AGE_S = 3 * 3600
_AGENT_SPEC = json.loads((Path(__file__).resolve().parent / "agent.json").read_text(encoding="utf-8"))


def _agents_call(method: str, url: str, body: Optional[dict] = None):
    req = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None,
                                 method=method, headers={"Authorization": f"Bearer {AAI_KEY}",
                                                         "Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=8).read()
    return json.loads(raw) if raw else {}


def _session_agent(session_id: str) -> str:
    spec = json.loads(json.dumps(_AGENT_SPEC))
    spec["name"] = SESSION_AGENT_PREFIX + session_id
    for tool in spec.get("tools", []):
        url = tool["http"]["url"].replace("__TOOL_BASE_URL__", PUBLIC_URL)
        tool["http"]["url"] = f"{url}?session_id={session_id}"
    try:
        agent_id = _agents_call("POST", AGENTS_API, spec)["id"]
    except Exception as exc:
        print(f"[agents] per-session agent failed, using the shared one: {exc}")
        return AGENT_ID
    playheads.kv_set(f"hark:agent:{session_id}", agent_id, ttl=SESSION_AGENT_MAX_AGE_S)
    _sweep_session_agents()
    return agent_id


def _sweep_session_agents() -> None:
    """Delete per-session agents older than a session can live. At most every 10 min."""
    if playheads.kv_incr("hark:agent-sweep", ttl=600) > 1:
        return
    try:
        now = time.time()
        for a in _agents_call("GET", AGENTS_API).get("agents", []):
            if not a.get("name", "").startswith(SESSION_AGENT_PREFIX):
                continue
            created = datetime.fromisoformat(a["created_at"]).replace(tzinfo=timezone.utc).timestamp()
            if now - created > SESSION_AGENT_MAX_AGE_S:
                _agents_call("DELETE", f"{AGENTS_API}/{a['id']}")
    except Exception as exc:
        print(f"[agents] sweep failed: {exc}")


class SessionEnd(BaseModel):
    session_id: str


@app.post("/api/session/end")
def end_session(e: SessionEnd):
    """Delete this listener's agent. Best effort: the sweep catches what this misses."""
    agent_id = playheads.kv_get(f"hark:agent:{e.session_id}")
    if agent_id and agent_id != AGENT_ID:
        try:
            _agents_call("DELETE", f"{AGENTS_API}/{agent_id}")
        except Exception as exc:
            print(f"[agents] delete failed: {exc}")
    return {"ok": True}


def _trim_passage(text: str) -> str:
    """Keep the newest PASSAGE_BUDGET_CHARS: the end is what "that" points at."""
    if len(text) <= PASSAGE_BUDGET_CHARS:
        return text
    cut = text[-PASSAGE_BUDGET_CHARS:]
    return "..." + cut[cut.find(" ") + 1:]


def _tool_session(from_body: Optional[str], request: Request) -> Optional[str]:
    """The session a tool call belongs to: pinned in the URL by _session_agent."""
    return request.query_params.get("session_id") or from_body


TOOL_NOTE_TTL = 60   # seconds a tool call stays claimable by the page


def _note_tool(session_id: Optional[str], name: str, arguments: dict) -> None:
    """Record a cloud tool call so the page can show it.

    AssemblyAI calls the tool from its own servers, so nothing about the call
    appears on the browser's websocket. Writing it into the shared store lets the
    next heartbeat carry it back -- the same out-of-band seam the playhead uses.
    """
    if not session_id:
        return
    try:
        playheads.kv_set(f"hark:tool:{session_id}",
                         json.dumps({"name": name, "arguments": arguments or {}}),
                         ttl=TOOL_NOTE_TTL)
    except Exception as exc:
        print(f"[tool] note failed: {exc}")


# ---------- AssemblyAI -> backend (the HTTP tool) ----------

class ToolCall(BaseModel):
    session_id: Optional[str] = None
    search: Optional[str] = None


@app.post("/tools/passage_at_playhead")
def passage_at_playhead(call: ToolCall, request: Request):
    """The whole idea, as one endpoint.

    Returns prose, not codes: the response is read by a language model that has
    to speak the answer, so a failure explains itself in a sentence it can say.
    """
    sid = _tool_session(call.session_id, request)
    _note_tool(sid, "passage_at_playhead", {"search": call.search} if call.search else {})
    return tool_passage(sid, call.search)


def tool_passage(session_id: Optional[str], search: Optional[str] = None) -> dict:
    """The passage at the listener's playhead, plus any named lookback.

    Shared by the cloud agent (over HTTP) and the local pipeline (in process),
    so both answer from the same retrieval and the same spoiler cap.
    """
    t = playheads.get(session_id)
    if t is None:
        return {"ok": False,
                "message": "I can't tell where you are in the book yet - "
                           "is it playing?"}
    lib, _title = library_for(session_id)
    back = _lookback(session_id)
    near = lib.window(t, before=back)
    if not near:
        return {"ok": False,
                "message": f"There's no indexed text around {int(t)//60}:{int(t)%60:02d}."}

    # The length rule goes first: at the end of a long passage the model half-obeyed
    # it (two long sentences for "one sentence").
    parts = [_length_rule(session_id), "",
             f"The listener is {int(t)//60} minutes {int(t)%60} seconds into the book.",
             f"This is what they have just heard (the last {int(back)} seconds):",
             _trim_passage(" ".join(c.text for c in near))]

    prior = playheads.get_context(session_id)
    if prior:
        parts += ["", "This listener has asked before:", prior,
                  "Only mention this if it is relevant to what they just asked."]

    if search:
        far = _search_earlier(lib, search, t, exclude={c.id for c in near})
        if far:
            parts += ["", f"Earlier in the book, on '{search}':",
                      " ".join(c.text for c in far)]

    return {"ok": True, "playhead_seconds": round(t, 1), "message": "\n".join(parts)}


class TopicCall(BaseModel):
    session_id: Optional[str] = None
    topic: str


@app.post("/tools/go_to_topic")
def go_to_topic(call: TopicCall, request: Request):
    """Navigation by meaning: the mirror image of passage_at_playhead.

    You cannot skim an audiobook. A sighted reader flips to the right page in
    seconds; by ear the only controls are a scrubber and guesswork. So the
    listener names a topic and the book moves to it.

    The spoiler cap deliberately does NOT apply here. It exists to stop the
    agent volunteering what is ahead; being asked to go there is consent.
    """
    sid = _tool_session(call.session_id, request)
    if "go_to_topic" in _tools_off(sid):
        return _tool_off_reply("go_to_topic")
    _note_tool(sid, "go_to_topic", {"topic": call.topic})
    return tool_go_to_topic(sid, call.topic)


def tool_go_to_topic(session_id: Optional[str], topic: str) -> dict:
    """Find where a topic is discussed and leave a seek for the browser."""
    vec = _embed(topic)
    if vec is None:
        return {"ok": False,
                "message": "I can't look up topics right now - say roughly where "
                           "you want to go instead."}
    lib, _title = library_for(session_id)
    hits = lib.search(vec, k=1)
    if not hits:
        return {"ok": False,
                "message": f"I couldn't find anything about {topic} in this book."}

    target = hits[0]
    # Start a little before the passage, so the listener hears it introduced
    # rather than landing mid-sentence.
    start = max(0.0, target.start_s - 8)
    playheads.request_seek(session_id, start)
    return {"ok": True, "seek_seconds": round(start, 1),
            "message": (f"Moving the book to {int(start)//60}:{int(start)%60:02d}, "
                        f"where this comes up. Tell the listener where you are taking "
                        f"them and what is there, in one sentence. This is what "
                        f"plays next:" + chr(10) + target.text)}


class OutlineCall(BaseModel):
    session_id: Optional[str] = None


@app.post("/tools/book_outline")
def book_outline(call: OutlineCall, request: Request):
    """What the whole book covers, so someone knows what they are walking into.

    The spoiler cap exists to stop the agent *volunteering* what is ahead. Being
    asked what a book is about is consent, the same way asking to be taken
    somewhere is -- so this deliberately reads the entire book, not the part
    already heard. Refusing here was the wrong behaviour: someone at 00:05 who
    asks "what is this?" got told it was too early to say, which is useless.

    Excerpts are sampled evenly across the duration rather than summarised by
    us. The agent already has a language model; what it lacks is the text.
    """
    sid = _tool_session(call.session_id, request)
    if "book_outline" in _tools_off(sid):
        return _tool_off_reply("book_outline")
    _note_tool(sid, "book_outline", {})
    return tool_outline(sid)


def tool_outline(session_id: Optional[str]) -> dict:
    """Excerpts sampled across the whole book, for 'what is this about?'"""
    lib, title = library_for(session_id)
    if not len(lib):
        return {"ok": False, "message": "There's no indexed book loaded right now."}

    picks = _spread(lib, 10)
    if not picks:
        return {"ok": False, "message": "I couldn't read the book's contents just now."}

    t = playheads.get(session_id)
    where = (f"They are {int(t)//60} minutes in." if t else "They are at the start.")
    body = "\n".join(f"[{int(c.start_s)//60:02d}:{int(c.start_s)%60:02d}] {c.text}"
                     for c in picks)
    return {"ok": True, "message": (
        f"Excerpts sampled evenly across the whole book, in order. "
        f"Total length {int(lib.duration_hint())//60} minutes. {where}\n\n{body}\n\n"
        f"Describe what this book covers and how it is organised, in three or "
        f"four sentences, so they know what they are getting into. They asked, "
        f"so telling them the shape of it is not a spoiler -- but if it is a "
        f"story, do not give away how it ends.\n\n{_length_rule(session_id)}")}


# ---------- learning tools: recap, explain simply, translate, quiz, save ----------
# Each is a variation on the same move -- take what the listener has heard,
# stop at the playhead, and tell the answer model what to do with it -- so they
# share one helper and keep the spoiler cap. The cloud agent calls them over
# HTTP; the local pipeline calls them in process through run_local_tool.

RECAP_MIN_DEFAULT, RECAP_MIN_MAX = 5.0, 10.0
QUIZ_TTL_S = 180
# While a quiz question is pending, these mean "not an answer".
QUIZ_CANCEL = re.compile(r"\b(stop|cancel|skip|no more|enough|never ?mind)\b", re.I)
QUIZ_AGAIN = re.compile(r"\b(another|next|new) (question|one)\b|\bquiz me\b", re.I)
_LANGUAGE_OK = re.compile(r"^[A-Za-z][A-Za-z \-()]{1,29}$")


def _heard(session_id: Optional[str], seconds: float):
    """(error, t, text) for the `seconds` of audio up to the playhead."""
    t = playheads.get(session_id)
    if t is None:
        return ({"ok": False, "message": "I can't tell where you are in the book yet - "
                                         "is it playing?"}, None, "")
    lib, _title = library_for(session_id)
    near = lib.window(t, before=seconds, after=0)
    if not near:
        return ({"ok": False, "message": f"There's no indexed text around "
                                         f"{int(t)//60}:{int(t)%60:02d}."}, t, "")
    return None, t, _trim_passage(" ".join(c.text for c in near))


def _clock(t: float) -> str:
    return f"{int(t)//60}:{int(t)%60:02d}"


def tool_recap(session_id: Optional[str], minutes=None) -> dict:
    """'Catch me up': the last few minutes, summarised in order."""
    try:
        minutes = max(1.0, min(RECAP_MIN_MAX, float(minutes or RECAP_MIN_DEFAULT)))
    except (TypeError, ValueError):
        minutes = RECAP_MIN_DEFAULT
    err, t, text = _heard(session_id, minutes * 60)
    if err:
        return err
    return {"ok": True, "passage": text, "message": (
        f"RECAP of the last {minutes:g} minutes, up to {_clock(t)}:\n{text}\n\n"
        f"Recap what they heard, in order, in three or four short spoken sentences. "
        f"Lead with the main idea. Nothing after {_clock(t)}.")}


def tool_explain_simply(session_id: Optional[str], style=None) -> dict:
    """Re-explain the passage at the playhead plainly, with an everyday example."""
    err, t, text = _heard(session_id, _lookback(session_id))
    if err:
        return err
    how = ("with one concrete example" if str(style or "").lower().startswith("ex")
           else "as if to a curious twelve-year-old, with one everyday analogy")
    return {"ok": True, "passage": text, "message": (
        f"What they just heard (up to {_clock(t)}):\n{text}\n\n"
        f"Explain the main idea of this simply, {how}. No jargon: if a technical "
        f"word is unavoidable, define it in passing. Three short sentences at most.")}


def tool_translate(session_id: Optional[str], language=None) -> dict:
    """Explain what was just heard in another language."""
    language = str(language or "").strip()
    if not _LANGUAGE_OK.match(language):
        return {"ok": False, "message": "Ask which language they would like it in."}
    err, t, text = _heard(session_id, _lookback(session_id))
    if err:
        return err
    return {"ok": True, "language": language, "passage": text, "message": (
        f"What they just heard (up to {_clock(t)}):\n{text}\n\n"
        f"Explain what this passage says, in {language}. Write the whole answer in "
        f"{language}, in its usual script, in two or three simple sentences. "
        f"Keep names and technical terms recognisable.")}


def tool_quiz(session_id: Optional[str], answer=None, question=None) -> dict:
    """Ask one question on what was heard; or, given their answer, mark it."""
    err, t, text = _heard(session_id, max(180.0, _lookback(session_id)))
    if err:
        return err
    if answer and question:
        return {"ok": True, "passage": text, "message": (
            f"What they heard (up to {_clock(t)}):\n{text}\n\n"
            f"QUIZ. You asked: {question}\nThey answered: {answer}\n\n"
            f"Tell them whether that is right, in one or two sentences. If it is "
            f"wrong or partly right, give the correct answer from the passage. "
            f"Then ask if they want another question.")}
    return {"ok": True, "passage": text, "message": (
        f"What they heard (up to {_clock(t)}):\n{text}\n\n"
        f"QUIZ. Ask them ONE short question that checks they understood the key "
        f"idea here -- not a trivia detail. Say only the question, then stop and "
        f"wait for their answer. When they answer, judge it against this passage.")}


def tool_save(session_id: Optional[str], concept=None) -> dict:
    """'Remember this': the page saves a note at the playhead; we confirm it."""
    err, t, text = _heard(session_id, 30.0)
    if err:
        return err
    concept = str(concept or "").strip()[:120]
    snippet = text[-300:]
    snippet = snippet[snippet.find(" ") + 1:] if len(text) > 300 else snippet
    return {"ok": True, "passage": "Saved to your notes: " + snippet,
            "note": {"at": round(t, 1), "concept": concept, "text": snippet},
            "message": (
                f"SAVED to their notes at {_clock(t)}"
                f"{': ' + concept if concept else ''}. The passage saved:\n{snippet}\n\n"
                f"Confirm in ONE short sentence, naming the idea that was saved.")}


class LearningCall(BaseModel):
    session_id: Optional[str] = None
    minutes: Optional[float] = None
    style: Optional[str] = None
    language: Optional[str] = None
    answer: Optional[str] = None
    question: Optional[str] = None
    concept: Optional[str] = None


def _learning_args(call: LearningCall) -> dict:
    return {k: v for k, v in call.dict().items()
            if k != "session_id" and v not in (None, "")}


@app.post("/tools/recap")
def recap(call: LearningCall, request: Request):
    sid = _tool_session(call.session_id, request)
    if "recap" in _tools_off(sid):
        return _tool_off_reply("recap")
    _note_tool(sid, "recap", _learning_args(call))
    return tool_recap(sid, call.minutes)


@app.post("/tools/explain_simply")
def explain_simply(call: LearningCall, request: Request):
    sid = _tool_session(call.session_id, request)
    if "explain_simply" in _tools_off(sid):
        return _tool_off_reply("explain_simply")
    _note_tool(sid, "explain_simply", _learning_args(call))
    return tool_explain_simply(sid, call.style)


@app.post("/tools/translate")
def translate(call: LearningCall, request: Request):
    sid = _tool_session(call.session_id, request)
    if "translate" in _tools_off(sid):
        return _tool_off_reply("translate")
    _note_tool(sid, "translate", _learning_args(call))
    return tool_translate(sid, call.language)


@app.post("/tools/quiz_me")
def quiz_me(call: LearningCall, request: Request):
    sid = _tool_session(call.session_id, request)
    if "quiz_me" in _tools_off(sid):
        return _tool_off_reply("quiz_me")
    _note_tool(sid, "quiz_me", _learning_args(call))
    return tool_quiz(sid, call.answer, call.question)


@app.post("/tools/save_concept")
def save_concept(call: LearningCall, request: Request):
    sid = _tool_session(call.session_id, request)
    if "save_concept" in _tools_off(sid):
        return _tool_off_reply("save_concept")
    _note_tool(sid, "save_concept", _learning_args(call))
    return tool_save(sid, call.concept)


class WebSearchCall(BaseModel):
    session_id: Optional[str] = None
    query: Optional[str] = None


@app.post("/tools/web_search")
def web_search(call: WebSearchCall, request: Request):
    """Look something up on the live web, for what the book cannot answer."""
    sid = _tool_session(call.session_id, request)
    if "web_search" in _tools_off(sid):
        return _tool_off_reply("web_search")
    _note_tool(sid, "web_search", {"query": call.query or ""})
    return tool_web_search(sid, call.query)


def tool_web_search(session_id: Optional[str], query: Optional[str]) -> dict:
    """Google Search, grounded through Gemini, phrased for the agent to speak.

    The book is the source of truth for what was said; the web is for what the
    book assumes or cannot know -- a definition, the latest research, whether a
    claim still holds. The answer is labelled as coming from the web.
    """
    query = (query or "").strip()[:300]
    if not query:
        return {"ok": False, "message": "There was nothing to look up. Ask them what "
                                         "they would like searched."}
    _, title = library_for(session_id)
    found = local_llm.google_search(query, context=title)
    if not found:
        return {"ok": False, "message": (
            "Web search is not available right now. Say you could not check online, "
            "and answer from the book if it covers the question.")}
    names = ", ".join(t for t, _ in found["sources"][:3]) or "the web"
    return {"ok": True, "passage": "From the web: " + found["text"], "message": (
        f"Google Search results for \"{query}\":\n{found['text']}\n\n"
        f"Sources: {names}\n\n"
        f"Answer from these results, and say it comes from the web, not the book "
        f"(name one source).\n\n{_length_rule(session_id)}"),
        "sources": [{"title": t, "url": u} for t, u in found["sources"]]}


def _length_rule(session_id: Optional[str]) -> str:
    """How long the answer should be, as a line the agent will read.

    The stored agent's prompt is fixed for everyone, so the preference travels
    on the tool response instead -- the one channel that is already per-session
    and that the model definitely reads.
    """
    if (playheads.get_brevity(session_id) or "full") == "short":
        return ("ANSWER LENGTH: ONE sentence, under 25 words. The listener chose short "
                "answers. Say the single most useful thing and stop: no context, no "
                "caveats, no second sentence.")
    return ("ANSWER LENGTH: two or three sentences. Do not restate the question, and "
            "do not close by summarising what you just said.")


def _spread(lib, k: int = 10) -> list:
    """K chunks spaced evenly across the book, in reading order."""
    duration = lib.duration_hint()
    if not duration:
        return []
    picked, seen = [], set()
    for i in range(k):
        # Sample at the middle of each band rather than the edges, so the first
        # pick is not the title page and the last is not the licence notice.
        hits = lib.window(duration * (i + 0.5) / k, before=0.0, after=0.0)
        if hits and hits[0].id not in seen:
            seen.add(hits[0].id)
            picked.append(hits[0])
    return picked


def _embed(query: str):
    """One embedding, or None if embeddings are unavailable."""
    try:
        from hark.core.brain import GeminiEmbedder
        vertex = local_llm.vertex_client()
        key = os.getenv("GEMINI_API_KEY", "")
        if vertex is None and not key:
            return None
        return GeminiEmbedder(key or None, client=vertex)([query], "query")[0]
    except Exception as exc:
        print(f"[tool] embed failed: {exc}")
        return None


def _search_earlier(lib, query: str, before_s: float, exclude) -> list:
    """Semantic lookback, capped at the playhead so it cannot spoil the book."""
    vec = _embed(query)
    if vec is None:
        return []
    try:
        hits = lib.search(vec, k=5, before_s=before_s)
        return [c for c in hits if c.id not in exclude][:2]
    except Exception as exc:
        print(f"[tool] lookback skipped: {exc}")
        return []


# ---------- the local voice pipeline ----------
#
# The cloud agent runs on AssemblyAI's servers and the browser talks to it
# directly; nothing on this host can imitate that, so in local mode the browser
# connects here and we own the loop: endpoint the speech, transcribe with
# Whistle, let Needle pick the tool, run the same retrieval the cloud agent
# runs, write the answer with Gemini (the one piece kept in the cloud, by
# choice), and speak it with PocketTTS.
#
# The wire protocol is deliberately the AssemblyAI one -- the same event names
# the browser already understands -- so the client's message loop is unchanged
# and only the socket address differs.

# A small local model cannot follow the full cloud agent prompt -- it is long,
# it talks about tools the model is not calling, and it makes a 0.5B model
# refuse or echo the question. This is the same job in four sentences.
_SMALL_MODEL_SYSTEM = (
    "You answer one question about an audiobook passage the listener just heard. "
    "Use only the passage. Reply in one or two short sentences, spoken aloud: no "
    "lists, no headings, no markdown. Never repeat the question, never mention "
    "tools or passages, and never say you cannot help.")


def _local_system_prompt(short: bool = False) -> str:
    if short:
        return _SMALL_MODEL_SYSTEM
    base = _AGENT_SPEC.get("system_prompt", "").strip()
    return (base + "\n\nThe book passage has ALREADY been retrieved for you and "
            "appears as a TOOL RESULT below. Do not ask the listener for anything "
            "and do not call tools: just answer from that result, out loud, in the "
            "language the listener spoke.")


# Emit sentences, not the whole reply: the first one can go to TTS while the
# rest is still being written. The minimum length stops a stray "Yes." becoming
# its own clip, which sounds clipped.
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def local_answer_stream(question: str, tool_message: str, llm: Optional[str] = None):
    """Yield the spoken reply sentence by sentence as the answer model writes it.

    The model is chosen per session -- Gemini in the cloud, or Gemma on a local
    llama.cpp server (see hark/server/local_llm.py) -- and streams its text, so the first sentence
    can be spoken while the rest is still being generated.
    """
    model = local_llm.build_answer_model(llm)
    if model is None:
        return
    prompt = (f"The listener just interrupted the book and said:\n{question}\n\n"
              f"TOOL RESULT (their position and the relevant passage):\n{tool_message}\n\n"
              f"Answer them now, following the rules.")
    system = _local_system_prompt(short=getattr(model, "name", "") == "gemma")
    buf = ""
    for piece in model.stream(prompt, system):
        buf += piece
        while True:
            m = _SENTENCE_END.search(buf)
            if not m or m.end() < 20:
                break
            sentence, buf = buf[:m.end()].strip(), buf[m.end():]
            yield sentence
    if buf.strip():
        yield buf.strip()


def _fallback_answer(tool_result: dict) -> str:
    """Answer model unavailable: say something true from the retrieved passage.

    Only the passage is read, not the instruction wrapper around it -- the tool
    message is written for a model, and read aloud verbatim it starts "The
    listener is 2 minutes 0 seconds into the book". And only the last sentence
    or two: the newest words are what "that" points at, and cutting a fixed
    slice off the front lands mid-word ("ense of direction"), which sounds
    broken read aloud.
    """
    message = tool_result.get("message", "") or ""
    # Tools that wrap the passage in instructions hand it over separately.
    passage = (tool_result.get("passage") or "").strip()
    if not passage and "just heard" in message:
        after = message.split("just heard", 1)[1]
        passage = after.split(":", 1)[1] if ":" in after else after
        passage = passage.split(chr(10) + chr(10), 1)[0].strip()
    if not passage:
        passage = message.split(chr(10) + chr(10))[-1].strip()
    if not passage:
        return "I could not reach the answer model just now."

    # The window is ~90 s before the playhead and ~15 s after it, so the last
    # ~14% is material the listener has NOT heard yet. Read aloud verbatim that
    # spoils what comes next -- drop the tail before choosing what to say.
    if len(passage) > 200:
        passage = passage[: int(len(passage) * 0.85)]

    tail = passage[-500:]
    if " " in tail:
        tail = tail[tail.find(" ") + 1:]          # drop a half-word at the front
    ends = [m.end() for m in re.finditer(r"[.!?]\s", tail)]
    if len(ends) >= 3:
        tail = tail[ends[-3]:]
    return "From the book: " + tail.strip()


def run_local_tool(session_id: Optional[str], name: str, args: dict) -> dict:
    """Execute the tool Needle chose, with the same retrieval the cloud uses."""
    args = args or {}
    if name == "go_to_topic":
        return tool_go_to_topic(session_id, str(args.get("topic", "")).strip())
    if name == "book_outline":
        return tool_outline(session_id)
    if name == "web_search":
        return tool_web_search(session_id, str(args.get("query", "")).strip())
    if name == "recap":
        return tool_recap(session_id, args.get("minutes"))
    if name == "explain_simply":
        return tool_explain_simply(session_id, args.get("style"))
    if name == "translate":
        return tool_translate(session_id, args.get("language"))
    if name == "quiz_me":
        return tool_quiz(session_id, args.get("answer"), args.get("question"))
    if name == "save_concept":
        return tool_save(session_id, args.get("concept"))
    return tool_passage(session_id, args.get("search"))


def _audio_frames(pcm16: bytes, chunk_ms: int = 200):
    """Split a reply into browser-sized pieces (~200 ms) so playback starts early."""
    step = int(WIRE_RATE * chunk_ms / 1000) * 2      # 2 bytes per sample
    for i in range(0, len(pcm16), step):
        yield pcm16[i:i + step]


# A live partial transcript, so the page shows words while the listener is still
# talking -- the "capturing" feedback the cloud path gives for free. Throttled so
# a long question does not queue one transcribe per audio frame, and held back
# until there is enough audio for a stable guess (a 0.6 s buffer produced
# nonsense like "Everybody loves him!").
PARTIAL_INTERVAL_S = 1.0
PARTIAL_MIN_S = 1.1
# Less voiced audio than this in a turn is not a question (only checked when
# Silero is gating; loudness alone cannot tell).
MIN_VOICED_S = 0.45


async def _partial_transcript(ws: WebSocket, endpointer, generation: int,
                              audio: np.ndarray) -> None:
    """Transcribe the turn so far and show it; dropped if the turn has moved on."""
    try:
        text = (await run_in_threadpool(
            local_voice.stt.transcribe, resample(audio, WIRE_RATE, STT_RATE)) or "").strip()
        if text and endpointer.generation == generation:
            await ws.send_json({"type": "transcript.user.delta", "text": text})
    except Exception as exc:
        print(f"[local] partial transcript failed: {exc}")


async def _stream_reply(ws: WebSocket, sentences, holder: dict,
                        voice: Optional[str] = None) -> None:
    """Speak a reply while it is still being written.

    `sentences` is a lazy iterable of the reply's sentences (Gemini's stream)
    and `holder` collects the full text for the transcript. Generation and
    synthesis are both blocking, so they run in one worker thread and hand
    events to the loop through a queue: ("text", sentence) as each sentence is
    written, then ("audio", pcm) as it is spoken, so the page shows the words
    and hears them in step -- and the first sentence is heard while the last is
    still being generated.
    """
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue" = asyncio.Queue()
    done = object()

    def emit(kind, payload):
        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    def worker():
        parts = []
        try:
            for sentence in sentences:
                parts.append(sentence)
                holder["text"] = " ".join(parts).strip()
                emit("text", sentence)
                for chunk in local_voice.tts.stream(sentence, voice=voice):
                    emit("audio", chunk)
        except Exception as exc:
            print(f"[local] reply stream failed: {exc}")
        finally:
            # Nothing was generated -- the answer model is down or returned an
            # empty call. Speak the passage so the page both shows words and
            # hears them; an empty model is not an exception and used to slip
            # past the fallback entirely, leaving the reply silent.
            if not parts:
                try:
                    emit("audio", local_voice.tts.speak(holder.get("fallback", ""), voice=voice))
                except Exception as exc2:
                    print(f"[local] tts failed entirely: {exc2}")
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await queue.get()
        if item is done:
            break
        kind, payload = item
        if kind == "audio" and payload:
            await ws.send_json({"type": "reply.audio",
                                "data": base64.b64encode(payload).decode("ascii")})
        elif kind == "text" and payload:
            await ws.send_json({"type": "transcript.agent", "text": payload})


@app.post("/api/local/warm")
async def warm_local():
    """Load the three models so the first question is not the slow one."""
    status = await run_in_threadpool(local_voice.warm)
    return {"ok": True, "models": status}


@app.websocket("/ws/voice")
async def voice_socket(ws: WebSocket):
    """The local equivalent of the AssemblyAI Voice Agent websocket."""
    await ws.accept()
    sid = ws.query_params.get("session_id")
    llm = ws.query_params.get("llm")
    if ws.query_params.get("tools_off") is not None:
        _set_tools_off(sid, ws.query_params.get("tools_off", "").split(","))
    # Allowlisted: an unknown voice is the default, never a URL to fetch.
    voice = tts_voice(ws.query_params.get("voice"))
    await ws.send_json({"type": "session.ready", "session_id": sid, "llm": llm,
                        "voice": voice})
    # Silero takes ~0.3 s to load; pay it off the event loop.
    endpointer = Endpointer(WIRE_RATE, vad=await run_in_threadpool(new_speech_detector))
    partials = os.getenv("HARK_STT_PARTIALS", "1") != "0"
    partial_task = None
    last_partial = 0.0
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") != "input.audio":
                continue
            frame = pcm16_to_float(base64.b64decode(msg.get("audio", "")))
            event = endpointer.feed(frame)
            if event is None:
                if partial_task is not None and partial_task.done():
                    partial_task = None
                # Show words while the listener is still talking, the way the
                # cloud path does. Throttled, and non-blocking: it runs as its
                # own task so it never holds up ingesting the audio.
                if (partials and partial_task is None and endpointer.speaking
                        and time.monotonic() - last_partial > PARTIAL_INTERVAL_S):
                    audio = endpointer.buffered()
                    if audio is not None and len(audio) >= PARTIAL_MIN_S * WIRE_RATE:
                        last_partial = time.monotonic()
                        partial_task = asyncio.create_task(_partial_transcript(
                            ws, endpointer, endpointer.generation, audio))
                continue
            kind, audio = event
            if kind == "start":
                await ws.send_json({"type": "input.speech.started"})
                continue
            await ws.send_json({"type": "input.speech.stopped"})

            audio16 = resample(audio, WIRE_RATE, STT_RATE)
            print(f"[local] utterance {len(audio) / WIRE_RATE:.1f}s "
                  f"({len(audio16) / STT_RATE:.1f}s at 16k)")
            # Too short to be a question: a click, a cough, or a consonant burst
            # from the speakers. Transcribing it just invites a hallucination.
            # Every way out below closes the turn with reply.done: the page shows
            # "Thinking" from speech.stopped until it hears back, and a silent
            # drop left it there for good.
            if len(audio16) < MIN_UTTERANCE_S * STT_RATE:
                print("[local] dropped: too short")
                await ws.send_json({"type": "reply.done", "status": "dropped"})
                continue
            # Long enough, but mostly not a voice: a fan, a door, the TV. Whistle
            # would put words to it anyway.
            if endpointer.gated and endpointer.last_voiced_s < MIN_VOICED_S:
                print(f"[local] dropped: only {endpointer.last_voiced_s:.2f}s of speech")
                await ws.send_json({"type": "reply.done", "status": "dropped"})
                continue
            text = (await run_in_threadpool(local_voice.stt.transcribe, audio16) or "").strip()
            print(f"[local] heard: {text!r}")
            if not text:
                await ws.send_json({"type": "reply.done", "status": "dropped"})
                continue
            # Whisper-family models answer noise with a handful of stock phrases
            # ("Thank you.", "Thanks for watching"). On a short clip, one of those
            # is the model filling in, not the listener talking.
            if text.lower().strip(" .,!?") in HALLUCINATIONS:
                print(f"[local] dropped a likely hallucination: {text!r}")
                await ws.send_json({"type": "reply.done", "status": "dropped"})
                continue
            await ws.send_json({"type": "transcript.user", "text": text})

            # A filler ("Uh...", "yeah") is worth showing -- the listener sees
            # their words -- but there is no question in it. Answering invents
            # one, and a small model just echoes the prompt back. Give the book
            # back instead.
            norm = text.lower().strip(" .,!?\u2026")
            if len(norm) < 2 or norm in CONTENTLESS:
                print(f"[local] no content, not answering: {text!r}")
                await ws.send_json({"type": "reply.done", "status": "complete"})
                continue

            quiz_key = f"hark:quiz:{sid}"
            pending = (playheads.kv_get(quiz_key)
                       if sid and "quiz_me" not in _tools_off(sid) else None)
            if pending and not QUIZ_CANCEL.search(text) and not QUIZ_AGAIN.search(text):
                # They are answering the question Hark just asked.
                name, args = "quiz_me", {"question": pending, "answer": text}
                playheads.kv_set(quiz_key, "", ttl=1)
            else:
                if pending:
                    playheads.kv_set(quiz_key, "", ttl=1)
                name, args = await run_in_threadpool(local_voice.router.route, text)
                if name in _tools_off(sid):
                    print(f"[local] {name} is switched off; answering from the passage")
                    name, args = "passage_at_playhead", {}
            # The router picks web_search but may leave the query out; the
            # question itself is the best query then.
            if name == "web_search" and not str((args or {}).get("query", "")).strip():
                args = {**(args or {}), "query": text}
            print(f"[local] tool: {name} {args}")
            started = time.monotonic()
            result = await run_in_threadpool(run_local_tool, sid, name, args)
            # Show what the agent decided to do, before the answer. In the cloud
            # path the same thing rides the heartbeat, because AssemblyAI calls
            # the tool server-to-server and the page cannot see it.
            await ws.send_json({"type": "tool.call", "name": name, "arguments": args})
            if name == "save_concept" and result.get("note"):
                await ws.send_json({"type": "note.saved", **result["note"]})
            await ws.send_json({"type": "reply.started"})
            # Written and spoken sentence by sentence: the first sentence plays
            # while the rest is still being generated.
            holder = {"text": "", "fallback": _fallback_answer(result)}
            sentences = local_answer_stream(text, result.get("message", ""), llm)
            await _stream_reply(ws, sentences, holder, voice)
            print(f"[local] answered in {time.monotonic() - started:.1f}s "
                  f"({len(holder['text'])} chars)")
            if name == "quiz_me" and not args.get("answer") and holder["text"] and sid:
                # Remember the question, so the next thing they say is marked.
                playheads.kv_set(f"hark:quiz:{sid}", holder["text"][:400], ttl=QUIZ_TTL_S)
            if not holder["text"]:
                # Gemini produced nothing, so the worker spoke the passage; the
                # page still needs the words to show.
                await ws.send_json({"type": "transcript.agent", "text": holder["fallback"]})
            await ws.send_json({"type": "reply.done", "status": "complete"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[local] session error: {type(exc).__name__}: {exc}")
        try:
            await ws.send_json({"type": "session.error", "message": str(exc)})
        except Exception:
            pass


# ---------- bring your own audiobook ----------

class NewBook(BaseModel):
    title: str = ""
    audio_url: str = ""
    # A real audiobook is one file per chapter. Give them in reading order and
    # they become one continuous book.
    audio_urls: list[str] = []


def _embedder():
    """The Gemini embedder, or None if this deployment has no key."""
    try:
        from hark.core.brain import GeminiEmbedder
        vertex = local_llm.vertex_client()
        key = os.getenv("GEMINI_API_KEY", "")
        if vertex is None and not key:
            return None
        return GeminiEmbedder(key or None, client=vertex)
    except Exception as exc:
        print(f"[books] embedder unavailable: {exc}")
        return None


def _builtin_title(book_id: str) -> str:
    title = CATALOG.get(book_id, {}).get("title")
    if title:
        return title
    return _setting("BOOK_TITLE", "Calculus Made Easy, ch. 3") if book_id == BOOK else book_id


def _builtin_card(book_id: Optional[str] = None) -> dict:
    """A built-in title as the shelf shows it. kind, author, topic and blurb come
    from the catalog and are empty for a title it does not describe."""
    book_id = book_id or BOOK
    lib = BUILTINS[book_id]
    meta = CATALOG.get(book_id, {})
    return {"id": book_id, "title": _builtin_title(book_id),
            "audio_url": meta.get("audio_url") or f"/audio/{book_id}.mp3",
            "status": "ready", "error": "", "progress": 100, "chunks": len(lib),
            "duration": round(lib.duration_hint(), 1), "builtin": True,
            "kind": meta.get("kind", "book"), "author": meta.get("author", ""),
            "topic": meta.get("topic", ""), "blurb": meta.get("blurb", ""),
            "source": meta.get("source", "")}


# Public domain narration, so anyone landing on the demo has something real to
# try without hunting for a link. These are indexed on demand into the visitor's
# own shelf, not pre-built: it costs nothing until someone wants one, and
# watching it index is the clearest demonstration of what this does.
# Already transcribed and indexed, and shown to everyone. Seeded once by
# scripts/seed_featured.py, which uses these exact ids so a re-run updates the
# same books rather than creating strangers.
# Seeded by scripts/seed_featured.py, which is where the source URLs live.
# This list is only the ids and their order on the shelf: longest first,
# because the length is the point -- a 54-hour book you can ask anything of at
# any second is the claim, and it should be the first row, not a footnote.
# An id that is missing or still indexing is skipped, so this can name books
# that are not finished yet.
FEATURED_IDS = [
    "featured-dumas-monte-cristo",             # 117 files, 54.3 h
    "featured-dostoevsky-crime-punishment",    #  40 files, 23.4 h
    "featured-locke-understanding",            #  32 files, 14.7 h
    "featured-thoreau-walden",                 #  23 files, 14.3 h
    "featured-hume-treatise",                  #  40 files, 14.0 h
    "featured-shelley-frankenstein-es",        #  28 files, 11.6 h (Spanish)
    "featured-thompson-calculus",              #  58 files, 10.1 h
    "featured-wilde-dorian-gray",              #  13 files,  6.2 h
    "featured-russell-problems-full",          #  15 files,  4.8 h
    "featured-wittgenstein-tractatus",         #   6 files,  4.2 h
    "featured-bennett-24hours",                #  13 files,  1.6 h
    "featured-allen-as-a-man-thinketh",        #   8 files,  0.9 h
]

# NOT indexed. One tap adds them, which takes a minute and is the clearest
# demonstration of what this does -- a book that did not exist when you sat
# down, answering questions about itself.
SUGGESTED: list[dict] = [
    # Deliberately NOT on the featured shelf and deliberately NOT indexed:
    # these exist to be added live. One tap transcribes and indexes a book that
    # did not exist a minute earlier, which is the only way to answer "isn't
    # this just RAG over a corpus you prepared?" without an argument.
    # Both are short on purpose -- a chapter indexes in well under a minute,
    # which is a shot you can film, and both are verified to serve.
    {"title": "Sun Tzu - The Art of War, ch. 1-2",
     "audio_url": "https://archive.org/download/art_of_war_librivox/art_of_war_01-02_sun_tzu_64kb.mp3",
     "note": "Eight minutes - indexes while you watch"},
    {"title": "Marcus Aurelius - Meditations, book 2",
     "audio_url": "https://archive.org/download/themeditationsofmarcusaurelius_1801_librivox/meditationsofmarcusaurelius_02_aurelius_64kb.mp3",
     "note": "Thirteen minutes - add it and ask it something"},
]



def _client_id(request: Request) -> str:
    """A random string the browser keeps in localStorage. Scopes a shelf so one
    visitor's books do not appear on another's; not a credential."""
    raw = (request.headers.get("x-client-id") or "").strip()
    return re.sub(r"[^A-Za-z0-9_-]", "", raw)[:48]


def _rate_limit(request: Request) -> None:
    """Per-IP ceiling on new books.

    The endpoint is unauthenticated by design -- there are no accounts -- so
    the thing that needs protecting is the transcription spend behind it.
    Called only once a link has passed validation: a rejected paste costs us
    one HEAD request, and charging someone's quota for a typo is just rude.

    RATE_PER_HOUR = 0 turns this off. The owner of the key gets to decide how
    much of it to leave lying in the road; the SSRF and content-type guards in
    books.py are not optional in the same way and stay on regardless.
    """
    if RATE_PER_HOUR <= 0:
        return
    ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() or "unknown"
    try:
        n = playheads.kv_incr(f"hark:rate:{ip}", 3600)
    except Exception as exc:
        print(f"[rate] check failed, allowing: {exc}")
        return
    if n > RATE_PER_HOUR:
        raise HTTPException(429, f"That's {RATE_PER_HOUR} books in an hour from this "
                                 f"address, which is the limit here. Try again later.")


def _featured() -> list[dict]:
    """Ready-made books, shown to everyone. Missing ones are skipped silently:
    a shelf that is one book short beats a page that will not load."""
    out = []
    found = books.load_many(playheads, FEATURED_IDS)
    for book_id in FEATURED_IDS:          # FEATURED_IDS fixes the shelf order
        rec = found.get(book_id)
        if rec and rec.status == "ready":
            card = rec.public()
            card["featured"] = True      # not removable; it is not their book
            out.append(card)
    return out


@app.get("/api/books")
def list_books(request: Request):
    """The built-in titles, the featured ones, then this browser's own."""
    mine = books.shelf(playheads, _client_id(request))
    featured = [f for f in _featured() if f["id"] not in {b["id"] for b in mine}]
    return {"books": [_builtin_card(b) for b in BUILTINS] + featured + mine,
            "suggested": SUGGESTED,
            # The page reads its own ceilings from here rather than hardcoding
            # them, so raising a setting moves the check and the wording with it.
            "limits": {"upload_mb": MAX_UPLOAD_MB, "source_mb": MAX_SOURCE_MB}}


@app.post("/api/books")
def add_book(new: NewBook, request: Request):
    """Start indexing a book from a URL.

    Returns as soon as the transcription job is queued. The browser then polls
    GET /api/books/{id}, and each poll advances the work by one step -- there
    is no worker here, and a request that waited for a whole audiobook would be
    killed by the platform long before it finished.
    """
    _require_code(request)
    if not AAI_KEY:
        raise HTTPException(500, "ASSEMBLYAI_API_KEY is not set on this deployment.")
    urls = [u.strip() for u in (new.audio_urls or [new.audio_url]) if u and u.strip()]
    if not urls:
        raise HTTPException(400, "No audio link given.")
    if len(urls) > MAX_PARTS:
        raise HTTPException(400, f"That's {len(urls)} files; {MAX_PARTS} is the most "
                                 f"this deployment will take as one book.")
    try:
        # Checked before anything is fetched or queued: this runs from our own
        # server, so an unvalidated link is a request forgery, and an unbounded
        # one is someone else's transcription bill. Every part is checked --
        # one bad link in chapter nine should fail now, not in ten minutes.
        for u in urls:
            books.check_source(u, int(MAX_SOURCE_MB * 1_000_000))
        _rate_limit(request)
        rec = books.create(playheads, new.title or _title_from_url(urls[0]), urls,
                           AAI_KEY, _client_id(request))
    except books.RejectedURL as e:
        raise HTTPException(400, str(e))
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:200]
        raise HTTPException(400, f"AssemblyAI could not take that link: {detail}")
    return rec.public()


@app.post("/api/books/upload")
async def upload_book(request: Request):
    """Add a book from a file on the listener's machine.

    The audio goes straight to AssemblyAI and is never stored here. The
    platform caps a request body at 4.5 MB, so this path is for a chapter or
    an episode; the page checks the size first and steers longer books to the
    URL form rather than letting them fail at the edge.
    """
    _require_code(request)
    if not AAI_KEY:
        raise HTTPException(500, "ASSEMBLYAI_API_KEY is not set on this deployment.")
    _rate_limit(request)
    title = request.headers.get("x-book-title", "") or "Uploaded book"
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "No audio arrived.")
    if len(raw) > MAX_UPLOAD_MB * 1_000_000:
        raise HTTPException(413, f"That file is {len(raw) / 1e6:.1f} MB and this "
                                 f"deployment accepts {MAX_UPLOAD_MB:g} MB - "
                                 f"paste a link to it instead.")
    try:
        # Hand the blocking work to a thread: a long recording takes many
        # minutes to reach AssemblyAI, and doing it on the event loop froze
        # every other request -- the page could not even poll for progress.
        url = await run_in_threadpool(books.upload_bytes, raw, AAI_KEY)
        rec = await run_in_threadpool(
            books.create, playheads, title, url, AAI_KEY, _client_id(request))
    except urllib.error.HTTPError as e:
        raise HTTPException(400, f"Upload failed: {e.read().decode()[:200]}")
    except OSError as e:
        # A long transfer over a slow link resets, stalls, or times out.
        # Uncaught, that came back as a plain-text 500 the page could not read,
        # so the real failure was invisible.
        raise HTTPException(502, f"The upload did not finish ({type(e).__name__}). "
                                 f"Try again, use a smaller file, or add it by link.")
    except ValueError as e:
        raise HTTPException(400, str(e))
    # The browser plays its own local copy; this URL is AssemblyAI-only.
    out = rec.public()
    out["audio_url"] = ""
    return out


@app.delete("/api/books/{book_id}")
def remove_book(book_id: str, request: Request):
    """Take a book off this browser's shelf.

    The index itself is left alone: it expires on its own, and someone else
    may be holding the same book. This unlists it for the person who asked.
    """
    books.forget(playheads, book_id, _client_id(request))
    return {"ok": True}


@app.get("/api/books/{book_id}/contents")
def book_contents(book_id: str):
    """Chapters, read off what the narrator announces.

    An audiobook file carries no structure at all -- it is one opaque stream,
    which is why skipping around is guesswork. The transcript gives it back.
    """
    if book_id in BUILTINS:
        return {"contents": books.contents(BUILTINS[book_id])}
    rec = books.load(playheads, book_id)
    if not rec or rec.status != "ready":
        return {"contents": []}
    # AssemblyAI's own segmentation first: its headlines are written from the
    # content, so they work on the many recordings that never announce a
    # chapter out loud. Reading the narrator's announcements is the fallback.
    # Two or more, or it is not a table of contents. A short recording often
    # comes back as a single chapter, which is less navigable than even parts.
    stored = books.stored_chapters(playheads, book_id)
    if len(stored) >= 2:
        return {"contents": stored, "source": "auto_chapters"}
    return {"contents": books.contents(books.RedisLibrary(playheads, rec)),
            "source": "transcript"}


@app.get("/api/books/{book_id}")
def book_status(book_id: str):
    """Status, and one slice of work.

    Polling is the scheduler. Each call checks the transcript or embeds the
    next hundred chunks, so progress only moves while someone is watching --
    which is exactly when it matters.
    """
    if book_id in BUILTINS:
        return _builtin_card(book_id)
    rec = books.load(playheads, book_id)
    if not rec:
        raise HTTPException(404, "No such book.")
    rec = books.advance(playheads, rec, AAI_KEY, _embedder())
    return rec.public()


def _title_from_url(url: str) -> str:
    """A readable name out of a file name.

    Archive.org and LibriVox names carry encoding junk ('..._64kb.mp3') that
    would otherwise end up on screen as the title of someone's book.
    """
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    name = re.sub(r"\.(mp3|m4a|m4b|wav|ogg|flac|aac|webm)$", "", name, flags=re.I)
    # No \b before the digits: an underscore is a word character, so "_64kb"
    # has no boundary to anchor to and the junk survives.
    name = re.sub(r"[_\- ]*\d{1,3} ?k(?:b|bps)?(?![a-z0-9])", "", name, flags=re.I)
    name = re.sub(r"[_-]+", " ", name).strip()
    words = [w if (w.isupper() and len(w) > 1) else w.capitalize() for w in name.split()]
    return " ".join(words) or "Untitled book"


@app.get("/api/health")
def health():
    return {"ok": True, "book": BOOK, "books": list(BUILTINS), "chunks": len(library),
            "agent_configured": bool(AGENT_ID), "store": playheads.kind,
            "access_code": bool(ACCESS_CODE),
            "voice_provider": VOICE_PROVIDER,
            "local_models": local_voice.status(),
            "tts_voices": [{"id": k, "label": v} for k, v in TTS_VOICES.items()],
            "tools": [{"id": tid, "label": lbl, "description": desc, "always_on": always}
                      for tid, lbl, desc, always in TOOL_CATALOG],
            "tts_voice_default": DEFAULT_TTS_VOICE,
            "vad": "silero" if vad_enabled() else "energy",
            "local_llms": local_llm.available_llms()}


# ---------- static ----------

# Committed audio is in public/audio (that is also what Vercel serves); local-only
# titles stream from media/audio -- only that subfolder, so their indexes and
# transcripts are never served.
#
# Served by hand rather than StaticFiles: Cloud Run refuses any HTTP/1 response
# over 32 MiB, and the <audio> element's first request is "Range: bytes=0-",
# which StaticFiles answers with the whole file -- a 46 MB podcast came back as
# a 500. A 206 may return less than was asked for, so ranges are capped and the
# browser simply asks for the next piece; a request with no Range is streamed.
AUDIO = ROOT / "public" / "audio"
AUDIO_CHUNK = 4 * 1024 * 1024
_AUDIO_TYPES = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav",
                ".ogg": "audio/ogg"}
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


def _audio_response(folder: Path, name: str, request: Request):
    from fastapi.responses import Response, StreamingResponse
    path = (folder / name).resolve()
    kind = _AUDIO_TYPES.get(path.suffix.lower())
    if not kind or path.parent != folder.resolve() or not path.is_file():
        raise HTTPException(404)
    size = path.stat().st_size
    head = {"Accept-Ranges": "bytes", "Content-Type": kind,
            "Cache-Control": "public, max-age=86400"}
    m = _RANGE.fullmatch(request.headers.get("range", "").strip())
    if not m or not (m.group(1) or m.group(2)):
        if request.method == "HEAD":
            return Response(headers={**head, "Content-Length": str(size)})

        def chunks():
            with open(path, "rb") as f:
                while piece := f.read(256 * 1024):
                    yield piece
        return StreamingResponse(chunks(), headers=head, media_type=kind)
    if m.group(1):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
    else:                                     # "bytes=-N": the last N bytes
        start, end = max(0, size - int(m.group(2))), size - 1
    if start >= size or end < start:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    end = min(end, size - 1, start + AUDIO_CHUNK - 1)
    head.update({"Content-Range": f"bytes {start}-{end}/{size}",
                 "Content-Length": str(end - start + 1)})
    if request.method == "HEAD":
        return Response(status_code=206, headers=head)
    with open(path, "rb") as f:
        f.seek(start)
        body = f.read(end - start + 1)
    return Response(content=body, status_code=206, headers=head)


if AUDIO.exists():
    @app.api_route("/audio/{name}", methods=["GET", "HEAD"])
    def committed_audio(name: str, request: Request):
        return _audio_response(AUDIO, name, request)

if (MEDIA / "audio").exists():
    @app.api_route("/media/{name}", methods=["GET", "HEAD"])
    def local_media(name: str, request: Request):
        return _audio_response(MEDIA / "audio", name, request)

IMAGES = ROOT / "public" / "images"
if IMAGES.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES)), name="images")

if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(WEB / "index.html"))


if __name__ == "__main__":
    import uvicorn

    # Long enough that a slow local synthesis does not drop the socket.
    # The models run in worker threads, and PocketTTS on a modest CPU can hold
    # the GIL for longer than uvicorn's default 20 s ping budget, so the event
    # loop misses a keepalive and closes the connection mid-reply.
    uvicorn.run(app, host="127.0.0.1", port=8000,
                ws_ping_interval=30, ws_ping_timeout=180)
