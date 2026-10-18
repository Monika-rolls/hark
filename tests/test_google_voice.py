"""The all-Google engine: Speech-to-Text, Gemini routing, Text-to-Speech.

The Google calls are faked; what is tested is the plumbing they plug into --
the same voice loop as the local engine, on its own socket path -- plus the two
pure pieces: the voice allowlist and choosing the TTS locale from the script.
"""
import base64

import numpy as np
from fastapi.testclient import TestClient

from hark.server import main
from hark.server.google_voice import DEFAULT_GOOGLE_VOICE, google_voice, locale_for

FRAME = 2048


def test_voices_are_allowlisted():
    assert google_voice("kore") == "Kore"
    for bad in ("", None, "en-US-Chirp3-HD-Kore", "../x", "https://evil/voice"):
        assert google_voice(bad) == DEFAULT_GOOGLE_VOICE


def test_the_answer_script_picks_the_voice_locale():
    assert locale_for("He means the ratio only makes sense for a function.") == "en-US"
    assert locale_for("यहाँ लेखक समझा रहे हैं कि dy over dx का मतलब") == "hi-IN"
    assert locale_for("これは関数の話です。") == "ja-JP"
    assert locale_for("இது ஒரு சார்பு") == "ta-IN"
    # A stray non-Latin name in an English answer stays English.
    assert locale_for("Robert Greene wrote about mastery; 名前 aside.") == "en-US"


def test_a_google_session_points_at_its_own_socket():
    data = TestClient(main.app).get("/api/session?provider=google").json()
    assert data["provider"] == "google" and data["ws_path"] == "/ws/voice/google"
    assert data["llm"] == "gemini"


def test_health_describes_the_google_engine():
    h = TestClient(main.app).get("/api/health").json()
    assert "available" in h["google_voice"]
    assert h["google_tts_voice_default"] in [v["id"] for v in h["google_tts_voices"]]


def test_the_google_socket_runs_the_same_loop_on_google_models(monkeypatch):
    monkeypatch.setenv("HARK_STT_VAD", "0")
    used = {}
    monkeypatch.setattr(main.google_engine.stt, "transcribe",
                        lambda audio: used.setdefault("stt", "Please quiz me."))
    monkeypatch.setattr(main.google_engine.router, "route",
                        lambda text: used.setdefault("route", ("quiz_me", {})))

    def fake_stream(text, voice=None):
        used["voice"] = voice
        return iter([b"\x00\x01" * 100])

    monkeypatch.setattr(main.google_engine.tts, "stream", fake_stream)
    monkeypatch.setattr(main, "run_local_tool", lambda sid, name, args: {"ok": True, "message": "m"})
    monkeypatch.setattr(main, "local_answer_stream",
                        lambda q, m, llm=None: used.setdefault("llm", llm) and iter(["A question?"]))

    def frame(value):
        return base64.b64encode(np.full(FRAME, value, np.int16).tobytes()).decode()

    with TestClient(main.app).websocket_connect("/ws/voice/google?session_id=sG&voice=kore&llm=gemma") as ws:
        ready = ws.receive_json()
        assert ready["engine"] == "google" and ready["voice"] == "Kore"
        for value, count in ((0, 20), (1600, 5), (0, 20)):
            for _ in range(count):
                ws.send_json({"type": "input.audio", "audio": frame(value)})
        seen = []
        while not seen or seen[-1] != "reply.done":
            seen.append(ws.receive_json()["type"])

    assert used["stt"] == "Please quiz me." and used["route"] == ("quiz_me", {})
    assert used["voice"] == "Kore" and used["llm"] == "gemini"    # Gemma is local-only
    assert "transcript.user.delta" not in seen                     # no billed partials
    assert "tool.call" in seen and "reply.audio" in seen
