"""Tools the listener can switch off in Settings.

The passage tool is the product and cannot be switched off. Any other tool,
once off, is refused at its endpoint (the cloud agent hears why) and, locally,
the question is answered from the passage instead.
"""
import base64

import numpy as np
from fastapi.testclient import TestClient

from hark.server import main

FRAME = 2048


def _beat(client, sid, **extra):
    r = client.post("/api/playhead", json={"session_id": sid, "seconds": 600,
                                           "book": "calculus", **extra})
    assert r.status_code == 200


def test_health_lists_the_tools_and_the_one_that_stays_on():
    tools = TestClient(main.app).get("/api/health").json()["tools"]
    by_id = {t["id"]: t for t in tools}
    assert by_id["passage_at_playhead"]["always_on"] is True
    assert {"web_search", "quiz_me", "save_concept", "translate"} <= set(by_id)
    assert all(t["label"] and t["description"] for t in tools)


def test_the_heartbeat_sets_them_and_ignores_what_cannot_be_off():
    client = TestClient(main.app)
    _beat(client, "sT1", tools_off=["web_search", "passage_at_playhead", "made_up"])
    assert main._tools_off("sT1") == {"web_search"}
    _beat(client, "sT1", tools_off=[])
    assert main._tools_off("sT1") == set()


def test_a_switched_off_tool_is_refused_for_the_cloud_agent(monkeypatch):
    monkeypatch.setattr(main.local_llm, "google_search",
                        lambda q, context="": {"text": "found", "sources": []})
    client = TestClient(main.app)
    _beat(client, "sT2", tools_off=["web_search"])
    off = client.post("/tools/web_search?session_id=sT2", json={"query": "x"}).json()
    assert off["ok"] is False and "Settings" in off["message"]
    _beat(client, "sT2", tools_off=[])
    assert client.post("/tools/web_search?session_id=sT2", json={"query": "x"}).json()["ok"]


def test_locally_a_switched_off_choice_answers_from_the_passage(monkeypatch):
    monkeypatch.setenv("HARK_STT_PARTIALS", "0")
    monkeypatch.setenv("HARK_STT_VAD", "0")
    monkeypatch.setattr(main.local_voice.stt, "transcribe", lambda audio: "search the web for it")
    monkeypatch.setattr(main.local_voice.router, "route", lambda text: ("web_search", {"query": "it"}))
    calls = []
    monkeypatch.setattr(main, "run_local_tool",
                        lambda sid, name, args: calls.append(name) or {"ok": True, "message": "m"})
    monkeypatch.setattr(main, "local_answer_stream", lambda q, m, llm=None: iter(["An answer."]))
    monkeypatch.setattr(main.local_voice.tts, "stream", lambda text, voice=None: iter([b"\x00\x01"]))

    def frame(value):
        return base64.b64encode(np.full(FRAME, value, np.int16).tobytes()).decode()

    url = "/ws/voice?session_id=sT3&tools_off=web_search,quiz_me"
    with TestClient(main.app).websocket_connect(url) as ws:
        ws.receive_json()
        for value, count in ((0, 20), (1600, 5), (0, 20)):
            for _ in range(count):
                ws.send_json({"type": "input.audio", "audio": frame(value)})
        while ws.receive_json()["type"] != "reply.done":
            pass
    assert calls == ["passage_at_playhead"]
    assert main._tools_off("sT3") == {"web_search", "quiz_me"}
