"""The local voice path: endpointing, resampling, routing dispatch, the toggle.

None of this needs a model, a key or a microphone. The heavy models (Whistle,
Needle, PocketTTS) load lazily behind `server.local_voice`, so the pieces that
decide *when* to transcribe and *which* tool to run are testable on their own --
which is where the bugs that survive to production live.
"""
import numpy as np
import pytest

from hark.server import local_llm, main
from hark.server.local_voice import (Endpointer, float_to_pcm16, pcm16_to_float,
                                resample)

RATE = 24000
FRAME = 2048


def _feed(endpointer, frames):
    return [e[0] for e in (endpointer.feed(f) for f in frames) if e]


def _quiet(n):
    return [np.zeros(FRAME, np.float32) for _ in range(n)]


def _speech(n):
    return [np.full(FRAME, 0.05, np.float32) for _ in range(n)]


def test_a_quiet_room_opens_no_turn():
    assert _feed(Endpointer(RATE), _quiet(60)) == []


def test_a_sustained_burst_opens_then_closes_a_turn():
    # Three hot frames is the attack; a run of quiet past the hangover ends it.
    events = _feed(Endpointer(RATE), _quiet(20) + _speech(5) + _quiet(20))
    assert events == ["start", "end"]


def test_one_hot_frame_is_a_click_not_speech():
    # A chair, a keyboard tap, or a consonant burst from the speakers is one
    # frame; speech sustains. Below the attack run it must not open a turn.
    events = _feed(Endpointer(RATE), _quiet(10) + _speech(1) + _quiet(20))
    assert events == []


def test_the_utterance_carries_the_speech_it_heard():
    ep = Endpointer(RATE)
    captured = None
    for frame in _quiet(20) + _speech(5) + _quiet(20):
        event = ep.feed(frame)
        if event and event[0] == "end":
            captured = event[1]
    assert captured is not None and len(captured) > 0
    assert float(np.max(np.abs(captured))) == pytest.approx(0.05)


def test_the_endpointer_exposes_the_turn_for_a_live_partial():
    # The hook the live transcript needs: the audio so far, and a generation
    # that changes when the turn does, so a stale partial is not shown.
    ep = Endpointer(RATE)
    assert ep.buffered() is None
    before = ep.generation
    for _ in range(10):
        ep.feed(np.zeros(FRAME, np.float32))
    started = None
    for _ in range(4):
        started = ep.feed(np.full(FRAME, 0.05, np.float32))
        if started:
            break
    assert started and started[0] == "start"
    assert ep.speaking and ep.generation != before
    buffered = ep.buffered()
    assert buffered is not None and len(buffered) > 0
    ended = None
    for _ in range(40):
        ended = ep.feed(np.zeros(FRAME, np.float32))
        if ended:
            break
    assert ended and ended[0] == "end"
    assert not ep.speaking and ep.buffered() is None



class _Verdict:
    """A stand-in SpeechDetector that says what it is told."""

    def __init__(self, voiced):
        self.voiced = voiced

    def is_speech(self, frame, speaking=False):
        return self.voiced


def test_loud_noise_that_is_not_a_voice_opens_no_turn():
    # A fan or a door clears the energy gate; the detector says no voice.
    events = _feed(Endpointer(RATE, vad=_Verdict(False)), _quiet(20) + _speech(30) + _quiet(20))
    assert events == []


def test_a_voice_opens_and_noise_closes_the_turn():
    vad = _Verdict(True)
    ep = Endpointer(RATE, vad=vad)
    events = _feed(ep, _quiet(20) + _speech(5))
    assert events == ["start"]
    # Still loud, but no longer a voice: that is the end of the question, not
    # a 28-second turn held open by the fan.
    vad.voiced = False
    assert _feed(ep, _speech(10)) == ["end"]
    assert ep.gated
    assert ep.last_voiced_s == pytest.approx(5 * FRAME / RATE)


def _silero():
    pytest.importorskip("silero_vad")
    from hark.server.local_voice import SpeechDetector
    return SpeechDetector()


def test_silero_hears_no_voice_in_loud_noise():
    rng = np.random.default_rng(0)
    noise = [(rng.standard_normal(FRAME) * 0.1).astype(np.float32) for _ in range(40)]
    ep = Endpointer(RATE, vad=_silero())
    assert _feed(ep, _quiet(20) + noise + _quiet(20)) == []


def test_silero_hears_the_narrator():
    sf = pytest.importorskip("soundfile")
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "public" / "audio" / "calculus.mp3"
    try:
        audio, rate = sf.read(str(path), frames=12 * 44100, dtype="float32", always_2d=True)
    except Exception as exc:
        pytest.skip(f"cannot decode the demo book here: {exc}")
    voice = resample(audio.mean(axis=1), rate, RATE)[RATE * 2:RATE * 10]
    frames = [voice[i:i + FRAME] for i in range(0, len(voice) - FRAME, FRAME)]
    ep = Endpointer(RATE, vad=_silero())
    assert "start" in _feed(ep, _quiet(20) + frames + _quiet(20))

def test_resample_moves_24k_to_16k():
    x = np.ones(RATE, np.float32)
    y = resample(x, RATE, 16000)
    assert len(y) == 16000
    assert np.allclose(y, 1.0, atol=1e-3)


def test_resample_is_identity_at_the_same_rate():
    x = np.linspace(-1, 1, 1234).astype(np.float32)
    assert np.array_equal(resample(x, 16000, 16000), x)


def test_pcm16_round_trips():
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0], np.float32)
    back = pcm16_to_float(float_to_pcm16(x))
    assert np.allclose(back, x, atol=1e-4)


def test_run_local_tool_dispatches_by_name(monkeypatch):
    seen = {}

    def passage(sid, search=None):
        seen.update(kind="passage", sid=sid, search=search)
        return {"ok": True}

    def topic(sid, name):
        seen.update(kind="topic", sid=sid, topic=name)
        return {"ok": True}

    def outline(sid):
        seen.update(kind="outline", sid=sid)
        return {"ok": True}

    monkeypatch.setattr(main, "tool_passage", passage)
    monkeypatch.setattr(main, "tool_go_to_topic", topic)
    monkeypatch.setattr(main, "tool_outline", outline)

    assert main.run_local_tool("s1", "go_to_topic", {"topic": "induction"})["ok"]
    assert seen == {"kind": "topic", "sid": "s1", "topic": "induction"}

    seen.clear()
    assert main.run_local_tool("s1", "book_outline", {})["ok"]
    assert seen == {"kind": "outline", "sid": "s1"}

    # Anything else, including an empty call from Needle, is the passage tool.
    seen.clear()
    assert main.run_local_tool("s1", "passage_at_playhead", {"search": "entropy"})["ok"]
    assert seen == {"kind": "passage", "sid": "s1", "search": "entropy"}

    seen.clear()
    assert main.run_local_tool("s1", "passage_at_playhead", {})["ok"]
    assert seen == {"kind": "passage", "sid": "s1", "search": None}


def test_audio_frames_are_browser_sized():
    pcm = bytes(RATE * 2)                 # one second of silence at 24 kHz
    frames = list(main._audio_frames(pcm))
    # 200 ms a frame, and the last one carries the remainder.
    assert len(frames) == 5
    assert all(len(f) == 9600 for f in frames)


pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def test_a_local_session_needs_no_key_and_no_access_code(monkeypatch):
    monkeypatch.setattr(main, "ACCESS_CODE", "JUDGES-ONLY")
    client = TestClient(main.app)
    r = client.get("/api/session?provider=local")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["provider"] == "local"
    assert data["ws_path"] == "/ws/voice"
    # Nothing vendor-side is handed out for a local session.
    assert "token" not in data and "agent_id" not in data
    assert set(data["models"]) == {"whistle", "needle", "pocket_tts"}


def test_health_reports_the_provider_and_the_local_models():
    client = TestClient(main.app)
    h = client.get("/api/health").json()
    assert h["voice_provider"] in ("assemblyai", "local")
    assert set(h["local_models"]) == {"whistle", "needle", "pocket_tts"}


def test_the_local_websocket_speaks_the_assemblyai_wire_protocol(monkeypatch):
    """The whole local loop, with the models faked.

    The point is the seam: the browser already understands one set of event
    names, so the local socket has to emit those and only those. If this drifts,
    the client stops reacting and the mode looks dead.
    """
    import base64

    # Partials would race the single-turn assertion below; covered separately.
    monkeypatch.setenv("HARK_STT_PARTIALS", "0")
    # A flat tone is loud but not a voice; this covers the protocol, not the VAD.
    monkeypatch.setenv("HARK_STT_VAD", "0")
    monkeypatch.setattr(main.local_voice.stt, "transcribe",
                        lambda audio: "what did that last sentence mean?")
    monkeypatch.setattr(main.local_voice.router, "route",
                        lambda text: ("passage_at_playhead", {}))
    monkeypatch.setattr(main, "run_local_tool",
                        lambda sid, name, args: {"ok": True, "message": "the passage"})
    monkeypatch.setattr(main, "local_answer_stream",
                        lambda q, m, llm=None: iter(["It means the argument holds."]))
    monkeypatch.setattr(main.local_voice.tts, "stream",
                        lambda text, voice=None: iter([b"\x00\x01" * 4800]))

    def frame(value):
        return base64.b64encode(np.full(FRAME, value, np.int16).tobytes()).decode()

    client = TestClient(main.app)
    with client.websocket_connect("/ws/voice?session_id=sX") as ws:
        assert ws.receive_json()["type"] == "session.ready"
        for value, count in ((0, 20), (1600, 5), (0, 20)):
            for _ in range(count):
                ws.send_json({"type": "input.audio", "audio": frame(value)})
        seen = [ws.receive_json()["type"] for _ in range(8)]

    assert seen == [
        "input.speech.started",     # the endpointer opened the turn
        "input.speech.stopped",     # and closed it
        "transcript.user",          # Whistle's text
        "tool.call",                # which book tool the agent chose
        "reply.started",
        "transcript.agent",         # the answer, shown on screen
        "reply.audio",              # and spoken
        "reply.done",
    ], seen


def test_the_answer_model_choice_defaults_and_selects(monkeypatch):
    monkeypatch.delenv("HARK_LOCAL_LLM", raising=False)
    assert local_llm.default_llm() == "gemini"
    monkeypatch.setenv("HARK_LOCAL_LLM", "gemma")
    assert local_llm.default_llm() == "gemma"
    # The backend was Qwen before; old settings still reach the same server.
    monkeypatch.setenv("HARK_LOCAL_LLM", "qwen")
    assert local_llm.default_llm() == "gemma"
    monkeypatch.setenv("HARK_LOCAL_LLM", "nonsense")
    assert local_llm.default_llm() == "gemini"
    # Gemma builds even when the probe fails (a slow server should not silently
    # downgrade the answer); Gemini needs a key.
    assert local_llm.build_answer_model("gemma").name == "gemma"
    assert local_llm.build_answer_model("qwen").name == "gemma"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert local_llm.build_answer_model("gemini") is None


def test_a_cloud_tool_call_rides_the_heartbeat_once():
    main._note_tool("sTool", "go_to_topic", {"topic": "the treasure"})
    client = TestClient(main.app)
    first = client.post("/api/playhead", json={"session_id": "sTool", "seconds": 1.0}).json()
    assert first.get("tool") == {"name": "go_to_topic",
                                 "arguments": {"topic": "the treasure"}}
    # Claimed once: the page is not told about the same call on every heartbeat.
    again = client.post("/api/playhead", json={"session_id": "sTool", "seconds": 2.0}).json()
    assert "tool" not in again


def test_the_fallback_reads_the_last_sentences_not_a_mid_word_slice():
    message = ("ANSWER LENGTH: two or three sentences.\n\nThe listener is 1 minute in.\n"
               "This is what they have just heard (the last 90 seconds):\n"
               "A first sentence about the argument. A second one that explains the "
               "point. The final sentence, which is the newest and the one 'that' means.")
    out = main._fallback_answer({"message": message})
    assert out.startswith("From the book:")
    assert "final sentence" in out
    assert not out.split("From the book: ", 1)[1].startswith(("irst", "econd", "he "))


def test_an_empty_answer_still_speaks_the_passage(monkeypatch):
    # When the answer model is down it returns nothing -- which is not an
    # exception. The reply must still be spoken, or the page shows words and
    # plays silence, which is exactly the bug this covers.
    import base64

    monkeypatch.setenv("HARK_STT_PARTIALS", "0")
    # A flat tone is loud but not a voice; this covers the protocol, not the VAD.
    monkeypatch.setenv("HARK_STT_VAD", "0")
    monkeypatch.setattr(main.local_voice.stt, "transcribe", lambda audio: "what did that mean?")
    monkeypatch.setattr(main.local_voice.router, "route",
                        lambda text: ("passage_at_playhead", {}))
    monkeypatch.setattr(
        main, "run_local_tool",
        lambda sid, name, args: {"ok": True, "message": (
            "The listener is 1 minute in.\nThis is what they have just heard "
            "(the last 90 seconds):\nA full sentence about the argument.")})
    monkeypatch.setattr(main, "local_answer_stream", lambda q, m, llm=None: iter(()))

    spoken = {}

    def fake_speak(text, voice=None):
        spoken["text"] = text
        return b"\x00\x01" * 100

    monkeypatch.setattr(main.local_voice.tts, "speak", fake_speak)

    def frame(value):
        return base64.b64encode(np.full(FRAME, value, np.int16).tobytes()).decode()

    client = TestClient(main.app)
    with client.websocket_connect("/ws/voice?session_id=sEmpty") as ws:
        assert ws.receive_json()["type"] == "session.ready"
        for value, count in ((0, 20), (1600, 5), (0, 20)):
            for _ in range(count):
                ws.send_json({"type": "input.audio", "audio": frame(value)})
        seen = [ws.receive_json()["type"] for _ in range(8)]

    assert seen == [
        "input.speech.started", "input.speech.stopped", "transcript.user",
        "tool.call", "reply.started", "reply.audio", "transcript.agent", "reply.done",
    ], seen
    assert spoken["text"].startswith("From the book:")



def test_a_voice_is_allowlisted_never_fetched():
    # PocketTTS takes URLs as voices too; a query parameter must not reach it.
    from hark.server.local_voice import DEFAULT_TTS_VOICE, TTS_VOICES, tts_voice
    assert tts_voice("George") == "george" and "george" in TTS_VOICES
    for bad in ("https://evil.example/x.wav", "hf://someone/voice.wav", "", None, "../alba"):
        assert tts_voice(bad) == DEFAULT_TTS_VOICE


def test_health_lists_the_voices():
    h = TestClient(main.app).get("/api/health").json()
    ids = [v["id"] for v in h["tts_voices"]]
    assert h["tts_voice_default"] in ids and "george" in ids


def test_the_chosen_voice_reaches_the_speaker(monkeypatch):
    import base64

    monkeypatch.setenv("HARK_STT_PARTIALS", "0")
    monkeypatch.setenv("HARK_STT_VAD", "0")
    monkeypatch.setattr(main.local_voice.stt, "transcribe", lambda audio: "what did that mean?")
    monkeypatch.setattr(main.local_voice.router, "route", lambda text: ("passage_at_playhead", {}))
    monkeypatch.setattr(main, "run_local_tool", lambda sid, name, args: {"ok": True, "message": "p"})
    monkeypatch.setattr(main, "local_answer_stream", lambda q, m, llm=None: iter(["An answer."]))
    voices = []

    def fake_stream(text, voice=None):
        voices.append(voice)
        return iter([b"\x00\x01" * 100])

    monkeypatch.setattr(main.local_voice.tts, "stream", fake_stream)

    def frame(value):
        return base64.b64encode(np.full(FRAME, value, np.int16).tobytes()).decode()

    with TestClient(main.app).websocket_connect("/ws/voice?session_id=sV&voice=george") as ws:
        assert ws.receive_json()["voice"] == "george"
        for value, count in ((0, 20), (1600, 5), (0, 20)):
            for _ in range(count):
                ws.send_json({"type": "input.audio", "audio": frame(value)})
        for _ in range(8):
            if ws.receive_json()["type"] == "reply.done":
                break
    assert voices == ["george"]
