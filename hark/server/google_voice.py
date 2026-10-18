"""The all-Google voice stack: Speech-to-Text (Chirp) + Gemini + Text-to-Speech.

The third engine beside the AssemblyAI agent and the on-machine models. Every
model here is Google's, and all of them authenticate the same way -- the Cloud
Run service account (or `gcloud auth application-default login` locally) -- so
there is no key in the container:

  stt     -- Cloud Speech-to-Text v2 with Chirp, falling back to the v1
             `latest_short` model if Chirp is unavailable in the region.
  router  -- Gemini function calling over the same tool schemas Needle sees,
             forced to pick exactly one tool.
  tts     -- Cloud Text-to-Speech Chirp 3 HD voices. The locale follows the
             script of the answer, so "explain that in Hindi" is spoken by a
             Hindi voice rather than an English one reading Devanagari.

The objects mirror LocalVoice's (`stt.transcribe`, `router.route`,
`tts.speak` / `tts.stream`), so the /ws/voice loop runs unchanged on either.
Nothing heavy is imported at module load.
"""
import os
import threading
import unicodedata
from typing import Iterator, Optional, Tuple

import numpy as np

from .local_voice import STT_RATE, WIRE_RATE, NeedleRouter, float_to_pcm16

# Chirp 3 HD voices the listener may pick, by name. An allowlist like the local
# voices: the name is spliced into a voice id, never passed through raw.
GOOGLE_TTS_VOICES = ["Aoede", "Kore", "Leda", "Zephyr", "Charon", "Fenrir", "Orus", "Puck"]
DEFAULT_GOOGLE_VOICE = "Aoede"

# Script of the answer -> the locale whose Chirp 3 HD voice should speak it.
_SCRIPT_LOCALE = [
    ("DEVANAGARI", "hi-IN"), ("TAMIL", "ta-IN"), ("TELUGU", "te-IN"),
    ("KANNADA", "kn-IN"), ("MALAYALAM", "ml-IN"), ("BENGALI", "bn-IN"),
    ("GUJARATI", "gu-IN"), ("GURMUKHI", "pa-IN"), ("HIRAGANA", "ja-JP"),
    ("KATAKANA", "ja-JP"), ("HANGUL", "ko-KR"), ("CJK", "cmn-CN"),
    ("THAI", "th-TH"), ("ARABIC", "ar-XA"),
]


def google_voice(name: Optional[str]) -> str:
    """A known Chirp 3 HD voice name (case-insensitive), else the default."""
    by_lower = {v.lower(): v for v in GOOGLE_TTS_VOICES}
    return by_lower.get((name or "").strip().lower(), DEFAULT_GOOGLE_VOICE)


def locale_for(text: str) -> str:
    """The locale for the dominant non-Latin script in `text`, else en-US."""
    counts: dict = {}
    for ch in text:
        if ch.isalpha() and ord(ch) > 0x2FF:
            name = unicodedata.name(ch, "")
            for script, locale in _SCRIPT_LOCALE:
                if name.startswith(script):
                    counts[locale] = counts.get(locale, 0) + 1
                    break
    letters = sum(1 for ch in text if ch.isalpha()) or 1
    if counts:
        locale, n = max(counts.items(), key=lambda kv: kv[1])
        if n / letters > 0.3:
            return locale
    return "en-US"


def _project() -> str:
    return os.getenv("HARK_VERTEX_PROJECT", "").strip()


class GoogleTranscriber:
    """Cloud Speech-to-Text. 16 kHz mono float in, text out."""

    def __init__(self):
        self._v2 = None
        self._v1 = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        try:
            from google.cloud import speech_v2  # noqa: F401
            return bool(_project())
        except Exception:
            return False

    def _recognize_v2(self, pcm: bytes) -> str:
        from google.api_core.client_options import ClientOptions
        from google.cloud import speech_v2
        from google.cloud.speech_v2 import types
        location = os.getenv("HARK_GOOGLE_STT_LOCATION", "asia-southeast1")
        with self._lock:
            if self._v2 is None:
                self._v2 = speech_v2.SpeechClient(client_options=ClientOptions(
                    api_endpoint=f"{location}-speech.googleapis.com"))
        config = types.RecognitionConfig(
            explicit_decoding_config=types.ExplicitDecodingConfig(
                encoding=types.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=STT_RATE, audio_channel_count=1),
            language_codes=[os.getenv("HARK_GOOGLE_STT_LANGUAGE", "en-US")],
            model=os.getenv("HARK_GOOGLE_STT_MODEL", "chirp_2"),
            features=types.RecognitionFeatures(enable_automatic_punctuation=True))
        resp = self._v2.recognize(request=types.RecognizeRequest(
            recognizer=f"projects/{_project()}/locations/{location}/recognizers/_",
            config=config, content=pcm), timeout=30)
        return " ".join(r.alternatives[0].transcript for r in resp.results
                        if r.alternatives).strip()

    def _recognize_v1(self, pcm: bytes) -> str:
        from google.cloud import speech
        with self._lock:
            if self._v1 is None:
                self._v1 = speech.SpeechClient()
        resp = self._v1.recognize(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=STT_RATE, language_code="en-US",
                model="latest_short", enable_automatic_punctuation=True),
            audio=speech.RecognitionAudio(content=pcm), timeout=30)
        return " ".join(r.alternatives[0].transcript for r in resp.results
                        if r.alternatives).strip()

    def transcribe(self, audio16k: np.ndarray) -> str:
        pcm = float_to_pcm16(audio16k)
        try:
            return self._recognize_v2(pcm)
        except Exception as exc:
            print(f"[google] chirp failed, using latest_short: {exc}")
            return self._recognize_v1(pcm)


class GeminiRouter:
    """Gemini picks the tool: the Needle schemas, as function declarations."""

    def available(self) -> bool:
        from . import local_llm
        return local_llm.vertex_client() is not None

    def route(self, text: str) -> Tuple[str, dict]:
        from google.genai import types
        from hark.core.brain import CHAT_MODEL
        from . import local_llm
        decls = [types.FunctionDeclaration(
            name=t["name"], description=t["description"], parameters=t["parameters"])
            for t in NeedleRouter.TOOLS]
        try:
            resp = local_llm.vertex_client().models.generate_content(
                model=os.getenv("HARK_LOCAL_CHAT_MODEL", "") or CHAT_MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "The listener is hearing an audiobook or podcast and just "
                        "spoke. Pick the one tool that serves what they said. "
                        "Questions about what was just said, 'that', 'this', or "
                        "'I'm lost' use passage_at_playhead."),
                    tools=[types.Tool(function_declarations=decls)],
                    tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                        mode="ANY")),
                    temperature=0.0))
            for part in resp.candidates[0].content.parts:
                call = getattr(part, "function_call", None)
                if call and call.name:
                    return call.name, dict(call.args or {})
        except Exception as exc:
            print(f"[google] routing failed, using passage: {exc}")
        return "passage_at_playhead", {}


class GoogleTts:
    """Cloud Text-to-Speech, Chirp 3 HD. Text in, 24 kHz PCM16 out."""

    def __init__(self):
        self._client = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        try:
            from google.cloud import texttospeech  # noqa: F401
            return bool(_project())
        except Exception:
            return False

    def speak(self, text: str, voice: Optional[str] = None) -> bytes:
        text = (text or "").strip()
        if not text:
            return b""
        from google.cloud import texttospeech as tts
        with self._lock:
            if self._client is None:
                self._client = tts.TextToSpeechClient()
        locale = locale_for(text)
        resp = self._client.synthesize_speech(
            input=tts.SynthesisInput(text=text),
            voice=tts.VoiceSelectionParams(
                language_code=locale, name=f"{locale}-Chirp3-HD-{google_voice(voice)}"),
            audio_config=tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.LINEAR16, sample_rate_hertz=WIRE_RATE),
            timeout=30)
        audio = resp.audio_content
        # LINEAR16 comes back as a WAV file; the wire wants the bare samples.
        return audio[44:] if audio[:4] == b"RIFF" else audio

    def stream(self, text: str, voice: Optional[str] = None) -> Iterator[bytes]:
        """One sentence at a time already; yield it in ~200 ms pieces."""
        pcm = self.speak(text, voice)
        step = int(WIRE_RATE * 0.2) * 2
        for i in range(0, len(pcm), step):
            yield pcm[i:i + step]


class GoogleVoice:
    """The three Google models behind LocalVoice's interface."""

    def __init__(self):
        self.stt = GoogleTranscriber()
        self.router = GeminiRouter()
        self.tts = GoogleTts()

    def status(self) -> dict:
        return {"stt": self.stt.available(), "router": self.router.available(),
                "tts": self.tts.available()}

    def available(self) -> bool:
        return all(self.status().values())
