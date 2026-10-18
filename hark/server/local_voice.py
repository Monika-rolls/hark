"""The all-local voice stack: Whistle (STT) + Needle (tool routing) + PocketTTS (TTS).

This is the counterpart to the AssemblyAI Voice Agent, which does STT, the LLM
and TTS in one websocket on someone else's servers. Here every model runs on the
machine the server is on, so the browser cannot talk to a vendor directly and
has to go through us -- see the `/ws/voice` endpoint in `hark/server/main.py`.

Three models, one process, each loaded once and reused:

  whistle  -- speech to text. 16.9 MB, CPU, 16 kHz mono, at most 30 s a call
              (so it is not a streaming model; the Endpointer below cuts turns).
  needle   -- the tool-calling model. It does NOT write the spoken answer --
              its docs are explicit that unsupported requests return an empty
              call. It does the routing: which of the three book tools to run and
              with what arguments. The answer prose comes from Gemini, which is
              the one piece the listener chose to keep in the cloud.
  pocket   -- text to speech, ~100M params on the CPU, English and friends.

Nothing here imports a heavy package at module load. The engine binary,
Whistle's `.cact` and PocketTTS's weights are all fetched on first use, and a
missing package becomes a clear error on `/api/health` rather than an
import-time crash that takes the whole server down.
"""
import os
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np

# The browser sends 24 kHz PCM16 (the rate the AssemblyAI path uses) and the
# sentence grammar above expects it, so keep the wire at 24 kHz and resample
# inside. Whistle wants 16 kHz mono float in [-1, 1].
WIRE_RATE = 24000
STT_RATE = 16000
# Whistle's hard ceiling is 30 s a call. Cut a little under it so a long
# question still transcribes rather than erroring.
MAX_UTTERANCE_S = 28.0
# Anything shorter than this is a cough, a key click, or a click from the
# speakers -- not a question. Transcribing it invites a hallucination.
MIN_UTTERANCE_S = 0.8
# Whisper-family models emit these on noise or near-silence. A short clip that
# comes back as one of them is the model filling in, not the listener talking.
HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thanks a lot", "thank you.",
    "thanks for watching", "thank you for watching", "thanks for watching!",
    "please subscribe", "subscribe", "you", "bye", "goodbye", "okay", "ok",
    "the", "a", "uh", "um", "hmm", "mm", "ah", "oh", "yeah", "you know",
    "subtitles by", "whisper", "music", "applause", "laughter",
}
# Fillers and acknowledgements. They are worth showing (the listener sees their
# words) but not worth answering: there is no question in them, and a small
# model will invent one, echoing the prompt back.
CONTENTLESS = {
    "uh", "uhh", "uh huh", "um", "umm", "hmm", "hm", "mm", "er", "ah", "oh",
    "yeah", "yep", "yes", "no", "nope", "ok", "okay", "right", "sure", "alright",
    "thanks", "thank you", "hello", "hi", "hey", "so", "well", "like", "i see",
}

# One model per process, and none of them is thread-safe. Whistle and Needle run
# on the same Cactus engine, so they share a lock; PocketTTS (torch) has its own.
# It used to be one lock for all three -- and since speaking holds it for a whole
# reply, one stuck reply meant no one could be heard again.
_MODEL_LOCK = threading.RLock()
_TTS_LOCK = threading.RLock()


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear resample of a mono signal. Speech-grade, no extra dependency."""
    if src == dst or len(x) == 0:
        return x.astype(np.float32)
    n = int(round(len(x) * dst / src))
    if n <= 1:
        return x.astype(np.float32)
    idx = np.linspace(0.0, len(x) - 1.0, n)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    clipped = np.clip(x, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _tensor_to_numpy(audio) -> np.ndarray:
    """PocketTTS returns torch tensors; this project speaks numpy."""
    try:
        return audio.detach().cpu().numpy().reshape(-1).astype(np.float32)
    except AttributeError:
        return np.asarray(audio, dtype=np.float32).reshape(-1)


class Endpointer:
    """Turn a stream of small audio frames into utterances.

    Whistle is not a streaming model, so something has to decide where a spoken
    question begins and ends. This is the same energy-VAD shape the retired
    desktop client used (`hark/core/mic.py`): an adaptive noise floor, a minimum
    trigger so a silent room cannot clear it, and a sustained run of hot audio
    before speech counts -- one click or a consonant burst from the speakers
    must not open a turn.

    Timing is in seconds, not frames, so the thresholds do not move if the
    browser's frame size changes, and both are settings: a shorter silence makes
    the answer come sooner, at the risk of cutting a thinking pause in half.

    Loud is not the same as speech: a fan, a TV or a door is loud too, and
    Whistle answers anything it is given with words. With a `vad` (a
    SpeechDetector), a frame only counts as hot when it is loud *and* voiced, and
    a voiceless stretch counts as quiet, so noise neither opens a turn nor holds
    one open. `last_voiced_s` says how much of the last turn was a voice.
    """

    def __init__(
        self,
        rate: int = WIRE_RATE,
        threshold_ratio: float = 4.0,
        min_trigger_rms: float = 0.01,
        attack_s: Optional[float] = None,
        silence_s: Optional[float] = None,
        max_seconds: float = MAX_UTTERANCE_S,
        vad: Optional["SpeechDetector"] = None,
    ):
        # 450 ms of silence ends the turn (was ~1 s, which added a wait to every
        # question). 150 ms of speech opens it. Both are settings.
        # The attack must be an UNBROKEN run of loud, voiced frames, and speech
        # dips between syllables: at 280 ms neither "Please quiz me." nor
        # "Wait, what did he mean by that last part?" nor narration at normal
        # level ever opened a turn. Noise is rejected by the trigger level,
        # Silero and the voiced-time floor, not by a long attack.
        if silence_s is None:
            silence_s = float(os.getenv("HARK_STT_SILENCE_MS", "450") or 450) / 1000.0
        if attack_s is None:
            attack_s = float(os.getenv("HARK_STT_ATTACK_MS", "150") or 150) / 1000.0
        self.rate = rate
        self._ratio = threshold_ratio
        self._min_trigger = min_trigger_rms
        self._attack_samples = max(1, int(round(attack_s * rate)))
        self._hangover_samples = max(1, int(round(silence_s * rate)))
        self._max_samples = int(max_seconds * rate)
        self._floor = 1e-4
        self._active_run = 0
        self._quiet_run = 0
        self._speaking = False
        self._frames: List[np.ndarray] = []
        self._tail: List[np.ndarray] = []
        self._gen = 0
        self._vad = vad
        self._voiced_run = 0
        self.last_voiced_s = 0.0

    @property
    def gated(self) -> bool:
        """True when a speech detector, not loudness alone, decides."""
        return self._vad is not None

    def _threshold(self) -> float:
        return max(self._floor * self._ratio, self._min_trigger)

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def generation(self) -> int:
        """Changes when a turn opens or closes, so a stale partial is dropped."""
        return self._gen

    def buffered(self) -> Optional[np.ndarray]:
        """The audio of the turn in progress, for a live partial transcript."""
        if not self._speaking or not self._frames:
            return None
        return np.concatenate(self._frames)

    def feed(self, frame: np.ndarray) -> Optional[Tuple[str, Optional[np.ndarray]]]:
        """Push one frame; return ("start", None) or ("end", audio) or None."""
        frame = frame.astype(np.float32)
        n = len(frame)
        rms = float(np.sqrt(np.mean(frame * frame))) if n else 0.0
        threshold = self._threshold()
        # Every frame goes to the detector, hot or not: it carries state from
        # one chunk to the next and scores worse on a stream with holes in it.
        voiced = self._vad.is_speech(frame, self._speaking) if self._vad else True
        hot = rms > threshold and voiced

        if not self._speaking:
            # Adapt the noise floor only while nobody is speaking, and only
            # upward toward room tone -- never toward a word.
            self._floor = 0.95 * self._floor + 0.05 * rms
            self._tail.append(frame)
            self._tail = self._tail[-4:]          # ~0.34 s of run-up
            self._active_run = self._active_run + n if hot else 0
            if self._active_run >= self._attack_samples:
                self._speaking = True
                self._gen += 1
                self._quiet_run = 0
                self._voiced_run = self._active_run
                self._frames = list(self._tail)
                return ("start", None)
            return None

        self._frames.append(frame)
        self._quiet_run = self._quiet_run + n if not hot else 0
        # Counted on the detector alone: the soft ends of words fall under the
        # loudness gate, and counting only loud *and* voiced frames dropped short
        # commands ("Please quiz me." measured 0.26 s of a 1.1 s turn).
        if voiced:
            self._voiced_run += n
        total = sum(len(f) for f in self._frames)
        if self._quiet_run >= self._hangover_samples or total >= self._max_samples:
            audio = np.concatenate(self._frames) if self._frames else np.zeros(0, np.float32)
            self.last_voiced_s = self._voiced_run / self.rate
            self._speaking = False
            self._gen += 1
            self._active_run = 0
            self._quiet_run = 0
            self._voiced_run = 0
            self._frames = []
            self._tail = []
            return ("end", audio)
        return None


class SpeechDetector:
    """Silero VAD: is this frame a voice, or only loud?

    Silero scores 32 ms chunks of 16 kHz audio for speech. Frames arrive at the
    wire rate in whatever size the browser sends, so they are resampled and
    buffered into whole chunks. Two thresholds, as Silero recommends: a chunk has
    to clear 0.5 to open a turn but only 0.35 to keep one open, so a soft word in
    the middle of a question does not end it.

    One per session -- the model keeps state across chunks, so two listeners
    cannot share one.
    """

    CHUNK = 512  # 32 ms at 16 kHz, the window Silero is trained on

    def __init__(self, on: float = 0.6, off: float = 0.4):
        from silero_vad import load_silero_vad
        self._model = load_silero_vad()
        self._on = on
        self._off = off
        self._buf = np.zeros(0, np.float32)
        self._voiced = False

    def is_speech(self, frame: np.ndarray, speaking: bool = False) -> bool:
        import torch
        self._buf = np.concatenate([self._buf, resample(frame, WIRE_RATE, STT_RATE)])
        best = None
        with torch.no_grad():
            while len(self._buf) >= self.CHUNK:
                chunk, self._buf = self._buf[:self.CHUNK], self._buf[self.CHUNK:]
                p = float(self._model(torch.from_numpy(chunk.copy()), STT_RATE).item())
                best = p if best is None else max(best, p)
        # A frame shorter than a chunk keeps the last verdict.
        if best is not None:
            self._voiced = best >= (self._off if speaking else self._on)
        return self._voiced


def vad_enabled() -> bool:
    """Silero gating is on unless HARK_STT_VAD=0, and the package is installed."""
    if os.getenv("HARK_STT_VAD", "1") == "0":
        return False
    try:
        import silero_vad  # noqa: F401
        return True
    except Exception:
        return False


def new_speech_detector() -> Optional[SpeechDetector]:
    """A detector for one session, or None to gate on loudness alone.

    A missing or broken silero-vad degrades to the energy gate with a note,
    rather than taking local mode down.
    """
    if not vad_enabled():
        return None
    try:
        return SpeechDetector()
    except Exception as exc:
        print(f"[local] silero vad failed to load, gating on loudness alone: {exc}")
        return None


class WhistleTranscriber:
    """Whistle, via cactus-needle. 16 kHz mono float in, text out."""

    def __init__(self, language: Optional[str] = None, weights: Optional[str] = None):
        self._language = language
        self._weights = weights

    def available(self) -> bool:
        try:
            import needle  # noqa: F401
            return True
        except Exception:
            return False

    def transcribe(self, audio16k: np.ndarray) -> str:
        import needle
        with _MODEL_LOCK:
            out = needle.transcribe(audio16k, language=self._language,
                                    weights=self._weights)
        return (out or {}).get("text", "").strip()

    def warm(self) -> bool:
        """Pay the first-call cost (engine load, weight fetch) before a question.

        Without this the first utterance of a session carries the whole load,
        on top of the transcription itself.
        """
        try:
            self.transcribe(np.zeros(int(STT_RATE * 0.3), np.float32))
            return True
        except Exception as exc:
            print(f"[local] whistle warm failed: {exc}")
            return False


class NeedleRouter:
    """Needle decides which book tool a spoken question wants.

    The three tools are the same ones the cloud agent exposes, so the rest of
    the backend does not care which provider is answering. Needle returns tool
    calls, never prose: an off-topic request is an empty `function_calls`, which
    we treat as "just answer about what they heard" -- the product's default.
    """

    # Mirrors hark/server/agent.json. Kept as raw schema because we only want the
    # call back, not Needle executing anything for us.
    TOOLS = [
        {
            "name": "passage_at_playhead",
            "description": (
                "Fetch the audiobook passage the listener has just heard, from "
                "their playback position. Use for any question about what was "
                "just said, especially when they say 'that', 'this', 'the last "
                "part' or 'the last bit' rather than naming a topic."
            ),
            # Needle is trained on imperatives and tends to refuse questions
            # (an empty call) because it is unsure which tool a question wants.
            # These triggers force the call for the phrasings this product
            # actually hears, so routing does not depend on the confidence score.
            "triggers": [
                r"\bwhat (did|does|do) (that|this|it|they|he|she)\b",
                r"\b(that|this) (last )?(part|bit|sentence|line|thing|phrase)\b",
                r"\bwhat did (he|she|they) (mean|say)\b",
                r"\bi('?m| am) lost\b",
                r"\bwhere am i\b",
                r"\bremind me\b",
                r"\bexplain (that|this)\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": (
                            "Optional concept to look for earlier in the book, "
                            "only when they name something they heard before."
                        ),
                    }
                },
                "required": [],
            },
        },
        {
            "name": "go_to_topic",
            "description": (
                "Move the audiobook to where a topic is discussed. Use when the "
                "listener wants to go somewhere rather than understand where "
                "they are: 'take me to the part about X', 'skip to X', 'where "
                "does he explain X'."
            ),
            "triggers": [r"\b(take me|jump|skip|go back|go)\b.*\b(to|about)\b",
                         r"\bwhere (does|do|is|are)\b.*\b(explain|discuss|talk)\b"],
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "What they want to hear about, as a phrase.",
                    }
                },
                "required": ["topic"],
            },
        },
        {
            "name": "book_outline",
            "description": (
                "What the whole book covers, sampled across its full length. Use "
                "when they ask what the book is about, what it covers, or for a "
                "summary of the whole thing."
            ),
            "triggers": [
                r"\bwhat('?s| is) (this|the) book about\b",
                r"\bwhat('?s| is) it about\b",
                r"\bbook about\b",
                r"\bsumm?ar(y|ise|ize)\b",
                r"\boverview\b",
                r"\bworth (listening|reading)\b",
                r"\bwhat does (this|the) book cover\b",
            ],
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "web_search",
            "description": (
                "Search the internet for things the book cannot answer: a general "
                "fact, a definition, the latest research, or whether something is "
                "still true today. Use when they say 'look up', 'search', 'online', "
                "'nowadays' or 'in general'."
            ),
            "triggers": [
                r"\b(look (it|that|this)? ?up|search|google)\b",
                r"\b(online|on the (web|internet))\b",
                r"\b(still true|nowadays|these days|today|latest|currently)\b",
                r"\bin general\b",
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, as a short web query.",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "recap",
            "description": (
                "Summarise the last few minutes they heard, in order. Use for "
                "'recap', 'catch me up', 'what did I miss', 'summarise the last "
                "five minutes'."
            ),
            "triggers": [r"\brecap\b", r"\bcatch me up\b", r"\bwhat did i (just )?miss\b",
                         r"\bsumm?ar(y|ise|ize) (the )?(last|past)\b"],
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {"type": "number",
                                "description": "How many minutes back, 1 to 10."}
                },
                "required": [],
            },
        },
        {
            "name": "explain_simply",
            "description": (
                "Re-explain what they just heard in plain words with an everyday "
                "analogy or example. Use for 'explain it simply', 'in plain "
                "English', 'like I'm five', 'give me an example'."
            ),
            "triggers": [r"\b(simpl(e|er|y)|easier|plain (english|words))\b",
                         r"\blike i'?m (5|five|12|twelve|a kid)\b", r"\beli5\b",
                         r"\b(an|for) example\b", r"\banalogy\b", r"\bdumb it down\b"],
            "parameters": {
                "type": "object",
                "properties": {
                    "style": {"type": "string",
                              "description": "'simple' (default) or 'example'."}
                },
                "required": [],
            },
        },
        {
            "name": "translate",
            "description": (
                "Explain what they just heard in another language. Use for 'say "
                "that in Hindi', 'explain in Japanese', 'translate'."
            ),
            "triggers": [r"\btranslate\b",
                         r"\b(in|into) (hindi|telugu|tamil|kannada|malayalam|marathi|"
                         r"bengali|gujarati|punjabi|urdu|japanese|korean|chinese|"
                         r"mandarin|thai|vietnamese|indonesian|bahasa|malay|tagalog|"
                         r"filipino|spanish|french|german|italian|portuguese|arabic)\b"],
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string",
                                 "description": "The language to answer in, e.g. Hindi."}
                },
                "required": ["language"],
            },
        },
        {
            "name": "quiz_me",
            "description": (
                "Ask them a question to check they understood what they heard. "
                "Use for 'quiz me', 'test me', 'ask me a question'."
            ),
            "triggers": [r"\bquiz\b", r"\btest me\b", r"\bask me (a )?question\b",
                         r"\bcheck (if|that) i understood\b"],
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "save_concept",
            "description": (
                "Save the idea they just heard to their notes. Use for 'remember "
                "this', 'save that', 'bookmark this', 'add it to my notes'."
            ),
            "triggers": [r"\b(remember|save|bookmark|note) (this|that|it)\b",
                         r"\badd (this|that|it) to (my )?notes\b"],
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string",
                                "description": "A short name for the idea, if they said one."}
                },
                "required": [],
            },
        },
    ]

    def __init__(self, weights: Optional[str] = None):
        self._weights = weights
        self._agent = None

    def available(self) -> bool:
        try:
            import needle  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        if self._agent is None:
            import needle
            self._agent = needle.Needle(tools=self.TOOLS, weights=self._weights)
        return self._agent

    def route(self, text: str) -> Tuple[str, dict]:
        """(tool name, arguments). Defaults to the passage tool on a refusal."""
        try:
            with _MODEL_LOCK:
                agent = self._load()
                # Each utterance is independent routing, not a conversation:
                # without this, the arguments of the previous question bleed
                # into the next one (a stale `search` from an earlier turn).
                agent.reset()
                result = agent.complete(text)
            calls = (result or {}).get("function_calls") or []
            if calls:
                name = calls[0].get("name") or "passage_at_playhead"
                args = calls[0].get("arguments") or {}
                return name, args
        except Exception as exc:
            print(f"[local] need routing failed, using passage: {exc}")
        return "passage_at_playhead", {}


# PocketTTS's built-in English voices the listener may pick from. An allowlist,
# not a passthrough: PocketTTS also accepts arbitrary http:// and hf:// URLs as
# a voice, and a query parameter must never be able to make the server fetch one.
TTS_VOICES = {
    "alba": "Alba", "anna": "Anna", "vera": "Vera", "mary": "Mary", "jane": "Jane",
    "eve": "Eve", "charles": "Charles", "george": "George", "paul": "Paul",
    "michael": "Michael", "jean": "Jean", "marius": "Marius",
}
DEFAULT_TTS_VOICE = "alba"


def tts_voice(name: Optional[str]) -> str:
    """The voice to use for a request: a known id, else the default."""
    name = (name or "").strip().lower()
    return name if name in TTS_VOICES else DEFAULT_TTS_VOICE


class PocketTtsSpeaker:
    """PocketTTS, kept in memory. Text in, 24 kHz PCM16 out for the browser."""

    def __init__(self, voice: str = DEFAULT_TTS_VOICE, language: str = "english"):
        self._voice = voice
        self._language = language
        self._model = None
        self._state = None
        self._states: dict = {}
        self._rate = WIRE_RATE

    def available(self) -> bool:
        try:
            import pocket_tts  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        if self._model is None:
            from pocket_tts import TTSModel
            # int8 dynamic quantisation: faster and smaller on CPU, with little
            # audible difference. HARK_TTS_QUANTIZE=0 turns it off.
            quantize = os.getenv("HARK_TTS_QUANTIZE", "1") != "0"
            try:
                model = TTSModel.load_model(language=self._language, quantize=quantize)
            except TypeError:
                # Older/newer signatures may not take these; the default model
                # is English, which matches the shipped voice.
                try:
                    model = TTSModel.load_model(language=self._language)
                except TypeError:
                    model = TTSModel.load_model()
            self._model = model
            self._rate = int(getattr(model, "sample_rate", WIRE_RATE) or WIRE_RATE)
            self._state = model.get_state_for_audio_prompt(self._voice)
            self._states[self._voice] = self._state
        return self._model

    def _voice_state(self, voice: Optional[str]):
        """The prompt state for a voice, computed once and kept (~a second each)."""
        self._load()
        voice = tts_voice(voice) if voice else self._voice
        if voice not in self._states:
            with _TTS_LOCK:
                self._states[voice] = self._model.get_state_for_audio_prompt(voice)
        return self._states[voice]

    def speak(self, text: str, voice: Optional[str] = None) -> bytes:
        """Synthesize one reply; return PCM16 at the browser's rate."""
        text = (text or "").strip()
        if not text:
            return b""
        model = self._load()
        state = self._voice_state(voice)
        with _TTS_LOCK:
            audio = model.generate_audio(state, text)
        samples = _tensor_to_numpy(audio)
        if self._rate != WIRE_RATE:
            samples = resample(samples, self._rate, WIRE_RATE)
        return float_to_pcm16(samples)

    def stream(self, text: str, voice: Optional[str] = None):
        """Yield PCM16 at the wire rate as PocketTTS produces it.

        Waiting for the whole reply to synthesize before sending the first byte
        was the largest single part of the delay. Streaming starts the audio in
        time-to-first-chunk (~200 ms) instead of the length of the clip.
        """
        text = (text or "").strip()
        if not text:
            return
        model = self._load()
        state = self._voice_state(voice)
        with _TTS_LOCK:
            for chunk in model.generate_audio_stream(state, text):
                samples = _tensor_to_numpy(chunk)
                if self._rate != WIRE_RATE:
                    samples = resample(samples, self._rate, WIRE_RATE)
                pcm = float_to_pcm16(samples)
                if pcm:
                    yield pcm


class LocalVoice:
    """The three models behind one object, for the /ws/voice endpoint."""

    def __init__(
        self,
        whistle: Optional[WhistleTranscriber] = None,
        router: Optional[NeedleRouter] = None,
        speaker: Optional[PocketTtsSpeaker] = None,
    ):
        self.stt = whistle or WhistleTranscriber()
        self.router = router or NeedleRouter()
        self.tts = speaker or PocketTtsSpeaker()

    def status(self) -> dict:
        return {
            "whistle": self.stt.available(),
            "needle": self.router.available(),
            "pocket_tts": self.tts.available(),
        }

    def warm(self) -> dict:
        """Load every model, so the first question is not the slow one.

        All three: the engine and weights download and initialise on first use,
        and paying that at session start is invisible next to paying it on the
        listener's first question.
        """
        try:
            self.stt.warm()
        except Exception as exc:
            print(f"[local] whistle warm failed: {exc}")
        try:
            self.router._load()
        except Exception as exc:
            print(f"[local] needle warm failed: {exc}")
        try:
            self.tts._load()
        except Exception as exc:
            print(f"[local] pocket-tts warm failed: {exc}")
        return self.status()
