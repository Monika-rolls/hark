"""Latency of every model in the local stack, and of the answer model it can use.

Measures, on this machine:

  whistle  speech to text, at three clip lengths -- cold, then warm
  needle   tool routing, per query -- cold, then warm
  gemini   the default answer model: time to first sentence, and total
  gemma    Google's Gemma on a local llama.cpp server, if one is serving (see below)
  pocket   text to speech: full synthesis, and time to the first audio chunk

Run it:

    python scripts/bench_local_models.py --runs 3

To include Gemma, start a llama.cpp server first (it serves on
http://127.0.0.1:8081/v1, where the app looks by default):

    .\scripts\start_local_llm.ps1
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # GEMINI_API_KEY etc., so the answer-model options are real

from hark.server import local_llm  # noqa: E402
from hark.server.local_voice import (PocketTtsSpeaker, STT_RATE, WhistleTranscriber,  # noqa: E402
                                NeedleRouter, resample)

QUERIES = [
    "what did that last sentence mean?",
    "I'm lost, recap where I am",
    "take me to the part about the Lorentz transformation",
    "where does he explain induction",
    "what is this book about?",
]
TTS_SHORT = "It means the argument holds under the assumptions just stated."
TTS_LONG = ("The passage is contrasting two views of the same idea. "
            "It says the first is a habit of thought rather than a fact, "
            "and the second follows from it only if you accept the premise.")


def load_speech(seconds: float) -> np.ndarray:
    """A real speech clip of the given length, at Whistle's 16 kHz."""
    import soundfile as sf
    src = ROOT / "podcast.mp3"
    if not src.exists():
        src = ROOT / "public" / "audio" / "calculus.mp3"
    rate = int(sf.info(str(src)).samplerate)
    data, rate = sf.read(str(src), dtype="float32", always_2d=True,
                         frames=int(rate * (5 + seconds)))
    seg = data[int(rate * 5):, 0][: int(rate * seconds)]
    return resample(seg, int(rate), STT_RATE)


def stats(times):
    warm = times[1:] or times
    return times[0], statistics.median(warm), min(warm)


def clock(fn, runs: int):
    times, out = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return times, out


def bench_stt(runs: int):
    print("\nWHISTLE -- speech to text (Cactus needle engine)")
    print("  clip      cold     warm(med)   xRT")
    stt = WhistleTranscriber()
    for seconds in (2.0, 5.0, 12.0):
        clip = load_speech(seconds)
        if not len(clip):
            continue
        times, _ = clock(lambda: stt.transcribe(clip), runs)
        cold, warm, _ = stats(times)
        print(f"  {seconds:4.1f}s  {cold:7.2f}s  {warm:8.2f}s  {seconds / warm:5.1f}x")


def bench_needle(runs: int):
    print("\nNEEDLE -- tool routing (Cactus needle engine)")
    router = NeedleRouter()
    t0 = time.perf_counter()
    router.route(QUERIES[0])
    cold = time.perf_counter() - t0
    per = []
    for q in QUERIES:
        times, out = clock(lambda: router.route(q), runs)
        per.extend(times)
        print(f"  {q[:46]:46s} -> {out[0]:20s} {statistics.median(times):5.2f}s")
    print(f"  cold {cold:.2f}s   warm median {statistics.median(per):.2f}s")


def bench_answer(runs: int):
    print("\nANSWER MODEL -- writes the spoken reply")
    from hark.server import main as app
    near = app.library.window(120.0)
    passage = " ".join(c.text for c in near)[:1800]
    tool = ("ANSWER LENGTH: two or three sentences.\n\nThe listener is 2 minutes "
            "0 seconds into the book.\nThis is what they have just heard (the "
            "last 90 seconds):\n" + passage)
    question = "wait, what did that last sentence mean?"
    print(f"  backend        available   to 1st sentence   total")
    for name in ("gemini", "gemma"):
        available = local_llm.available_llms().get(name)
        if not available:
            print(f"  {name:12s}  no{'':10s}  (skipped)")
            continue

        def run():
            t0 = time.perf_counter()
            first, parts = None, []
            for sentence in app.local_answer_stream(question, tool, name):
                if first is None:
                    first = time.perf_counter() - t0
                parts.append(sentence)
            return first, time.perf_counter() - t0, " ".join(parts)

        ttfs, totals, out = [], [], ""
        for _ in range(runs):
            first, total, out = run()
            if first is not None:
                ttfs.append(first)
            totals.append(total)
        first = statistics.median(ttfs) if ttfs else float("nan")
        print(f"  {name:12s}  yes{'':9s}  {first:11.2f}s   {statistics.median(totals):6.2f}s"
              f"   {out[:60]!r}")


def bench_tts(runs: int):
    print("\nPOCKETTTS -- text to speech (PyTorch, CPU)")
    tts = PocketTtsSpeaker()
    print("  text      full(med)   TTFC(med)   total(med)   audio")
    for label, text in (("short", TTS_SHORT), ("long", TTS_LONG)):
        full, pcm = [], b""
        for _ in range(runs):
            t0 = time.perf_counter()
            pcm = tts.speak(text)
            full.append(time.perf_counter() - t0)

        def stream():
            t0 = time.perf_counter()
            first, total = None, 0
            for chunk in tts.stream(text):
                total += len(chunk)
                if first is None:
                    first = time.perf_counter() - t0
            return first, time.perf_counter() - t0, total

        ttfc, tot = [], []
        audio = 0
        for _ in range(runs):
            first, total, n = stream()
            if first is not None:
                ttfc.append(first)
            tot.append(total)
            audio = n
        seconds = audio / 2 / 24000
        print(f"  {label:7s}  {statistics.median(full):7.2f}s  "
              f"{statistics.median(ttfc):8.2f}s  {statistics.median(tot):9.2f}s  "
              f"{seconds:4.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="timed repetitions (cold is the first)")
    args = ap.parse_args()

    import torch
    import needle
    print(f"engines: cactus-needle {getattr(needle, '__version__', '?')} (STT + routing), "
          f"torch {torch.__version__} (TTS), cuda {torch.cuda.is_available()}")
    print(f"answer model: default {local_llm.default_llm()}   "
          f"available {local_llm.available_llms()}   runs {args.runs}")

    bench_stt(args.runs)
    bench_needle(args.runs)
    bench_answer(args.runs)
    bench_tts(args.runs)


if __name__ == "__main__":
    main()
