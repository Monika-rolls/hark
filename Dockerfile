# Hark on Cloud Run: FastAPI + the local voice pipeline (Silero VAD, Whistle,
# Needle, PocketTTS), with Gemini answering. Gemma stays on the listener's own
# machine; see scripts/start_local_llm.ps1.
FROM python:3.11-slim

# NEEDLE_TELEMETRY=0: cactus-needle otherwise reports anonymous usage counts.
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf NEEDLE_TELEMETRY=0 \
    HARK_VOICE_PROVIDER=local HARK_LOCAL_LLM=gemini
WORKDIR /app

COPY requirements.txt requirements-cloudrun.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cloudrun.txt

COPY . .
# Fetch the speech models at build time, so a cold start does not download them.
RUN python -c "from hark.server.local_voice import LocalVoice; print(LocalVoice().warm())"

CMD ["sh", "-c", "exec uvicorn hark.server.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
