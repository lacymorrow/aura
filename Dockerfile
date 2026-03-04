FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY pyproject.toml .
RUN python3.11 -m pip install --no-cache-dir -e ".[dev]"

COPY . .

# Pre-download models on build (optional, makes runtime faster)
# RUN python3.11 -c "from src.pipeline.vad import VoiceActivityDetector; VoiceActivityDetector().model"

EXPOSE 8000

CMD ["python3.11", "-m", "src.cli", "process", "--help"]
