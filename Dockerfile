FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ffmpeg libsndfile1 git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source first (needed for editable install)
COPY . .

# Install Python deps + clear bytecode cache
RUN python -m pip install --upgrade pip && \
    python -m pip install -e "." && \
    find /app -name '*.pyc' -delete && \
    find /app -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; true

# Create data dirs
RUN mkdir -p data/uploads data/processed models

# Volume for persistent data (audio, results, models cache)
VOLUME ["/app/data", "/app/models", "/root/.cache"]

EXPOSE 8000

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
