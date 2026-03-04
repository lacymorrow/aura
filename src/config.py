"""Aura configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # Paths
    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")

    # Audio processing
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = 30.0
    speech_pad_ms: int = 400

    # Transcription
    whisper_model: str = "large-v3"
    whisper_device: str = "auto"  # "auto", "cuda", "cpu"
    whisper_compute_type: str = "float16"
    whisper_beam_size: int = 5
    whisper_language: str | None = None  # None = auto-detect

    # Diarization
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    hf_token: str = ""  # Required for pyannote
    min_speakers: int = 1
    max_speakers: int = 10

    # Speaker identification
    speaker_embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_match_threshold: float = 0.75
    speaker_candidate_threshold: float = 0.55
    min_embedding_duration_s: float = 2.0

    # Knowledge extraction
    llm_provider: str = "openai"  # "openai", "local", "anthropic"
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""

    # Database
    database_url: str = "postgresql://localhost:5432/aura"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_prefix": "AURA_", "env_file": ".env"}


settings = Settings()
