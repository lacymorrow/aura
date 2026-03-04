"""Transcription using faster-whisper (CTranslate2 backend).

Generates word-level timestamped transcripts from audio files.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Word:
    """A single transcribed word with timing."""

    text: str
    start: float  # seconds
    end: float  # seconds
    probability: float


@dataclass
class TranscriptSegment:
    """A transcription segment (typically a sentence or phrase)."""

    text: str
    start: float  # seconds
    end: float  # seconds
    words: list[Word] = field(default_factory=list)
    language: str = ""
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class Transcript:
    """Full transcript of an audio file."""

    segments: list[TranscriptSegment]
    language: str
    duration: float  # total audio duration in seconds

    @property
    def text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments)

    @property
    def word_count(self) -> int:
        return sum(len(seg.words) for seg in self.segments)


class Transcriber:
    """faster-whisper transcription engine."""

    def __init__(
        self,
        model_size: str = settings.whisper_model,
        device: str = settings.whisper_device,
        compute_type: str = settings.whisper_compute_type,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            device = self.device
            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"

            compute = self.compute_type if device == "cuda" else "int8"

            logger.info(
                f"Loading Whisper {self.model_size} on {device} ({compute})..."
            )
            self._model = WhisperModel(
                self.model_size, device=device, compute_type=compute
            )
            logger.info("Whisper model loaded.")
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = settings.whisper_language,
        beam_size: int = settings.whisper_beam_size,
    ) -> Transcript:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file.
            language: Language code (e.g. "en") or None for auto-detect.
            beam_size: Beam size for decoding.

        Returns:
            Transcript with word-level timestamps.
        """
        audio_path = Path(audio_path)
        logger.info(f"Transcribing {audio_path.name}...")

        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_speech_duration_ms=250,
                max_speech_duration_s=30,
            ),
        )

        segments = []
        for seg in segments_iter:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        Word(
                            text=w.word,
                            start=round(w.start, 3),
                            end=round(w.end, 3),
                            probability=round(w.probability, 4),
                        )
                    )

            segments.append(
                TranscriptSegment(
                    text=seg.text.strip(),
                    start=round(seg.start, 3),
                    end=round(seg.end, 3),
                    words=words,
                    language=info.language,
                    avg_logprob=round(seg.avg_logprob, 4),
                    no_speech_prob=round(seg.no_speech_prob, 4),
                )
            )

        transcript = Transcript(
            segments=segments,
            language=info.language,
            duration=info.duration,
        )

        logger.info(
            f"Transcription complete: {len(segments)} segments, "
            f"{transcript.word_count} words, "
            f"language={info.language}, "
            f"duration={info.duration:.1f}s"
        )

        return transcript
