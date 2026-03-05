"""Voice Activity Detection using Silero VAD.

Strips silence and non-speech audio, returning only speech segments
with timestamps relative to the original audio.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A detected speech segment."""

    start: float  # seconds
    end: float  # seconds
    confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class VoiceActivityDetector:
    """Silero VAD wrapper for detecting speech segments in audio."""

    def __init__(
        self,
        threshold: float = settings.vad_threshold,
        min_speech_duration_ms: int = settings.min_speech_duration_ms,
        max_speech_duration_s: float = settings.max_speech_duration_s,
        speech_pad_ms: int = settings.speech_pad_ms,
        sample_rate: int = settings.sample_rate,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate
        self._model = None

    def _ensure_loaded(self):
        """Lazy-load Silero VAD model and utilities."""
        if self._model is None:
            logger.info("Loading Silero VAD model...")
            try:
                # silero-vad >= 5.x package API
                from silero_vad import load_silero_vad, get_speech_timestamps as _gst
                self._model = load_silero_vad()
                self._get_speech_timestamps = _gst
            except ImportError:
                # Fallback: torch.hub API
                self._model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                self._get_speech_timestamps = utils[0]
            logger.info("Silero VAD model loaded.")

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    def detect(self, audio_path: str | Path) -> list[SpeechSegment]:
        """Detect speech segments in an audio file.

        Args:
            audio_path: Path to audio file (any format soundfile supports).

        Returns:
            List of SpeechSegment with start/end times and confidence.
        """
        self._ensure_loaded()
        audio_path = Path(audio_path)
        logger.info(f"Running VAD on {audio_path.name}")

        # Load and resample audio to 16kHz mono
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # mono
        if sr != self.sample_rate:
            import torchaudio

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze(0).numpy()

        # Run Silero VAD
        audio_tensor = torch.from_numpy(audio)

        speech_timestamps = self._get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            max_speech_duration_s=self.max_speech_duration_s,
            speech_pad_ms=self.speech_pad_ms,
            sampling_rate=self.sample_rate,
            return_seconds=False,
        )

        segments = []
        for ts in speech_timestamps:
            start_sec = ts["start"] / self.sample_rate
            end_sec = ts["end"] / self.sample_rate
            segments.append(
                SpeechSegment(
                    start=round(start_sec, 3),
                    end=round(end_sec, 3),
                    confidence=1.0,  # Silero doesn't return per-segment confidence
                )
            )

        total_speech = sum(s.duration for s in segments)
        total_audio = len(audio) / self.sample_rate
        logger.info(
            f"VAD complete: {len(segments)} segments, "
            f"{total_speech:.1f}s speech / {total_audio:.1f}s total "
            f"({total_speech / total_audio * 100:.0f}%)"
        )

        return segments

    def extract_speech_audio(
        self, audio_path: str | Path, segments: list[SpeechSegment]
    ) -> tuple[np.ndarray, int]:
        """Extract only the speech portions of audio, concatenated.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        chunks = []
        for seg in segments:
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            chunks.append(audio[start_sample:end_sample])

        if not chunks:
            return np.array([], dtype="float32"), sr

        return np.concatenate(chunks), sr

    def save_speech_segments(
        self,
        audio_path: str | Path,
        segments: list[SpeechSegment],
        output_dir: str | Path,
    ) -> list[Path]:
        """Save each speech segment as a separate WAV file.

        Returns:
            List of paths to saved segment files.
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        paths = []
        for i, seg in enumerate(segments):
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            chunk = audio[start_sample:end_sample]

            out_path = output_dir / f"{audio_path.stem}_seg{i:04d}_{seg.start:.1f}-{seg.end:.1f}.wav"
            sf.write(str(out_path), chunk, sr)
            paths.append(out_path)

        return paths
