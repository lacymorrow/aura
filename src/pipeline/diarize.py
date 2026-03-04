"""Speaker diarization using pyannote.audio.

Determines "who spoke when" — segments audio by speaker turns.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    """A segment where a single speaker is talking."""

    speaker: str  # e.g. "SPEAKER_00"
    start: float  # seconds
    end: float  # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Full diarization output for an audio file."""

    turns: list[SpeakerTurn]
    num_speakers: int

    @property
    def speakers(self) -> list[str]:
        return sorted(set(t.speaker for t in self.turns))

    def speaker_duration(self, speaker: str) -> float:
        return sum(t.duration for t in self.turns if t.speaker == speaker)

    def get_turns_for_speaker(self, speaker: str) -> list[SpeakerTurn]:
        return [t for t in self.turns if t.speaker == speaker]


class Diarizer:
    """pyannote.audio speaker diarization pipeline."""

    def __init__(
        self,
        model: str = settings.diarization_model,
        hf_token: str = settings.hf_token,
    ):
        self.model_name = model
        self.hf_token = hf_token
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            logger.info(f"Loading diarization pipeline: {self.model_name}")

            if self.hf_token:
                self._pipeline = Pipeline.from_pretrained(
                    self.model_name, use_auth_token=self.hf_token
                )
            else:
                self._pipeline = Pipeline.from_pretrained(self.model_name)

            # Move to GPU if available
            import torch

            if torch.cuda.is_available():
                self._pipeline = self._pipeline.to(torch.device("cuda"))
                logger.info("Diarization pipeline loaded on GPU.")
            else:
                logger.info("Diarization pipeline loaded on CPU.")

        return self._pipeline

    def diarize(
        self,
        audio_path: str | Path,
        min_speakers: int = settings.min_speakers,
        max_speakers: int = settings.max_speakers,
    ) -> DiarizationResult:
        """Run speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file.
            min_speakers: Minimum expected number of speakers.
            max_speakers: Maximum expected number of speakers.

        Returns:
            DiarizationResult with speaker turns.
        """
        audio_path = Path(audio_path)
        logger.info(f"Diarizing {audio_path.name}...")

        diarization = self.pipeline(
            str(audio_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    speaker=speaker,
                    start=round(turn.start, 3),
                    end=round(turn.end, 3),
                )
            )

        # Sort by start time
        turns.sort(key=lambda t: t.start)

        num_speakers = len(set(t.speaker for t in turns))
        result = DiarizationResult(turns=turns, num_speakers=num_speakers)

        logger.info(
            f"Diarization complete: {num_speakers} speakers, "
            f"{len(turns)} turns"
        )
        for spk in result.speakers:
            dur = result.speaker_duration(spk)
            logger.info(f"  {spk}: {dur:.1f}s")

        return result
