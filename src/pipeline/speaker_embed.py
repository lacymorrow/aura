"""Speaker embedding extraction using SpeechBrain ECAPA-TDNN.

Generates 192-dimensional voice embeddings (voiceprints) for speaker segments.
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
class SpeakerEmbedding:
    """A voice embedding for a speaker segment."""

    speaker: str  # diarization label (e.g. "SPEAKER_00")
    embedding: np.ndarray  # 192-dim vector
    start: float  # segment start time
    end: float  # segment end time
    duration: float  # total speech used for this embedding


class SpeakerEmbedder:
    """Extract speaker embeddings using ECAPA-TDNN."""

    def __init__(
        self,
        model_source: str = settings.speaker_embedding_model,
        min_duration: float = settings.min_embedding_duration_s,
    ):
        self.model_source = model_source
        self.min_duration = min_duration
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            from speechbrain.inference.speaker import EncoderClassifier

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading speaker encoder: {self.model_source} on {device}")

            self._encoder = EncoderClassifier.from_hparams(
                source=self.model_source,
                run_opts={"device": device},
            )
            logger.info("Speaker encoder loaded.")
        return self._encoder

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract a single embedding from an audio array.

        Args:
            audio: Audio samples (float32, mono).
            sample_rate: Sample rate of the audio.

        Returns:
            192-dim numpy embedding vector.
        """
        if len(audio) < sample_rate * self.min_duration:
            logger.warning(
                f"Audio too short ({len(audio) / sample_rate:.1f}s < {self.min_duration}s)"
            )

        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        embedding = self.encoder.encode_batch(audio_tensor)
        return embedding.squeeze().cpu().numpy()

    def extract_from_file(
        self,
        audio_path: str | Path,
        start: float = 0.0,
        end: float | None = None,
    ) -> np.ndarray:
        """Extract embedding from a region of an audio file.

        Args:
            audio_path: Path to audio file.
            start: Start time in seconds.
            end: End time in seconds (None = end of file).

        Returns:
            192-dim numpy embedding vector.
        """
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        start_sample = int(start * sr)
        end_sample = int(end * sr) if end else len(audio)
        segment = audio[start_sample:end_sample]

        return self.extract_embedding(segment, sr)

    def extract_per_speaker(
        self,
        audio_path: str | Path,
        speaker_turns: list,  # list of SpeakerTurn from diarizer
    ) -> list[SpeakerEmbedding]:
        """Extract one embedding per speaker by concatenating their turns.

        For each unique speaker label, concatenates all their speech segments
        and extracts a single embedding from the combined audio. This gives
        a more robust voiceprint than any single turn.

        Args:
            audio_path: Path to audio file.
            speaker_turns: Speaker turns from diarization.

        Returns:
            List of SpeakerEmbedding, one per unique speaker.
        """
        audio_path = Path(audio_path)
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Group turns by speaker
        speaker_segments: dict[str, list] = {}
        for turn in speaker_turns:
            if turn.speaker not in speaker_segments:
                speaker_segments[turn.speaker] = []
            speaker_segments[turn.speaker].append(turn)

        embeddings = []
        for speaker, turns in speaker_segments.items():
            # Concatenate all audio for this speaker
            chunks = []
            total_duration = 0.0
            for turn in turns:
                start_sample = int(turn.start * sr)
                end_sample = int(turn.end * sr)
                chunks.append(audio[start_sample:end_sample])
                total_duration += turn.duration

            if total_duration < self.min_duration:
                logger.warning(
                    f"Speaker {speaker} has only {total_duration:.1f}s of speech "
                    f"(min: {self.min_duration}s), embedding may be unreliable"
                )

            combined = np.concatenate(chunks)
            emb = self.extract_embedding(combined, sr)

            embeddings.append(
                SpeakerEmbedding(
                    speaker=speaker,
                    embedding=emb,
                    start=turns[0].start,
                    end=turns[-1].end,
                    duration=total_duration,
                )
            )
            logger.info(
                f"Extracted embedding for {speaker}: "
                f"{total_duration:.1f}s speech, dim={emb.shape[0]}"
            )

        return embeddings

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
