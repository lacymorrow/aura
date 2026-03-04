"""Main pipeline orchestrator.

Runs the full processing chain on an audio file:
VAD → Transcription → Diarization → Speaker Embedding → Alignment → Knowledge Extraction
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Complete output from processing a single audio file."""

    audio_path: str
    duration: float
    language: str
    num_speakers: int
    num_segments: int
    num_speech_segments: int
    speech_ratio: float
    transcript_text: str
    labeled_transcript: dict
    speaker_embeddings: list[dict]
    extraction: dict | None = None
    processing_time: float = 0.0
    errors: list[str] = field(default_factory=list)


class AudioProcessor:
    """Orchestrates the full audio processing pipeline."""

    def __init__(
        self,
        enable_extraction: bool = True,
        owner_speaker: str = "SPEAKER_00",
    ):
        self.enable_extraction = enable_extraction
        self.owner_speaker = owner_speaker

        # Lazy-loaded components
        self._vad = None
        self._transcriber = None
        self._diarizer = None
        self._embedder = None
        self._extractor = None

    @property
    def vad(self):
        if self._vad is None:
            from src.pipeline.vad import VoiceActivityDetector

            self._vad = VoiceActivityDetector()
        return self._vad

    @property
    def transcriber(self):
        if self._transcriber is None:
            from src.pipeline.transcribe import Transcriber

            self._transcriber = Transcriber()
        return self._transcriber

    @property
    def diarizer(self):
        if self._diarizer is None:
            from src.pipeline.diarize import Diarizer

            self._diarizer = Diarizer()
        return self._diarizer

    @property
    def embedder(self):
        if self._embedder is None:
            from src.pipeline.speaker_embed import SpeakerEmbedder

            self._embedder = SpeakerEmbedder()
        return self._embedder

    @property
    def extractor(self):
        if self._extractor is None:
            from src.pipeline.extract import KnowledgeExtractor

            self._extractor = KnowledgeExtractor()
        return self._extractor

    def process(
        self,
        audio_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> ProcessingResult:
        """Run the full pipeline on an audio file.

        Args:
            audio_path: Path to the audio file.
            output_dir: Directory to save results (optional).

        Returns:
            ProcessingResult with all pipeline outputs.
        """
        audio_path = Path(audio_path)
        start_time = time.time()
        errors = []

        logger.info(f"{'=' * 60}")
        logger.info(f"Processing: {audio_path.name}")
        logger.info(f"{'=' * 60}")

        # --- Stage 1: VAD ---
        logger.info("\n--- Stage 1: Voice Activity Detection ---")
        t0 = time.time()
        speech_segments = self.vad.detect(audio_path)
        logger.info(f"VAD took {time.time() - t0:.1f}s")

        # --- Stage 2: Transcription ---
        logger.info("\n--- Stage 2: Transcription ---")
        t0 = time.time()
        transcript = self.transcriber.transcribe(audio_path)
        logger.info(f"Transcription took {time.time() - t0:.1f}s")

        # --- Stage 3: Diarization ---
        logger.info("\n--- Stage 3: Speaker Diarization ---")
        t0 = time.time()
        diarization = self.diarizer.diarize(audio_path)
        logger.info(f"Diarization took {time.time() - t0:.1f}s")

        # --- Stage 4: Speaker Embeddings ---
        logger.info("\n--- Stage 4: Speaker Embedding ---")
        t0 = time.time()
        embeddings = self.embedder.extract_per_speaker(
            audio_path, diarization.turns
        )
        logger.info(f"Embedding extraction took {time.time() - t0:.1f}s")

        # --- Stage 5: Alignment ---
        logger.info("\n--- Stage 5: Transcript-Diarization Alignment ---")
        from src.pipeline.align import align

        labeled_transcript = align(transcript, diarization)

        # --- Stage 6: Knowledge Extraction ---
        extraction_result = None
        if self.enable_extraction:
            logger.info("\n--- Stage 6: Knowledge Extraction ---")
            t0 = time.time()
            try:
                extraction_result = self.extractor.extract(
                    labeled_transcript, owner_speaker=self.owner_speaker
                )
                logger.info(f"Extraction took {time.time() - t0:.1f}s")
            except Exception as e:
                logger.error(f"Knowledge extraction failed: {e}")
                errors.append(f"extraction: {e}")

        # --- Assemble result ---
        total_speech = sum(s.duration for s in speech_segments)
        speech_ratio = total_speech / transcript.duration if transcript.duration > 0 else 0

        result = ProcessingResult(
            audio_path=str(audio_path),
            duration=transcript.duration,
            language=transcript.language,
            num_speakers=diarization.num_speakers,
            num_segments=len(labeled_transcript.segments),
            num_speech_segments=len(speech_segments),
            speech_ratio=round(speech_ratio, 3),
            transcript_text=labeled_transcript.text,
            labeled_transcript=labeled_transcript.to_dict(),
            speaker_embeddings=[
                {
                    "speaker": e.speaker,
                    "embedding": e.embedding.tolist(),
                    "duration": e.duration,
                }
                for e in embeddings
            ],
            extraction=extraction_result.raw_json if extraction_result else None,
            processing_time=round(time.time() - start_time, 1),
            errors=errors,
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing complete in {result.processing_time}s")
        logger.info(f"  Duration: {result.duration:.1f}s")
        logger.info(f"  Speech: {speech_ratio * 100:.0f}%")
        logger.info(f"  Speakers: {result.num_speakers}")
        logger.info(f"  Segments: {result.num_segments}")
        if errors:
            logger.warning(f"  Errors: {errors}")
        logger.info(f"{'=' * 60}")

        # Save results if output_dir specified
        if output_dir:
            self._save_results(result, Path(output_dir))

        return result

    def _save_results(self, result: ProcessingResult, output_dir: Path):
        """Save processing results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(result.audio_path).stem

        # Save labeled transcript
        transcript_path = output_dir / f"{stem}_transcript.json"
        with open(transcript_path, "w") as f:
            json.dump(result.labeled_transcript, f, indent=2)
        logger.info(f"Saved transcript: {transcript_path}")

        # Save readable transcript
        text_path = output_dir / f"{stem}_transcript.txt"
        with open(text_path, "w") as f:
            f.write(result.transcript_text)
        logger.info(f"Saved text: {text_path}")

        # Save speaker embeddings (without the actual vectors for readability)
        embeddings_path = output_dir / f"{stem}_speakers.json"
        speakers_info = [
            {"speaker": e["speaker"], "duration": e["duration"]}
            for e in result.speaker_embeddings
        ]
        with open(embeddings_path, "w") as f:
            json.dump(speakers_info, f, indent=2)

        # Save extraction results
        if result.extraction:
            extract_path = output_dir / f"{stem}_knowledge.json"
            with open(extract_path, "w") as f:
                json.dump(result.extraction, f, indent=2)
            logger.info(f"Saved knowledge: {extract_path}")

        # Save full result
        full_path = output_dir / f"{stem}_full.json"
        # Exclude embeddings from full save (too large)
        save_data = {
            "audio_path": result.audio_path,
            "duration": result.duration,
            "language": result.language,
            "num_speakers": result.num_speakers,
            "num_segments": result.num_segments,
            "speech_ratio": result.speech_ratio,
            "processing_time": result.processing_time,
            "errors": result.errors,
            "extraction": result.extraction,
        }
        with open(full_path, "w") as f:
            json.dump(save_data, f, indent=2)
