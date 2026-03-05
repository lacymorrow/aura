"""Main pipeline orchestrator.

Runs the full processing chain on an audio file:
VAD → Transcription → Diarization → Speaker Embedding → Alignment → Knowledge Extraction

Production-grade: retries, graceful degradation, structured output.
"""

import hashlib
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
    file_hash: str
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
    speaker_matches: dict = field(default_factory=dict)  # label -> {speaker_id, name, confidence, similarity}
    processing_time: float = 0.0
    stage_times: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _file_hash(path: Path) -> str:
    """SHA-256 of first 1MB (fast enough for dedup)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()


class AudioProcessor:
    """Orchestrates the full audio processing pipeline."""

    def __init__(
        self,
        enable_extraction: bool = True,
        enable_db: bool = False,
        owner_speaker: str = "SPEAKER_00",
        retry_extraction: int = 2,
    ):
        self.enable_extraction = enable_extraction
        self.enable_db = enable_db
        self.owner_speaker = owner_speaker
        self.retry_extraction = retry_extraction

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

        Gracefully degrades: if extraction fails, you still get transcript +
        diarization + embeddings. If diarization fails, you still get transcript.

        Args:
            audio_path: Path to the audio file.
            output_dir: Directory to save results (optional).

        Returns:
            ProcessingResult with all pipeline outputs.
        """
        audio_path = Path(audio_path)
        start_time = time.time()
        errors = []
        warnings = []
        stage_times = {}

        fhash = _file_hash(audio_path)

        logger.info(f"{'=' * 60}")
        logger.info(f"Processing: {audio_path.name}")
        logger.info(f"  Hash: {fhash[:12]}...")
        logger.info(f"{'=' * 60}")

        # --- Stage 1: VAD ---
        logger.info("\n--- Stage 1: Voice Activity Detection ---")
        t0 = time.time()
        try:
            speech_segments = self.vad.detect(audio_path)
        except Exception as e:
            logger.error(f"VAD failed: {e}")
            errors.append(f"vad: {e}")
            speech_segments = []
        stage_times["vad"] = round(time.time() - t0, 2)

        # --- Stage 2: Transcription ---
        logger.info("\n--- Stage 2: Transcription ---")
        t0 = time.time()
        try:
            transcript = self.transcriber.transcribe(audio_path)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise  # Can't continue without a transcript
        stage_times["transcribe"] = round(time.time() - t0, 2)

        # --- Stage 3: Diarization ---
        logger.info("\n--- Stage 3: Speaker Diarization ---")
        t0 = time.time()
        diarization = None
        try:
            diarization = self.diarizer.diarize(audio_path)
        except Exception as e:
            logger.error(f"Diarization failed (degrading gracefully): {e}")
            errors.append(f"diarization: {e}")
            warnings.append("Diarization failed — transcript will lack speaker labels")
        stage_times["diarize"] = round(time.time() - t0, 2)

        # --- Stage 4: Speaker Embeddings ---
        embeddings = []
        speaker_matches = {}  # diarization_label -> SpeakerMatch
        if diarization:
            logger.info("\n--- Stage 4: Speaker Embedding ---")
            t0 = time.time()
            try:
                embeddings = self.embedder.extract_per_speaker(
                    audio_path, diarization.turns
                )

                # Match against known speakers in DB
                if self.enable_db and embeddings:
                    try:
                        from src.speakers.registry import SpeakerRegistry
                        registry = SpeakerRegistry()
                        for emb in embeddings:
                            match = registry.identify(emb, recording_duration=emb.duration)
                            speaker_matches[emb.speaker] = match
                            logger.info(
                                f"  {emb.speaker} → {match.name or f'speaker_{match.speaker_id[:8]}'} "
                                f"(confidence={match.confidence}, sim={match.similarity:.4f}, new={match.is_new})"
                            )
                    except Exception as e:
                        logger.error(f"Speaker matching failed (non-fatal): {e}")
                        errors.append(f"speaker_matching: {e}")

            except Exception as e:
                logger.error(f"Embedding extraction failed: {e}")
                errors.append(f"embeddings: {e}")
            stage_times["embed"] = round(time.time() - t0, 2)

        # --- Stage 5: Alignment ---
        logger.info("\n--- Stage 5: Transcript-Diarization Alignment ---")
        from src.pipeline.align import align, LabeledTranscript, LabeledSegment

        if diarization:
            labeled_transcript = align(transcript, diarization)
        else:
            # Fallback: no diarization, treat everything as single speaker
            labeled_transcript = LabeledTranscript(
                segments=[
                    LabeledSegment(
                        text=seg.text,
                        start=seg.start,
                        end=seg.end,
                        speaker="SPEAKER_00",
                        words=[],
                    )
                    for seg in transcript.segments
                ],
                language=transcript.language,
                duration=transcript.duration,
                num_speakers=1,
            )

        # --- Stage 6: Knowledge Extraction ---
        extraction_result = None
        if self.enable_extraction:
            logger.info("\n--- Stage 6: Knowledge Extraction ---")
            t0 = time.time()
            last_error = None
            for attempt in range(1, self.retry_extraction + 1):
                try:
                    extraction_result = self.extractor.extract(
                        labeled_transcript, owner_speaker=self.owner_speaker
                    )
                    break
                except Exception as e:
                    last_error = e
                    if attempt < self.retry_extraction:
                        logger.warning(f"Extraction attempt {attempt} failed: {e}. Retrying...")
                        time.sleep(2 ** attempt)  # exponential backoff
                    else:
                        logger.error(f"Knowledge extraction failed after {attempt} attempts: {e}")
                        errors.append(f"extraction: {e}")
            stage_times["extract"] = round(time.time() - t0, 2)

        # --- Assemble result ---
        total_speech = sum(s.duration for s in speech_segments)
        speech_ratio = total_speech / transcript.duration if transcript.duration > 0 else 0

        result = ProcessingResult(
            audio_path=str(audio_path),
            file_hash=fhash,
            duration=transcript.duration,
            language=transcript.language,
            num_speakers=diarization.num_speakers if diarization else 1,
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
            speaker_matches={
                label: {
                    "speaker_id": match.speaker_id,
                    "name": match.name,
                    "confidence": match.confidence,
                    "similarity": round(match.similarity, 4),
                    "is_new": match.is_new,
                }
                for label, match in speaker_matches.items()
            },
            processing_time=round(time.time() - start_time, 1),
            stage_times=stage_times,
            errors=errors,
            warnings=warnings,
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing complete in {result.processing_time}s")
        logger.info(f"  Duration: {result.duration:.1f}s")
        logger.info(f"  Speech: {speech_ratio * 100:.0f}%")
        logger.info(f"  Speakers: {result.num_speakers}")
        logger.info(f"  Segments: {result.num_segments}")
        logger.info(f"  Stage times: {stage_times}")
        if warnings:
            for w in warnings:
                logger.warning(f"  ⚠ {w}")
        if errors:
            logger.warning(f"  Errors: {errors}")
        logger.info(f"{'=' * 60}")

        # Save results
        if output_dir:
            self._save_results(result, Path(output_dir))

        # Persist to database
        if self.enable_db:
            try:
                from src.db.persist import persist_result
                persist_result(result, speaker_matches=speaker_matches or None)
                logger.info("Results persisted to database.")
            except Exception as e:
                logger.error(f"DB persistence failed (non-fatal): {e}")
                errors.append(f"db: {e}")

        return result

    def _save_results(self, result: ProcessingResult, output_dir: Path):
        """Save processing results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(result.audio_path).stem

        # Save labeled transcript (JSON)
        transcript_path = output_dir / f"{stem}_transcript.json"
        with open(transcript_path, "w") as f:
            json.dump(result.labeled_transcript, f, indent=2)
        logger.info(f"Saved transcript: {transcript_path}")

        # Save readable transcript (TXT)
        text_path = output_dir / f"{stem}_transcript.txt"
        with open(text_path, "w") as f:
            f.write(result.transcript_text)
        logger.info(f"Saved text: {text_path}")

        # Save speaker embeddings metadata
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

        # Save full result metadata (no raw embeddings — too large)
        full_path = output_dir / f"{stem}_full.json"
        save_data = {
            "audio_path": result.audio_path,
            "file_hash": result.file_hash,
            "duration": result.duration,
            "language": result.language,
            "num_speakers": result.num_speakers,
            "num_segments": result.num_segments,
            "speech_ratio": result.speech_ratio,
            "processing_time": result.processing_time,
            "stage_times": result.stage_times,
            "errors": result.errors,
            "warnings": result.warnings,
            "extraction": result.extraction,
        }
        with open(full_path, "w") as f:
            json.dump(save_data, f, indent=2)
