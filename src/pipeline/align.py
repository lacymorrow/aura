"""Transcript-diarization alignment.

Merges Whisper transcription output with pyannote diarization to produce
speaker-labeled transcripts: every word and segment gets a speaker tag.
"""

import logging
from dataclasses import dataclass, field

from src.pipeline.transcribe import Transcript, TranscriptSegment, Word
from src.pipeline.diarize import DiarizationResult, SpeakerTurn

logger = logging.getLogger(__name__)


@dataclass
class LabeledWord:
    """A word with speaker attribution."""

    text: str
    start: float
    end: float
    speaker: str
    probability: float


@dataclass
class LabeledSegment:
    """A transcript segment with speaker label."""

    text: str
    start: float
    end: float
    speaker: str
    words: list[LabeledWord] = field(default_factory=list)


@dataclass
class LabeledTranscript:
    """Full transcript with speaker labels on every segment and word."""

    segments: list[LabeledSegment]
    language: str
    duration: float
    num_speakers: int

    @property
    def text(self) -> str:
        return "\n".join(
            f"[{seg.speaker}] {seg.text}" for seg in self.segments
        )

    def to_dict(self) -> dict:
        """Serialize to a dictionary."""
        return {
            "language": self.language,
            "duration": self.duration,
            "num_speakers": self.num_speakers,
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "speaker": seg.speaker,
                    "words": [
                        {
                            "text": w.text,
                            "start": w.start,
                            "end": w.end,
                            "speaker": w.speaker,
                            "probability": w.probability,
                        }
                        for w in seg.words
                    ],
                }
                for seg in self.segments
            ],
        }

    def get_speaker_text(self, speaker: str) -> str:
        """Get all text spoken by a specific speaker."""
        return " ".join(
            seg.text for seg in self.segments if seg.speaker == speaker
        )

    @property
    def speakers(self) -> list[str]:
        return sorted(set(seg.speaker for seg in self.segments))


def _find_speaker_at(time: float, turns: list[SpeakerTurn]) -> str:
    """Find which speaker is active at a given timestamp.

    Uses midpoint of the time range if it falls between turns.
    Falls back to nearest turn.
    """
    for turn in turns:
        if turn.start <= time <= turn.end:
            return turn.speaker

    # Fallback: find nearest turn
    min_dist = float("inf")
    nearest = "UNKNOWN"
    for turn in turns:
        mid = (turn.start + turn.end) / 2
        dist = abs(time - mid)
        if dist < min_dist:
            min_dist = dist
            nearest = turn.speaker

    return nearest


def _assign_speaker_to_segment(
    seg: TranscriptSegment, turns: list[SpeakerTurn]
) -> str:
    """Determine the dominant speaker for a transcript segment.

    If the segment has words, vote by word midpoints.
    Otherwise, use segment midpoint.
    """
    if seg.words:
        speaker_votes: dict[str, float] = {}
        for word in seg.words:
            mid = (word.start + word.end) / 2
            spk = _find_speaker_at(mid, turns)
            dur = word.end - word.start
            speaker_votes[spk] = speaker_votes.get(spk, 0) + dur

        return max(speaker_votes, key=speaker_votes.get) if speaker_votes else "UNKNOWN"
    else:
        mid = (seg.start + seg.end) / 2
        return _find_speaker_at(mid, turns)


def align(
    transcript: Transcript, diarization: DiarizationResult
) -> LabeledTranscript:
    """Align transcript segments with diarization speaker labels.

    Each transcript segment and word gets assigned to the speaker who
    was talking during that time according to diarization.

    Args:
        transcript: Whisper transcription output.
        diarization: pyannote diarization output.

    Returns:
        LabeledTranscript with speaker labels.
    """
    logger.info(
        f"Aligning {len(transcript.segments)} transcript segments "
        f"with {len(diarization.turns)} speaker turns..."
    )

    labeled_segments = []

    for seg in transcript.segments:
        # Assign dominant speaker
        speaker = _assign_speaker_to_segment(seg, diarization.turns)

        # Label individual words
        labeled_words = []
        for word in seg.words:
            word_mid = (word.start + word.end) / 2
            word_speaker = _find_speaker_at(word_mid, diarization.turns)
            labeled_words.append(
                LabeledWord(
                    text=word.text,
                    start=word.start,
                    end=word.end,
                    speaker=word_speaker,
                    probability=word.probability,
                )
            )

        labeled_segments.append(
            LabeledSegment(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                speaker=speaker,
                words=labeled_words,
            )
        )

    # Merge consecutive segments from the same speaker
    merged = _merge_consecutive_speaker_segments(labeled_segments)

    result = LabeledTranscript(
        segments=merged,
        language=transcript.language,
        duration=transcript.duration,
        num_speakers=diarization.num_speakers,
    )

    logger.info(
        f"Alignment complete: {len(merged)} merged segments, "
        f"{result.num_speakers} speakers"
    )

    return result


def _merge_consecutive_speaker_segments(
    segments: list[LabeledSegment],
) -> list[LabeledSegment]:
    """Merge consecutive segments from the same speaker."""
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.speaker == merged[-1].speaker:
            # Merge: extend the previous segment
            merged[-1] = LabeledSegment(
                text=merged[-1].text + " " + seg.text,
                start=merged[-1].start,
                end=seg.end,
                speaker=seg.speaker,
                words=merged[-1].words + seg.words,
            )
        else:
            merged.append(seg)

    return merged
