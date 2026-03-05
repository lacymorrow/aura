"""Pipeline unit tests.

These tests validate data structures and logic without requiring GPU or models.
For integration tests that need the full ML stack, use Docker.
"""

import json
import numpy as np
import pytest

from src.pipeline.vad import SpeechSegment
from src.pipeline.transcribe import Transcript, TranscriptSegment, Word
from src.pipeline.diarize import DiarizationResult, SpeakerTurn
from src.pipeline.align import align, LabeledTranscript
from src.pipeline.speaker_embed import SpeakerEmbedder


class TestSpeechSegment:
    def test_duration(self):
        seg = SpeechSegment(start=1.0, end=3.5, confidence=0.9)
        assert seg.duration == pytest.approx(2.5)

    def test_zero_duration(self):
        seg = SpeechSegment(start=5.0, end=5.0, confidence=1.0)
        assert seg.duration == 0.0


class TestDiarizationResult:
    def test_speakers(self):
        result = DiarizationResult(
            turns=[
                SpeakerTurn(speaker="SPEAKER_00", start=0, end=5),
                SpeakerTurn(speaker="SPEAKER_01", start=5, end=10),
                SpeakerTurn(speaker="SPEAKER_00", start=10, end=15),
            ],
            num_speakers=2,
        )
        assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]
        assert result.speaker_duration("SPEAKER_00") == pytest.approx(10.0)
        assert result.speaker_duration("SPEAKER_01") == pytest.approx(5.0)

    def test_get_turns_for_speaker(self):
        result = DiarizationResult(
            turns=[
                SpeakerTurn(speaker="SPEAKER_00", start=0, end=5),
                SpeakerTurn(speaker="SPEAKER_01", start=5, end=10),
            ],
            num_speakers=2,
        )
        turns = result.get_turns_for_speaker("SPEAKER_01")
        assert len(turns) == 1
        assert turns[0].start == 5


class TestAlignment:
    def _make_transcript(self, segments_data):
        segments = []
        for text, start, end in segments_data:
            segments.append(TranscriptSegment(
                text=text, start=start, end=end, words=[
                    Word(text=text, start=start, end=end, probability=0.95)
                ],
            ))
        return Transcript(segments=segments, language="en", duration=max(e for _, _, e in segments_data))

    def _make_diarization(self, turns_data):
        turns = [SpeakerTurn(speaker=s, start=st, end=en) for s, st, en in turns_data]
        num_speakers = len(set(t.speaker for t in turns))
        return DiarizationResult(turns=turns, num_speakers=num_speakers)

    def test_basic_alignment(self):
        transcript = self._make_transcript([
            ("Hello", 0.0, 1.0),
            ("World", 2.0, 3.0),
        ])
        diarization = self._make_diarization([
            ("SPEAKER_00", 0.0, 1.5),
            ("SPEAKER_01", 1.5, 3.5),
        ])

        result = align(transcript, diarization)
        assert isinstance(result, LabeledTranscript)
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "SPEAKER_00"
        assert result.segments[1].speaker == "SPEAKER_01"

    def test_merge_consecutive_same_speaker(self):
        transcript = self._make_transcript([
            ("Hello", 0.0, 1.0),
            ("there", 1.0, 2.0),
            ("friend", 3.0, 4.0),
        ])
        diarization = self._make_diarization([
            ("SPEAKER_00", 0.0, 4.5),
        ])

        result = align(transcript, diarization)
        # All 3 segments should merge into 1
        assert len(result.segments) == 1
        assert "Hello" in result.segments[0].text
        assert "friend" in result.segments[0].text

    def test_labeled_transcript_text(self):
        transcript = self._make_transcript([
            ("Hi", 0.0, 1.0),
            ("Hey", 2.0, 3.0),
        ])
        diarization = self._make_diarization([
            ("SPEAKER_00", 0.0, 1.5),
            ("SPEAKER_01", 1.5, 3.5),
        ])

        result = align(transcript, diarization)
        text = result.text
        assert "[SPEAKER_00]" in text
        assert "[SPEAKER_01]" in text

    def test_to_dict_roundtrip(self):
        transcript = self._make_transcript([("Test", 0.0, 1.0)])
        diarization = self._make_diarization([("SPEAKER_00", 0.0, 1.5)])

        result = align(transcript, diarization)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["language"] == "en"
        assert len(d["segments"]) == 1
        # Verify it's JSON-serializable
        json.dumps(d)


class TestSpeakerEmbedder:
    def test_cosine_similarity_identical(self):
        v = np.random.randn(192).astype(np.float32)
        sim = SpeakerEmbedder.cosine_similarity(v, v)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = SpeakerEmbedder.cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-5)

    def test_cosine_similarity_opposite(self):
        a = np.random.randn(192).astype(np.float32)
        sim = SpeakerEmbedder.cosine_similarity(a, -a)
        assert sim == pytest.approx(-1.0, abs=1e-5)

    def test_cosine_similarity_range(self):
        for _ in range(10):
            a = np.random.randn(192).astype(np.float32)
            b = np.random.randn(192).astype(np.float32)
            sim = SpeakerEmbedder.cosine_similarity(a, b)
            assert -1.0 <= sim <= 1.0


class TestWatcher:
    def test_audio_extensions(self):
        from src.pipeline.watcher import AUDIO_EXTENSIONS
        assert ".wav" in AUDIO_EXTENSIONS
        assert ".mp3" in AUDIO_EXTENSIONS
        assert ".flac" in AUDIO_EXTENSIONS
        assert ".txt" not in AUDIO_EXTENSIONS
        assert ".py" not in AUDIO_EXTENSIONS
