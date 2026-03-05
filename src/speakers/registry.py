"""Speaker registry — persistent speaker identification across sessions.

Matches new voiceprints against known speakers using cosine similarity.
Updates speaker profiles with running-average embeddings.
"""

import datetime
import logging
from dataclasses import dataclass

import numpy as np

from src.config import settings
from src.db.engine import get_session
from src.db.models import Speaker
from src.pipeline.speaker_embed import SpeakerEmbedding, SpeakerEmbedder

logger = logging.getLogger(__name__)


@dataclass
class SpeakerMatch:
    """Result of matching an embedding against the registry."""

    speaker_id: str  # UUID
    name: str | None
    similarity: float
    confidence: str  # "high", "medium", "low"
    is_new: bool  # True if this speaker was just created


class SpeakerRegistry:
    """Manages persistent speaker profiles with voiceprint matching."""

    def __init__(
        self,
        match_threshold: float = settings.speaker_match_threshold,
        candidate_threshold: float = settings.speaker_candidate_threshold,
    ):
        self.match_threshold = match_threshold
        self.candidate_threshold = candidate_threshold

    def identify(
        self,
        embedding: SpeakerEmbedding,
        recording_duration: float = 0.0,
    ) -> SpeakerMatch:
        """Match an embedding against known speakers, or create a new one.

        Args:
            embedding: The speaker embedding to identify.
            recording_duration: Duration of speech used for this embedding.

        Returns:
            SpeakerMatch with identification result.
        """
        session = get_session()
        try:
            # Load all known speaker embeddings
            known_speakers = session.query(Speaker).filter(
                Speaker.embedding.isnot(None)
            ).all()

            best_match = None
            best_similarity = -1.0

            for speaker in known_speakers:
                known_emb = np.array(speaker.embedding, dtype=np.float32)
                sim = SpeakerEmbedder.cosine_similarity(embedding.embedding, known_emb)

                if sim > best_similarity:
                    best_similarity = sim
                    best_match = speaker

            # Determine confidence level
            if best_match and best_similarity >= self.match_threshold:
                # High confidence match — update the speaker's embedding (running average)
                self._update_speaker_embedding(
                    session, best_match, embedding, recording_duration
                )
                session.commit()

                logger.info(
                    f"Matched {embedding.speaker} → {best_match.name or best_match.label} "
                    f"(sim={best_similarity:.4f}, confidence=high)"
                )
                return SpeakerMatch(
                    speaker_id=str(best_match.id),
                    name=best_match.name,
                    similarity=best_similarity,
                    confidence="high",
                    is_new=False,
                )

            elif best_match and best_similarity >= self.candidate_threshold:
                # Medium confidence — possible match but not certain
                logger.info(
                    f"Candidate match {embedding.speaker} → {best_match.name or best_match.label} "
                    f"(sim={best_similarity:.4f}, confidence=medium)"
                )
                return SpeakerMatch(
                    speaker_id=str(best_match.id),
                    name=best_match.name,
                    similarity=best_similarity,
                    confidence="medium",
                    is_new=False,
                )

            else:
                # No match — create new speaker
                new_speaker = self._create_speaker(
                    session, embedding, recording_duration
                )
                session.commit()

                logger.info(
                    f"New speaker created: {new_speaker.label} "
                    f"(best sim={best_similarity:.4f} < {self.candidate_threshold})"
                )
                return SpeakerMatch(
                    speaker_id=str(new_speaker.id),
                    name=None,
                    similarity=best_similarity if best_match else 0.0,
                    confidence="low",
                    is_new=True,
                )

        finally:
            session.close()

    def _update_speaker_embedding(
        self,
        session,
        speaker: Speaker,
        new_embedding: SpeakerEmbedding,
        duration: float,
    ):
        """Update speaker's embedding with running weighted average."""
        old_emb = np.array(speaker.embedding, dtype=np.float32)
        new_emb = new_embedding.embedding

        # Weight by number of samples
        old_weight = speaker.embedding_count
        new_weight = 1
        total_weight = old_weight + new_weight

        updated_emb = (old_emb * old_weight + new_emb * new_weight) / total_weight
        # Re-normalize
        updated_emb = updated_emb / np.linalg.norm(updated_emb)

        speaker.embedding = updated_emb.tolist()
        speaker.embedding_count = total_weight
        speaker.total_speech_seconds += duration
        speaker.last_seen = datetime.datetime.utcnow()

    def _create_speaker(
        self,
        session,
        embedding: SpeakerEmbedding,
        duration: float,
    ) -> Speaker:
        """Create a new speaker profile."""
        # Count existing speakers for labeling
        count = session.query(Speaker).count()

        speaker = Speaker(
            label=f"person_{count:03d}",
            embedding=embedding.embedding.tolist(),
            embedding_count=1,
            total_speech_seconds=duration,
        )
        session.add(speaker)
        return speaker

    def get_all_speakers(self) -> list[dict]:
        """List all known speakers."""
        session = get_session()
        try:
            speakers = session.query(Speaker).order_by(Speaker.first_seen).all()
            return [
                {
                    "id": str(s.id),
                    "name": s.name,
                    "label": s.label,
                    "is_owner": s.is_owner,
                    "total_speech_seconds": s.total_speech_seconds,
                    "embedding_count": s.embedding_count,
                    "first_seen": s.first_seen.isoformat() if s.first_seen else None,
                    "last_seen": s.last_seen.isoformat() if s.last_seen else None,
                }
                for s in speakers
            ]
        finally:
            session.close()

    def name_speaker(self, speaker_id: str, name: str) -> bool:
        """Assign a human name to a speaker."""
        session = get_session()
        try:
            speaker = session.query(Speaker).filter_by(id=speaker_id).first()
            if speaker:
                speaker.name = name
                session.commit()
                logger.info(f"Named speaker {speaker.label} → {name}")
                return True
            return False
        finally:
            session.close()

    def set_owner(self, speaker_id: str) -> bool:
        """Mark a speaker as the device owner."""
        session = get_session()
        try:
            # Unset any existing owner
            session.query(Speaker).filter_by(is_owner=True).update({"is_owner": False})
            speaker = session.query(Speaker).filter_by(id=speaker_id).first()
            if speaker:
                speaker.is_owner = True
                session.commit()
                logger.info(f"Set owner: {speaker.name or speaker.label}")
                return True
            return False
        finally:
            session.close()
