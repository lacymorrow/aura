"""Persist pipeline results to the database.

Takes a ProcessingResult and writes recordings, conversations,
knowledge entries, and speaker profiles to postgres.
"""

import datetime
import logging
from pathlib import Path

from src.config import settings
from src.db.engine import get_session
from src.db.models import (
    Recording,
    Conversation,
    ConversationSpeaker,
    KnowledgeEntry,
    Person,
    PersonKnowledge,
)

logger = logging.getLogger(__name__)


def persist_result(result, speaker_matches: dict | None = None) -> str:
    """Save a ProcessingResult to the database.

    Args:
        result: ProcessingResult from the pipeline.
        speaker_matches: Optional dict mapping diarization labels to SpeakerMatch objects.

    Returns:
        Recording UUID as string.
    """
    session = get_session()
    try:
        # 1. Create recording
        recording = Recording(
            filename=Path(result.audio_path).name,
            file_hash=result.file_hash,
            duration_seconds=result.duration,
            language=result.language,
            speech_ratio=result.speech_ratio,
            num_speakers=result.num_speakers,
            num_segments=result.num_segments,
            processing_time_seconds=result.processing_time,
            status="done" if not result.errors else "partial",
            error_message="; ".join(result.errors) if result.errors else None,
            processed_at=datetime.datetime.utcnow(),
            output_dir=str(Path(result.audio_path).stem),
        )
        session.add(recording)
        session.flush()  # get recording.id

        # 2. Create conversation
        extraction = result.extraction or {}
        conversation = Conversation(
            recording_id=recording.id,
            summary=extraction.get("summary", ""),
            topics=extraction.get("topics", []),
            sentiment=extraction.get("sentiment", {}).get("overall", "neutral")
                if isinstance(extraction.get("sentiment"), dict) else "neutral",
            transcript_text=result.transcript_text,
            transcript_json=result.labeled_transcript,
            extraction_json=extraction,
        )
        session.add(conversation)
        session.flush()

        # 3. Link speakers to conversation
        if speaker_matches:
            for emb_data in result.speaker_embeddings:
                label = emb_data["speaker"]
                match = speaker_matches.get(label)
                if match:
                    # Support both SpeakerMatch objects and dicts
                    speaker_id = match.speaker_id if hasattr(match, "speaker_id") else match.get("speaker_id")
                    confidence = match.confidence if hasattr(match, "confidence") else match.get("confidence")
                    if speaker_id:
                        cs = ConversationSpeaker(
                            conversation_id=conversation.id,
                            speaker_id=speaker_id,
                            diarization_label=label,
                            speech_seconds=emb_data.get("duration", 0),
                            is_owner=confidence == "high" and label == "SPEAKER_00",
                        )
                        session.add(cs)

        # 4. Create knowledge entries + people
        if extraction:
            _persist_knowledge(session, conversation.id, extraction)

        session.commit()
        logger.info(f"Persisted recording {recording.id} to database")
        return str(recording.id)

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to persist result: {e}")
        raise
    finally:
        session.close()


def _persist_knowledge(session, conversation_id, extraction: dict):
    """Persist extracted knowledge entries and people."""

    # Facts
    for fact in extraction.get("facts", []):
        entry = KnowledgeEntry(
            conversation_id=conversation_id,
            kind="fact",
            subject=fact.get("subject", ""),
            content=fact.get("fact", ""),
            confidence=fact.get("confidence", 0.8),
        )
        session.add(entry)

    # Commitments
    for commit in extraction.get("commitments", []):
        entry = KnowledgeEntry(
            conversation_id=conversation_id,
            kind="commitment",
            subject=commit.get("speaker", ""),
            content=commit.get("description", ""),
            speaker_label=commit.get("speaker"),
            deadline=commit.get("deadline"),
        )
        session.add(entry)

    # Events
    for event in extraction.get("events", []):
        entry = KnowledgeEntry(
            conversation_id=conversation_id,
            kind="event",
            subject=event.get("name", ""),
            content=event.get("name", ""),
            event_date=event.get("date"),
            metadata_={"participants": event.get("participants", []), "type": event.get("type")},
        )
        session.add(entry)

    # People
    for person_data in extraction.get("people_mentioned", []):
        name = person_data.get("name", "Unknown")

        # Upsert person by name
        person = session.query(Person).filter_by(name=name).first()
        if person:
            # Update existing
            person.last_mentioned = datetime.datetime.utcnow()
            person.mention_count += 1
            # Merge facts (deduplicate)
            existing_facts = set(person.facts or [])
            new_facts = set(person_data.get("facts", []))
            person.facts = list(existing_facts | new_facts)
            if person_data.get("relationship_to_owner") and not person.relationship_to_owner:
                person.relationship_to_owner = person_data["relationship_to_owner"]
        else:
            person = Person(
                name=name,
                relationship_to_owner=person_data.get("relationship_to_owner"),
                facts=person_data.get("facts", []),
            )
            session.add(person)
            session.flush()

        # Link person to any fact knowledge entries about them
        fact_entries = session.query(KnowledgeEntry).filter_by(
            conversation_id=conversation_id,
            kind="fact",
            subject=name,
        ).all()
        for entry in fact_entries:
            link = PersonKnowledge(
                person_id=person.id,
                knowledge_entry_id=entry.id,
            )
            session.add(link)


def is_already_processed(file_hash: str) -> bool:
    """Check if a file has already been processed (by hash)."""
    session = get_session()
    try:
        exists = session.query(Recording).filter_by(file_hash=file_hash).first() is not None
        return exists
    finally:
        session.close()
