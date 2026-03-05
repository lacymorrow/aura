"""SQLAlchemy models for Aura's persistent storage.

Tables:
- recordings: metadata about each audio file processed
- speakers: known speaker profiles with voice embeddings
- conversations: processed conversation sessions
- knowledge_entries: extracted facts, commitments, events
- people: identified people across conversations
"""

import datetime
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Recording(Base):
    """An audio file that was processed."""

    __tablename__ = "recordings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    file_hash = Column(String(64), unique=True, nullable=False, index=True)
    file_size_bytes = Column(Integer)
    duration_seconds = Column(Float, nullable=False)
    sample_rate = Column(Integer, default=16000)
    language = Column(String(10))
    speech_ratio = Column(Float)
    num_speakers = Column(Integer)
    num_segments = Column(Integer)
    processing_time_seconds = Column(Float)
    status = Column(String(20), default="pending", index=True)  # pending, processing, done, failed
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    processed_at = Column(DateTime)
    output_dir = Column(String(512))

    # Relationships
    conversations = relationship("Conversation", back_populates="recording")

    __table_args__ = (
        Index("ix_recordings_status_created", "status", "created_at"),
    )


class Speaker(Base):
    """A known speaker profile with persistent voiceprint."""

    __tablename__ = "speakers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(256))  # human-assigned name (null until identified)
    label = Column(String(50))  # auto-assigned label (e.g. "SPEAKER_00")
    is_owner = Column(Boolean, default=False, index=True)
    embedding = Column(ARRAY(Float))  # 192-dim voiceprint
    embedding_count = Column(Integer, default=1)  # how many samples averaged
    total_speech_seconds = Column(Float, default=0.0)
    first_seen = Column(DateTime, default=datetime.datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.datetime.utcnow)
    metadata_ = Column("metadata", JSON, default=dict)  # freeform attributes

    # Relationships
    conversation_speakers = relationship("ConversationSpeaker", back_populates="speaker")

    __table_args__ = (
        Index("ix_speakers_is_owner", "is_owner"),
    )


class Conversation(Base):
    """A processed conversation session."""

    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recording_id = Column(UUID(as_uuid=True), ForeignKey("recordings.id"), nullable=False)
    summary = Column(Text)
    topics = Column(ARRAY(String))
    sentiment = Column(String(20))  # positive, neutral, negative
    transcript_text = Column(Text)
    transcript_json = Column(JSON)
    extraction_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    recording = relationship("Recording", back_populates="conversations")
    speakers = relationship("ConversationSpeaker", back_populates="conversation")
    knowledge_entries = relationship("KnowledgeEntry", back_populates="conversation")

    __table_args__ = (
        Index("ix_conversations_created", "created_at"),
    )


class ConversationSpeaker(Base):
    """Junction table: which speakers appeared in which conversations."""

    __tablename__ = "conversation_speakers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    speaker_id = Column(UUID(as_uuid=True), ForeignKey("speakers.id"), nullable=False)
    diarization_label = Column(String(50))  # SPEAKER_00, SPEAKER_01, etc.
    speech_seconds = Column(Float)
    is_owner = Column(Boolean, default=False)

    # Relationships
    conversation = relationship("Conversation", back_populates="speakers")
    speaker = relationship("Speaker", back_populates="conversation_speakers")


class KnowledgeEntry(Base):
    """An extracted piece of knowledge (fact, commitment, event)."""

    __tablename__ = "knowledge_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    kind = Column(String(20), nullable=False, index=True)  # fact, commitment, event
    subject = Column(String(256))  # who/what this is about
    content = Column(Text, nullable=False)  # the actual knowledge
    confidence = Column(Float, default=0.8)
    speaker_label = Column(String(50))  # who said it
    deadline = Column(String(100))  # for commitments
    event_date = Column(String(100))  # for events
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    conversation = relationship("Conversation", back_populates="knowledge_entries")
    person_links = relationship("PersonKnowledge", back_populates="knowledge_entry")

    __table_args__ = (
        Index("ix_knowledge_kind_subject", "kind", "subject"),
    )


class Person(Base):
    """A person identified across conversations (may link to a Speaker)."""

    __tablename__ = "people"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(256), nullable=False, index=True)
    speaker_id = Column(UUID(as_uuid=True), ForeignKey("speakers.id"), nullable=True)
    relationship_to_owner = Column(String(256))
    facts = Column(JSON, default=list)  # accumulated facts as list of strings
    first_mentioned = Column(DateTime, default=datetime.datetime.utcnow)
    last_mentioned = Column(DateTime, default=datetime.datetime.utcnow)
    mention_count = Column(Integer, default=1)
    metadata_ = Column("metadata", JSON, default=dict)

    # Relationships
    knowledge_links = relationship("PersonKnowledge", back_populates="person")


class PersonKnowledge(Base):
    """Links people to knowledge entries."""

    __tablename__ = "person_knowledge"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_id = Column(UUID(as_uuid=True), ForeignKey("people.id"), nullable=False)
    knowledge_entry_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_entries.id"), nullable=False)

    person = relationship("Person", back_populates="knowledge_links")
    knowledge_entry = relationship("KnowledgeEntry", back_populates="person_links")
