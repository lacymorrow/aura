"""Knowledge extraction from labeled transcripts using LLMs.

Extracts structured information: people, facts, commitments, topics,
relationships, and events from conversation transcripts.
"""

import json
import logging
from dataclasses import dataclass, field

from src.config import settings
from src.pipeline.align import LabeledTranscript

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are analyzing a transcript from a personal audio recording device. The user wearing the device is labeled as the "owner" — all other speakers are people they're interacting with.

Extract structured knowledge from this conversation. Be precise and factual — only extract what is explicitly stated or strongly implied.

Transcript:
{transcript}

Owner speaker label: {owner_speaker}

Extract the following as JSON:

{{
  "summary": "2-3 sentence summary of the conversation",
  "topics": ["main topics discussed"],
  "people_mentioned": [
    {{
      "name": "name or reference used",
      "speaker_label": "which speaker label if they're in the conversation, or null if only mentioned",
      "facts": ["specific facts learned about this person"],
      "relationship_to_owner": "how they relate to the owner if apparent"
    }}
  ],
  "facts": [
    {{
      "subject": "who/what the fact is about",
      "fact": "the specific fact",
      "confidence": 0.0-1.0
    }}
  ],
  "commitments": [
    {{
      "speaker": "who made the commitment (owner or other)",
      "description": "what was promised/agreed to",
      "target": "who it's directed at",
      "deadline": "deadline if mentioned, else null"
    }}
  ],
  "events": [
    {{
      "name": "event name or description",
      "date": "date if mentioned, else null",
      "participants": ["who's involved"],
      "type": "past or upcoming"
    }}
  ],
  "sentiment": {{
    "overall": "positive/neutral/negative",
    "notable_moments": ["any significant emotional moments"]
  }}
}}

Return ONLY valid JSON. No markdown, no explanation."""


@dataclass
class ExtractedPerson:
    name: str
    speaker_label: str | None = None
    facts: list[str] = field(default_factory=list)
    relationship_to_owner: str | None = None


@dataclass
class ExtractedFact:
    subject: str
    fact: str
    confidence: float = 0.8


@dataclass
class ExtractedCommitment:
    speaker: str
    description: str
    target: str | None = None
    deadline: str | None = None


@dataclass
class ExtractedEvent:
    name: str
    date: str | None = None
    participants: list[str] = field(default_factory=list)
    type: str = "unknown"  # "past" or "upcoming"


@dataclass
class ExtractionResult:
    """Structured knowledge extracted from a conversation."""

    summary: str
    topics: list[str]
    people: list[ExtractedPerson]
    facts: list[ExtractedFact]
    commitments: list[ExtractedCommitment]
    events: list[ExtractedEvent]
    sentiment_overall: str
    sentiment_moments: list[str]
    raw_json: dict = field(default_factory=dict)


class KnowledgeExtractor:
    """LLM-based knowledge extraction from transcripts."""

    def __init__(
        self,
        provider: str = settings.llm_provider,
        model: str = settings.llm_model,
        api_key: str = settings.llm_api_key,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider."""
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "local":
            return self._call_local(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _call_local(self, prompt: str) -> str:
        """Call a local LLM via OpenAI-compatible API (e.g., ollama, vllm)."""
        from openai import OpenAI

        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content

    def extract(
        self,
        transcript: LabeledTranscript,
        owner_speaker: str = "SPEAKER_00",
    ) -> ExtractionResult:
        """Extract structured knowledge from a labeled transcript.

        Args:
            transcript: Speaker-labeled transcript.
            owner_speaker: Which speaker label corresponds to the device owner.

        Returns:
            ExtractionResult with extracted knowledge.
        """
        logger.info(
            f"Extracting knowledge from {len(transcript.segments)} segments..."
        )

        # Format transcript for the prompt
        formatted = "\n".join(
            f"[{seg.speaker}] ({seg.start:.1f}s-{seg.end:.1f}s): {seg.text}"
            for seg in transcript.segments
        )

        prompt = EXTRACTION_PROMPT.format(
            transcript=formatted,
            owner_speaker=owner_speaker,
        )

        raw_response = self._call_llm(prompt)

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re

            match = re.search(r"```(?:json)?\s*(.*?)```", raw_response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                logger.error(f"Failed to parse LLM response as JSON: {raw_response[:200]}")
                raise

        result = ExtractionResult(
            summary=data.get("summary", ""),
            topics=data.get("topics", []),
            people=[
                ExtractedPerson(
                    name=p.get("name", "Unknown"),
                    speaker_label=p.get("speaker_label"),
                    facts=p.get("facts", []),
                    relationship_to_owner=p.get("relationship_to_owner"),
                )
                for p in data.get("people_mentioned", [])
            ],
            facts=[
                ExtractedFact(
                    subject=f.get("subject", ""),
                    fact=f.get("fact", ""),
                    confidence=f.get("confidence", 0.8),
                )
                for f in data.get("facts", [])
            ],
            commitments=[
                ExtractedCommitment(
                    speaker=c.get("speaker", ""),
                    description=c.get("description", ""),
                    target=c.get("target"),
                    deadline=c.get("deadline"),
                )
                for c in data.get("commitments", [])
            ],
            events=[
                ExtractedEvent(
                    name=e.get("name", ""),
                    date=e.get("date"),
                    participants=e.get("participants", []),
                    type=e.get("type", "unknown"),
                )
                for e in data.get("events", [])
            ],
            sentiment_overall=data.get("sentiment", {}).get("overall", "neutral"),
            sentiment_moments=data.get("sentiment", {}).get("notable_moments", []),
            raw_json=data,
        )

        logger.info(
            f"Extraction complete: {len(result.people)} people, "
            f"{len(result.facts)} facts, {len(result.commitments)} commitments, "
            f"{len(result.events)} events"
        )

        return result
