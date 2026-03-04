"""Aura CLI — process audio files through the pipeline."""

import logging
import sys
from pathlib import Path

import click


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
def cli():
    """Aura — Audio processing for memory augmentation."""
    pass


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None, help="Output directory")
@click.option("--no-extract", is_flag=True, help="Skip knowledge extraction (LLM)")
@click.option("--owner", default="SPEAKER_00", help="Owner speaker label")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def process(audio_path: str, output: str | None, no_extract: bool, owner: str, verbose: bool):
    """Process an audio file through the full pipeline.

    Runs: VAD → Transcription → Diarization → Speaker ID → Alignment → Extraction
    """
    setup_logging(verbose)

    from src.pipeline.processor import AudioProcessor

    output_dir = output or f"data/processed/{Path(audio_path).stem}"

    processor = AudioProcessor(
        enable_extraction=not no_extract,
        owner_speaker=owner,
    )

    result = processor.process(audio_path, output_dir=output_dir)

    # Print summary
    click.echo(f"\n{'=' * 50}")
    click.echo(f"✅ Processing complete")
    click.echo(f"{'=' * 50}")
    click.echo(f"  File: {audio_path}")
    click.echo(f"  Duration: {result.duration:.1f}s ({result.duration / 60:.1f}min)")
    click.echo(f"  Speech: {result.speech_ratio * 100:.0f}%")
    click.echo(f"  Speakers: {result.num_speakers}")
    click.echo(f"  Segments: {result.num_segments}")
    click.echo(f"  Time: {result.processing_time}s")
    click.echo(f"  Output: {output_dir}/")
    if result.errors:
        click.echo(f"  ⚠️ Errors: {result.errors}")

    click.echo(f"\n--- Transcript ---\n")
    click.echo(result.transcript_text)

    if result.extraction:
        click.echo(f"\n--- Knowledge ---")
        ex = result.extraction
        if ex.get("summary"):
            click.echo(f"\nSummary: {ex['summary']}")
        if ex.get("topics"):
            click.echo(f"Topics: {', '.join(ex['topics'])}")
        if ex.get("people_mentioned"):
            click.echo(f"\nPeople:")
            for p in ex["people_mentioned"]:
                click.echo(f"  • {p['name']}")
                for f in p.get("facts", []):
                    click.echo(f"    - {f}")
        if ex.get("commitments"):
            click.echo(f"\nCommitments:")
            for c in ex["commitments"]:
                click.echo(f"  • {c['description']} (by {c['speaker']})")
        if ex.get("facts"):
            click.echo(f"\nFacts:")
            for f in ex["facts"]:
                click.echo(f"  • {f['subject']}: {f['fact']}")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True)
def vad(audio_path: str, verbose: bool):
    """Run only Voice Activity Detection on an audio file."""
    setup_logging(verbose)

    from src.pipeline.vad import VoiceActivityDetector

    detector = VoiceActivityDetector()
    segments = detector.detect(audio_path)

    total_speech = sum(s.duration for s in segments)
    click.echo(f"\n{len(segments)} speech segments, {total_speech:.1f}s total speech")
    for seg in segments:
        click.echo(f"  {seg.start:7.1f}s → {seg.end:7.1f}s  ({seg.duration:.1f}s)")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True)
def transcribe(audio_path: str, verbose: bool):
    """Run only transcription on an audio file."""
    setup_logging(verbose)

    from src.pipeline.transcribe import Transcriber

    transcriber = Transcriber()
    transcript = transcriber.transcribe(audio_path)

    click.echo(f"\nLanguage: {transcript.language}")
    click.echo(f"Duration: {transcript.duration:.1f}s")
    click.echo(f"Segments: {len(transcript.segments)}")
    click.echo(f"\n{transcript.text}")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True)
def diarize(audio_path: str, verbose: bool):
    """Run only speaker diarization on an audio file."""
    setup_logging(verbose)

    from src.pipeline.diarize import Diarizer

    diarizer = Diarizer()
    result = diarizer.diarize(audio_path)

    click.echo(f"\n{result.num_speakers} speakers, {len(result.turns)} turns")
    for turn in result.turns:
        click.echo(f"  {turn.start:7.1f}s → {turn.end:7.1f}s  {turn.speaker} ({turn.duration:.1f}s)")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True)
def speakers(audio_path: str, verbose: bool):
    """Extract speaker embeddings from an audio file."""
    setup_logging(verbose)

    from src.pipeline.diarize import Diarizer
    from src.pipeline.speaker_embed import SpeakerEmbedder

    diarizer = Diarizer()
    embedder = SpeakerEmbedder()

    diarization = diarizer.diarize(audio_path)
    embeddings = embedder.extract_per_speaker(audio_path, diarization.turns)

    click.echo(f"\n{len(embeddings)} speaker embeddings extracted:")
    for emb in embeddings:
        click.echo(
            f"  {emb.speaker}: dim={emb.embedding.shape[0]}, "
            f"speech={emb.duration:.1f}s"
        )

    # Show pairwise similarities
    if len(embeddings) > 1:
        click.echo(f"\nPairwise cosine similarities:")
        for i, a in enumerate(embeddings):
            for b in embeddings[i + 1 :]:
                sim = SpeakerEmbedder.cosine_similarity(a.embedding, b.embedding)
                click.echo(f"  {a.speaker} ↔ {b.speaker}: {sim:.4f}")


if __name__ == "__main__":
    cli()
