"""Aura CLI — process audio files through the pipeline."""

import logging
import sys
from pathlib import Path

import click


def setup_logging(verbose: bool = False, log_file: str | None = None):
    """Configure logging with optional file output."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stderr)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


@click.group()
def cli():
    """Aura — Audio processing for memory augmentation."""
    pass


# ---------- Pipeline commands ----------


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None, help="Output directory")
@click.option("--no-extract", is_flag=True, help="Skip knowledge extraction (LLM)")
@click.option("--db", is_flag=True, help="Persist results to PostgreSQL")
@click.option("--owner", default="SPEAKER_00", help="Owner speaker label")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
@click.option("--log-file", type=click.Path(), default=None, help="Log to file")
def process(audio_path: str, output: str | None, no_extract: bool, db: bool, owner: str, verbose: bool, log_file: str | None):
    """Process an audio file through the full pipeline.

    Runs: VAD → Transcription → Diarization → Speaker ID → Alignment → Extraction
    """
    setup_logging(verbose, log_file)

    from src.pipeline.processor import AudioProcessor

    output_dir = output or f"data/processed/{Path(audio_path).stem}"

    processor = AudioProcessor(
        enable_extraction=not no_extract,
        enable_db=db,
        owner_speaker=owner,
    )

    result = processor.process(audio_path, output_dir=output_dir)
    _print_result(result, output_dir)


@cli.command()
@click.option("-d", "--upload-dir", type=click.Path(), default=None, help="Upload directory")
@click.option("-o", "--output-dir", type=click.Path(), default=None, help="Output directory")
@click.option("--no-extract", is_flag=True, help="Skip knowledge extraction")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
@click.option("--log-file", type=click.Path(), default=None, help="Log to file")
def batch(upload_dir: str | None, output_dir: str | None, no_extract: bool, verbose: bool, log_file: str | None):
    """Process all unprocessed audio files in the upload directory."""
    setup_logging(verbose, log_file)

    from src.pipeline.watcher import process_batch

    results = process_batch(
        upload_dir=upload_dir,
        processed_dir=output_dir,
        enable_extraction=not no_extract,
    )

    click.echo(f"\n{'=' * 50}")
    click.echo(f"Batch processing complete")
    click.echo(f"{'=' * 50}")

    done = sum(1 for r in results if r["status"] == "done")
    failed = sum(1 for r in results if r["status"] == "failed")
    click.echo(f"  Processed: {done}")
    click.echo(f"  Failed: {failed}")

    if results:
        total_duration = sum(r.get("duration", 0) for r in results if r["status"] == "done")
        total_time = sum(r.get("processing_time", 0) for r in results if r["status"] == "done")
        click.echo(f"  Total audio: {total_duration / 60:.1f} min")
        click.echo(f"  Total processing time: {total_time:.1f}s")
        if total_duration > 0:
            click.echo(f"  Speed: {total_duration / total_time:.1f}x realtime")

    for r in results:
        status = "✅" if r["status"] == "done" else "❌"
        click.echo(f"\n  {status} {r['file']}")
        if r["status"] == "done":
            click.echo(f"     {r['duration']:.0f}s, {r['speakers']} speakers, {r['processing_time']:.1f}s")
            if r.get("errors"):
                click.echo(f"     ⚠️ {r['errors']}")
        else:
            click.echo(f"     Error: {r.get('error', 'unknown')}")


@cli.command()
@click.option("-d", "--upload-dir", type=click.Path(), default=None, help="Upload directory")
@click.option("-o", "--output-dir", type=click.Path(), default=None, help="Output directory")
@click.option("--interval", type=int, default=60, help="Poll interval in seconds")
@click.option("--no-extract", is_flag=True, help="Skip knowledge extraction")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
@click.option("--log-file", type=click.Path(), default=None, help="Log to file")
def watch(upload_dir: str | None, output_dir: str | None, interval: int, no_extract: bool, verbose: bool, log_file: str | None):
    """Watch upload directory and process new files automatically.

    Polls for new audio files and processes them as they appear.
    Designed to run as a background service.
    """
    setup_logging(verbose, log_file)

    from src.pipeline.watcher import watch as run_watcher

    click.echo(f"👁  Watching for new audio files (poll every {interval}s)")
    click.echo(f"   Upload dir: {upload_dir or 'data/uploads'}")
    click.echo(f"   Output dir: {output_dir or 'data/processed'}")
    click.echo(f"   Press Ctrl+C to stop\n")

    try:
        run_watcher(
            upload_dir=upload_dir,
            processed_dir=output_dir,
            poll_interval=interval,
            enable_extraction=not no_extract,
        )
    except KeyboardInterrupt:
        click.echo("\nWatcher stopped.")


# ---------- Individual stage commands ----------


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

    if len(embeddings) > 1:
        click.echo(f"\nPairwise cosine similarities:")
        for i, a in enumerate(embeddings):
            for b in embeddings[i + 1:]:
                sim = SpeakerEmbedder.cosine_similarity(a.embedding, b.embedding)
                click.echo(f"  {a.speaker} ↔ {b.speaker}: {sim:.4f}")


# ---------- Speaker commands ----------


@cli.group(name="speaker")
def speaker_group():
    """Speaker registry management."""
    pass


@speaker_group.command(name="list")
@click.option("-v", "--verbose", is_flag=True)
def speaker_list(verbose: bool):
    """List all known speakers in the registry."""
    setup_logging(verbose)

    from src.speakers.registry import SpeakerRegistry
    registry = SpeakerRegistry()
    speakers = registry.get_all_speakers()

    if not speakers:
        click.echo("No speakers registered yet.")
        return

    click.echo(f"\n{len(speakers)} known speakers:")
    click.echo(f"{'=' * 60}")
    for s in speakers:
        name = s["name"] or "(unnamed)"
        owner = " 👑" if s["is_owner"] else ""
        speech = s["total_speech_seconds"]
        samples = s["embedding_count"]
        click.echo(
            f"  {s['label']}: {name}{owner}"
            f"  |  {speech:.0f}s speech, {samples} samples"
            f"  |  id={s['id'][:8]}"
        )
        if s["first_seen"]:
            click.echo(f"           first: {s['first_seen'][:19]}  last: {s['last_seen'][:19]}")


@speaker_group.command(name="name")
@click.argument("speaker_id")
@click.argument("name")
@click.option("-v", "--verbose", is_flag=True)
def speaker_name(speaker_id: str, name: str, verbose: bool):
    """Assign a name to a speaker.

    SPEAKER_ID can be a full UUID or a prefix (first 8+ chars).
    """
    setup_logging(verbose)

    from src.speakers.registry import SpeakerRegistry
    registry = SpeakerRegistry()

    # Support partial IDs
    speaker_id = _resolve_speaker_id(speaker_id)
    if not speaker_id:
        click.echo("❌ Speaker not found.")
        return

    if registry.name_speaker(speaker_id, name):
        click.echo(f"✅ Named speaker → {name}")
    else:
        click.echo(f"❌ Speaker {speaker_id} not found.")


@speaker_group.command(name="set-owner")
@click.argument("speaker_id")
@click.option("-v", "--verbose", is_flag=True)
def speaker_set_owner(speaker_id: str, verbose: bool):
    """Mark a speaker as the device owner.

    SPEAKER_ID can be a full UUID or a prefix (first 8+ chars).
    """
    setup_logging(verbose)

    from src.speakers.registry import SpeakerRegistry
    registry = SpeakerRegistry()

    speaker_id = _resolve_speaker_id(speaker_id)
    if not speaker_id:
        click.echo("❌ Speaker not found.")
        return

    if registry.set_owner(speaker_id):
        click.echo(f"✅ Set as device owner.")
    else:
        click.echo(f"❌ Speaker {speaker_id} not found.")


def _resolve_speaker_id(partial_id: str) -> str | None:
    """Resolve a partial speaker ID to a full UUID."""
    from src.db.engine import get_session
    from src.db.models import Speaker

    session = get_session()
    try:
        # Try exact UUID match first (only if it looks like a full UUID)
        if len(partial_id) >= 32:
            speaker = session.query(Speaker).filter_by(id=partial_id).first()
            if speaker:
                return str(speaker.id)

        # Prefix match against all speakers
        speakers = session.query(Speaker).all()
        matches = [s for s in speakers if str(s.id).startswith(partial_id)]
        if len(matches) == 1:
            return str(matches[0].id)
        elif len(matches) > 1:
            click.echo(f"⚠️  Ambiguous ID prefix '{partial_id}' — matches {len(matches)} speakers")
            for m in matches:
                click.echo(f"    {str(m.id)[:8]} — {m.name or m.label}")
            return None
        return None
    finally:
        session.close()


# ---------- Database commands ----------


@cli.group()
def db():
    """Database management commands."""
    pass


@db.command(name="init")
@click.option("-v", "--verbose", is_flag=True)
def db_init(verbose: bool):
    """Initialize database tables."""
    setup_logging(verbose)

    from src.db.engine import init_db
    init_db()
    click.echo("✅ Database initialized.")


@db.command(name="reset")
@click.option("-v", "--verbose", is_flag=True)
@click.option("--yes", is_flag=True, help="Skip confirmation")
def db_reset(verbose: bool, yes: bool):
    """Drop and recreate all database tables. WARNING: destroys all data."""
    setup_logging(verbose)

    if not yes:
        click.confirm("This will destroy ALL data. Continue?", abort=True)

    from src.db.engine import get_engine, init_db
    from src.db.models import Base

    engine = get_engine()
    Base.metadata.drop_all(engine)
    click.echo("Dropped all tables.")
    Base.metadata.create_all(engine)
    click.echo("✅ Database reset complete.")


@db.command(name="status")
@click.option("-v", "--verbose", is_flag=True)
def db_status(verbose: bool):
    """Check database connection and table status."""
    setup_logging(verbose)

    from src.db.engine import get_engine
    from src.db.models import Recording, Speaker, Conversation, Person, KnowledgeEntry

    try:
        engine = get_engine()
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        from sqlalchemy.orm import Session as DBSession
        with DBSession(engine) as session:
            counts = {
                "recordings": session.query(Recording).count(),
                "speakers": session.query(Speaker).count(),
                "conversations": session.query(Conversation).count(),
                "people": session.query(Person).count(),
                "knowledge_entries": session.query(KnowledgeEntry).count(),
            }

        click.echo("✅ Database connected")
        for table, count in counts.items():
            click.echo(f"  {table}: {count}")
    except Exception as e:
        click.echo(f"❌ Database error: {e}")
        sys.exit(1)


# ---------- Status command ----------


@cli.command()
def status():
    """Show system status and configuration."""
    from src.config import settings

    click.echo("Aura Pipeline Status")
    click.echo(f"{'=' * 40}")
    click.echo(f"  Upload dir:    {settings.upload_dir}")
    click.echo(f"  Processed dir: {settings.processed_dir}")
    click.echo(f"  Whisper model: {settings.whisper_model}")
    click.echo(f"  Diarization:   {settings.diarization_model}")
    click.echo(f"  LLM provider:  {settings.llm_provider}")
    click.echo(f"  LLM model:     {settings.llm_model}")
    click.echo(f"  LLM key:       {'✅ set' if settings.llm_api_key else '❌ not set'}")
    click.echo(f"  HF token:      {'✅ set' if settings.hf_token else '❌ not set'}")
    click.echo(f"  Database URL:  {settings.database_url}")

    # Check for unprocessed files
    from src.pipeline.watcher import find_unprocessed
    unprocessed = find_unprocessed()
    click.echo(f"\n  Unprocessed files: {len(unprocessed)}")
    for f in unprocessed[:5]:
        click.echo(f"    • {f.name}")
    if len(unprocessed) > 5:
        click.echo(f"    ... and {len(unprocessed) - 5} more")

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            click.echo(f"\n  GPU: {gpu_name} ({gpu_mem:.0f}GB)")
        else:
            click.echo(f"\n  GPU: ❌ not available (CPU mode)")
    except ImportError:
        click.echo(f"\n  GPU: unknown (torch not loaded)")


# ---------- Helpers ----------


def _print_result(result, output_dir):
    """Print a processing result summary."""
    click.echo(f"\n{'=' * 50}")
    click.echo(f"✅ Processing complete")
    click.echo(f"{'=' * 50}")
    click.echo(f"  File: {result.audio_path}")
    click.echo(f"  Duration: {result.duration:.1f}s ({result.duration / 60:.1f}min)")
    click.echo(f"  Speech: {result.speech_ratio * 100:.0f}%")
    click.echo(f"  Speakers: {result.num_speakers}")
    click.echo(f"  Segments: {result.num_segments}")
    click.echo(f"  Time: {result.processing_time}s")
    click.echo(f"  Stages: {result.stage_times}")
    click.echo(f"  Output: {output_dir}/")
    if result.warnings:
        for w in result.warnings:
            click.echo(f"  ⚠️ {w}")
    if result.errors:
        click.echo(f"  ❌ Errors: {result.errors}")

    if result.speaker_matches:
        click.echo(f"\n--- Speaker Identification ---")
        for label, match in result.speaker_matches.items():
            name = match.get("name") or f"speaker_{match['speaker_id'][:8]}"
            new = " (NEW)" if match.get("is_new") else ""
            click.echo(f"  {label} → {name}  (confidence={match['confidence']}, sim={match['similarity']:.4f}){new}")

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


# ---------- Ingest server ----------


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", type=int, default=8080, help="Bind port")
@click.option("-v", "--verbose", is_flag=True)
def serve(host: str, port: int, verbose: bool):
    """Start the ingest HTTP server (receives uploads from ESP32 devices)."""
    setup_logging(verbose)
    import uvicorn
    click.echo(f"🌐 Starting Aura ingest server on {host}:{port}")
    uvicorn.run("src.api.ingest:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    cli()
