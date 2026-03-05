"""File watcher — monitors upload directory for new audio files.

Polls the upload directory for new audio files and queues them for processing.
Designed for the batch workflow: device records all day, uploads at night,
watcher picks up new files and processes them sequentially.
"""

import hashlib
import logging
import time
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".webm"}


def file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def find_unprocessed(
    upload_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> list[Path]:
    """Find audio files in upload_dir that haven't been processed yet.

    A file is considered processed if a directory with its stem name
    exists in processed_dir.

    Args:
        upload_dir: Directory to scan for audio files.
        processed_dir: Directory where processed results are stored.

    Returns:
        List of unprocessed audio file paths, sorted by modification time.
    """
    upload_dir = upload_dir or settings.upload_dir
    processed_dir = processed_dir or settings.processed_dir

    upload_dir = Path(upload_dir)
    processed_dir = Path(processed_dir)

    if not upload_dir.exists():
        logger.warning(f"Upload directory does not exist: {upload_dir}")
        return []

    audio_files = [
        f for f in upload_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]

    unprocessed = []
    for f in audio_files:
        result_dir = processed_dir / f.stem
        full_result = result_dir / f"{f.stem}_full.json"
        if not full_result.exists():
            unprocessed.append(f)

    # Sort by modification time (oldest first)
    unprocessed.sort(key=lambda f: f.stat().st_mtime)

    if unprocessed:
        logger.info(f"Found {len(unprocessed)} unprocessed audio files")
    return unprocessed


def process_batch(
    upload_dir: Path | None = None,
    processed_dir: Path | None = None,
    enable_extraction: bool = True,
    enable_db: bool = False,
) -> list[dict]:
    """Process all unprocessed audio files in the upload directory.

    Args:
        upload_dir: Directory to scan for audio files.
        processed_dir: Directory to save results.
        enable_extraction: Whether to run LLM extraction.
        enable_db: Whether to persist results to database.

    Returns:
        List of result summaries.
    """
    from src.pipeline.processor import AudioProcessor

    upload_dir = Path(upload_dir or settings.upload_dir)
    processed_dir = Path(processed_dir or settings.processed_dir)

    files = find_unprocessed(upload_dir, processed_dir)
    if not files:
        logger.info("No unprocessed files found.")
        return []

    logger.info(f"Processing {len(files)} files...")

    processor = AudioProcessor(enable_extraction=enable_extraction)
    results = []

    for i, audio_path in enumerate(files, 1):
        logger.info(f"\n[{i}/{len(files)}] {audio_path.name}")
        output_dir = processed_dir / audio_path.stem

        try:
            result = processor.process(audio_path, output_dir=output_dir)
            results.append({
                "file": audio_path.name,
                "status": "done",
                "duration": result.duration,
                "speakers": result.num_speakers,
                "segments": result.num_segments,
                "processing_time": result.processing_time,
                "errors": result.errors,
            })
        except Exception as e:
            logger.error(f"Failed to process {audio_path.name}: {e}")
            results.append({
                "file": audio_path.name,
                "status": "failed",
                "error": str(e),
            })

    # Summary
    done = sum(1 for r in results if r["status"] == "done")
    failed = sum(1 for r in results if r["status"] == "failed")
    logger.info(f"\nBatch complete: {done} succeeded, {failed} failed out of {len(files)} files")

    return results


def watch(
    upload_dir: Path | None = None,
    processed_dir: Path | None = None,
    poll_interval: int = 60,
    enable_extraction: bool = True,
):
    """Watch upload directory and process new files as they appear.

    Polls every `poll_interval` seconds. Designed to run as a background service.

    Args:
        upload_dir: Directory to watch.
        processed_dir: Output directory.
        poll_interval: Seconds between polls.
        enable_extraction: Whether to run LLM extraction.
    """
    from src.pipeline.processor import AudioProcessor

    upload_dir = Path(upload_dir or settings.upload_dir)
    processed_dir = Path(processed_dir or settings.processed_dir)

    logger.info(f"Watching {upload_dir} for new audio files (poll every {poll_interval}s)")

    processor = AudioProcessor(enable_extraction=enable_extraction)

    while True:
        try:
            files = find_unprocessed(upload_dir, processed_dir)
            for audio_path in files:
                output_dir = processed_dir / audio_path.stem
                try:
                    logger.info(f"Processing new file: {audio_path.name}")
                    processor.process(audio_path, output_dir=output_dir)
                except Exception as e:
                    logger.error(f"Failed to process {audio_path.name}: {e}")
        except Exception as e:
            logger.error(f"Watcher error: {e}")

        time.sleep(poll_interval)
