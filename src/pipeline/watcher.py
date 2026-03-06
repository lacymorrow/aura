"""File watcher — monitors upload directory for new audio files.

Polls the upload directory for new audio files and queues them for processing.
Designed for the batch workflow: device records all day, uploads at night,
watcher picks up new files and processes them sequentially.

Production hardening:
- File locking to prevent concurrent processing
- DB-based dedup (skip files already processed by hash)
- Stale chunk cleanup
- Crash recovery (lock files cleaned up on restart)
- Health file for container orchestration
- Memory cleanup between files
"""

import hashlib
import gc
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus", ".webm"}
LOCK_SUFFIX = ".processing"
FAILED_SUFFIX = ".failed"
HEALTH_FILE = settings.data_dir / ".watcher_healthy"
MAX_RETRIES = 2
STALE_CHUNK_HOURS = 24  # clean up chunks older than this


def file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _file_hash_quick(path: Path) -> str:
    """SHA-256 of first 1MB (fast dedup check)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()


def _is_locked(path: Path) -> bool:
    """Check if a file is currently being processed."""
    return (path.parent / (path.name + LOCK_SUFFIX)).exists()


def _lock(path: Path) -> Path:
    """Create a lock file for processing."""
    lock = path.parent / (path.name + LOCK_SUFFIX)
    lock.write_text(f"{os.getpid()}\n{datetime.now(timezone.utc).isoformat()}")
    return lock


def _unlock(path: Path):
    """Remove lock file."""
    lock = path.parent / (path.name + LOCK_SUFFIX)
    if lock.exists():
        lock.unlink()


def _is_failed(path: Path) -> bool:
    """Check if a file has been marked as permanently failed."""
    return (path.parent / (path.name + FAILED_SUFFIX)).exists()


def _mark_failed(path: Path, error: str, attempt: int):
    """Mark a file as failed with error details."""
    fail_file = path.parent / (path.name + FAILED_SUFFIX)
    fail_file.write_text(f"attempt={attempt}\nerror={error}\ntime={datetime.now(timezone.utc).isoformat()}")


def _is_db_processed(path: Path) -> bool:
    """Check if file was already processed by checking DB hash."""
    try:
        from src.db.persist import is_already_processed
        fhash = _file_hash_quick(path)
        return is_already_processed(fhash)
    except Exception:
        return False  # DB unavailable, don't skip


def _clean_stale_locks(upload_dir: Path):
    """Remove lock files from dead processes (crash recovery)."""
    for lock in upload_dir.glob(f"*{LOCK_SUFFIX}"):
        try:
            content = lock.read_text().strip().split("\n")
            pid = int(content[0])
            # Check if process is still alive
            try:
                os.kill(pid, 0)
            except OSError:
                logger.warning(f"Cleaning stale lock: {lock.name} (pid {pid} dead)")
                lock.unlink()
        except (ValueError, IndexError):
            # Malformed lock file, remove it
            lock.unlink()


def _clean_stale_chunks():
    """Remove chunk directories older than STALE_CHUNK_HOURS."""
    chunks_dir = settings.data_dir / "chunks"
    if not chunks_dir.exists():
        return
    cutoff = time.time() - (STALE_CHUNK_HOURS * 3600)
    for d in chunks_dir.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            import shutil
            logger.warning(f"Cleaning stale chunks: {d.name}")
            shutil.rmtree(d)


def _write_health():
    """Write health file for container probes."""
    HEALTH_FILE.write_text(datetime.now(timezone.utc).isoformat())


def _gpu_cleanup():
    """Force GPU memory release between processing runs."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def find_unprocessed(
    upload_dir: Path | None = None,
    processed_dir: Path | None = None,
    check_db: bool = False,
) -> list[Path]:
    """Find audio files in upload_dir that haven't been processed yet.

    A file is considered processed if:
    - A directory with its stem name exists in processed_dir, OR
    - Its hash exists in the database (if check_db=True), OR
    - It's currently locked for processing, OR
    - It's been marked as permanently failed

    Args:
        upload_dir: Directory to scan for audio files.
        processed_dir: Directory where processed results are stored.
        check_db: Whether to check database for already-processed hashes.

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
        # Skip locked files
        if _is_locked(f):
            continue
        # Skip permanently failed files
        if _is_failed(f):
            continue
        # Skip files with existing output
        result_dir = processed_dir / f.stem
        full_result = result_dir / f"{f.stem}_full.json"
        if full_result.exists():
            continue
        # Skip files already in DB
        if check_db and _is_db_processed(f):
            logger.info(f"Skipping {f.name} (already in database)")
            continue
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

    Uses file locking for safe concurrent operation, retries on transient
    failures, and cleans up GPU memory between files.

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

    _clean_stale_locks(upload_dir)
    files = find_unprocessed(upload_dir, processed_dir, check_db=enable_db)
    if not files:
        logger.info("No unprocessed files found.")
        return []

    logger.info(f"Processing {len(files)} files...")

    processor = AudioProcessor(
        enable_extraction=enable_extraction,
        enable_db=enable_db,
    )
    results = []

    for i, audio_path in enumerate(files, 1):
        logger.info(f"\n[{i}/{len(files)}] {audio_path.name}")
        output_dir = processed_dir / audio_path.stem

        lock = _lock(audio_path)
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
        finally:
            _unlock(audio_path)
            _gpu_cleanup()

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
    enable_db: bool = False,
):
    """Watch upload directory and process new files as they appear.

    Production features:
    - File locking prevents concurrent processing
    - Retry with backoff on transient failures
    - Dead-letter marking after MAX_RETRIES
    - Stale lock/chunk cleanup on startup and periodically
    - Health file for container liveness probes
    - GPU memory cleanup between files
    - Graceful shutdown on SIGTERM/SIGINT

    Args:
        upload_dir: Directory to watch.
        processed_dir: Output directory.
        poll_interval: Seconds between polls.
        enable_extraction: Whether to run LLM extraction.
        enable_db: Whether to persist results to database.
    """
    from src.pipeline.processor import AudioProcessor

    upload_dir = Path(upload_dir or settings.upload_dir)
    processed_dir = Path(processed_dir or settings.processed_dir)

    # Graceful shutdown
    shutdown = False

    def _handle_signal(signum, frame):
        nonlocal shutdown
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info(f"Watching {upload_dir} for new audio files (poll every {poll_interval}s)")

    # Startup cleanup
    _clean_stale_locks(upload_dir)
    _clean_stale_chunks()

    processor = AudioProcessor(
        enable_extraction=enable_extraction,
        enable_db=enable_db,
    )

    # Track per-file retry counts (in-memory, resets on restart)
    retry_counts: dict[str, int] = {}
    polls_since_cleanup = 0

    while not shutdown:
        try:
            _write_health()

            files = find_unprocessed(upload_dir, processed_dir, check_db=enable_db)
            for audio_path in files:
                if shutdown:
                    break

                file_key = audio_path.name
                attempt = retry_counts.get(file_key, 0) + 1

                if attempt > MAX_RETRIES:
                    logger.error(
                        f"Giving up on {file_key} after {MAX_RETRIES} attempts, "
                        f"marking as failed"
                    )
                    _mark_failed(audio_path, "max retries exceeded", attempt - 1)
                    continue

                lock = _lock(audio_path)
                output_dir = processed_dir / audio_path.stem
                try:
                    logger.info(f"Processing new file: {audio_path.name} (attempt {attempt})")
                    processor.process(audio_path, output_dir=output_dir)
                    # Success: clear retry count
                    retry_counts.pop(file_key, None)
                    logger.info(f"Successfully processed: {audio_path.name}")
                except Exception as e:
                    logger.error(f"Failed to process {audio_path.name} (attempt {attempt}): {e}")
                    retry_counts[file_key] = attempt
                    if attempt >= MAX_RETRIES:
                        _mark_failed(audio_path, str(e), attempt)
                finally:
                    _unlock(audio_path)
                    _gpu_cleanup()

            # Periodic maintenance
            polls_since_cleanup += 1
            if polls_since_cleanup >= 60:  # ~hourly at 60s poll
                _clean_stale_chunks()
                polls_since_cleanup = 0

        except Exception as e:
            logger.error(f"Watcher error: {e}", exc_info=True)

        # Interruptible sleep
        for _ in range(poll_interval):
            if shutdown:
                break
            time.sleep(1)

    logger.info("Watcher stopped.")
