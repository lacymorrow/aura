"""Aura Ingest API — receives audio uploads from ESP32 devices over WiFi.

Endpoints:
    POST /ingest          — Upload an audio file (multipart/form-data)
    POST /ingest/chunk    — Upload a chunk for resumable uploads
    POST /ingest/complete — Signal that all chunks for a session are uploaded
    GET  /ingest/status   — Server health check + queue depth
    GET  /ingest/device/:id — Check what files a device has already uploaded
"""

import hashlib
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aura Ingest API",
    description="Receives audio uploads from Aura wearable devices",
    version="0.1.0",
)

# Directories
INCOMING_DIR = settings.upload_dir
CHUNKS_DIR = settings.data_dir / "chunks"
MANIFEST_DIR = settings.data_dir / "manifests"

for d in [INCOMING_DIR, CHUNKS_DIR, MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------- Health ----------


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "aura-ingest", "version": "0.1.0"}


@app.get("/ingest/status")
async def ingest_status():
    """Server status: queue depth, disk space, uptime."""
    pending = list(INCOMING_DIR.glob("*.wav")) + list(INCOMING_DIR.glob("*.opus")) + list(INCOMING_DIR.glob("*.raw"))
    chunks_in_progress = len(list(CHUNKS_DIR.iterdir())) if CHUNKS_DIR.exists() else 0

    disk = shutil.disk_usage(str(settings.data_dir))

    return {
        "status": "ok",
        "pending_files": len(pending),
        "chunks_in_progress": chunks_in_progress,
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "disk_total_gb": round(disk.total / (1024**3), 1),
        "accepted_formats": ["wav", "opus", "raw", "pcm", "adpcm"],
    }


# ---------- Simple upload (small files, single POST) ----------


@app.post("/ingest")
async def ingest_upload(
    file: UploadFile = File(...),
    device_id: str = Form(...),
    timestamp: Optional[str] = Form(None),
    duration_seconds: Optional[float] = Form(None),
    sample_rate: Optional[int] = Form(16000),
    channels: Optional[int] = Form(1),
    bit_depth: Optional[int] = Form(16),
    format: Optional[str] = Form("wav"),
):
    """Upload a complete audio file.

    For ESP32: send as multipart/form-data with the audio file
    and metadata fields.

    Args:
        file: The audio file
        device_id: Unique device identifier (e.g., MAC address)
        timestamp: ISO 8601 recording start time (device clock)
        duration_seconds: Recording duration in seconds
        sample_rate: Audio sample rate (default 16000)
        channels: Number of channels (default 1, mono)
        bit_depth: Bits per sample (default 16)
        format: Audio format: wav, opus, raw, pcm, adpcm
    """
    # Validate
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id required")

    # Generate filename: {device_id}_{timestamp}_{hash}.{ext}
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    ts_safe = ts.replace(":", "-").replace("T", "_")[:19]
    ext = format if format in ("wav", "opus") else "raw"

    # Read file content
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Hash for dedup
    file_hash = hashlib.sha256(content).hexdigest()[:16]

    filename = f"{device_id}_{ts_safe}_{file_hash}.{ext}"
    dest = INCOMING_DIR / filename

    # Check for duplicate
    if dest.exists():
        return JSONResponse(
            status_code=200,
            content={
                "status": "duplicate",
                "message": "File already uploaded",
                "filename": filename,
                "hash": file_hash,
            },
        )

    # Write file
    dest.write_bytes(content)

    # Write metadata sidecar
    metadata = {
        "device_id": device_id,
        "timestamp": ts,
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "channels": channels,
        "bit_depth": bit_depth,
        "format": format,
        "file_hash": file_hash,
        "filename": filename,
        "size_bytes": len(content),
        "received_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = INCOMING_DIR / f"{filename}.meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    logger.info(f"Received {filename} ({len(content)} bytes) from device {device_id}")

    return {
        "status": "ok",
        "filename": filename,
        "hash": file_hash,
        "size_bytes": len(content),
    }


# ---------- Chunked upload (large files, resumable) ----------


@app.post("/ingest/chunk")
async def ingest_chunk(
    file: UploadFile = File(...),
    device_id: str = Form(...),
    session_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    timestamp: Optional[str] = Form(None),
    sample_rate: Optional[int] = Form(16000),
    channels: Optional[int] = Form(1),
    bit_depth: Optional[int] = Form(16),
    format: Optional[str] = Form("wav"),
):
    """Upload a single chunk of a larger recording.

    For ESP32 with limited RAM: split the recording into chunks
    (e.g., 64KB each) and upload sequentially.

    Args:
        file: Chunk data
        device_id: Unique device identifier
        session_id: Unique upload session (generated by device per recording)
        chunk_index: 0-based chunk index
        total_chunks: Total number of chunks expected
        timestamp: ISO 8601 recording start time
        sample_rate: Audio sample rate
        channels: Number of channels
        bit_depth: Bits per sample
        format: Audio format
    """
    session_dir = CHUNKS_DIR / f"{device_id}_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    chunk_path = session_dir / f"chunk_{chunk_index:06d}.bin"
    chunk_path.write_bytes(content)

    # Save session metadata (overwrite each time — last chunk wins)
    meta = {
        "device_id": device_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "total_chunks": total_chunks,
        "sample_rate": sample_rate,
        "channels": channels,
        "bit_depth": bit_depth,
        "format": format,
        "last_chunk_received": chunk_index,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    received = len(list(session_dir.glob("chunk_*.bin")))

    logger.info(
        f"Chunk {chunk_index}/{total_chunks - 1} for session {session_id} "
        f"({len(content)} bytes, {received}/{total_chunks} received)"
    )

    return {
        "status": "ok",
        "chunk_index": chunk_index,
        "chunks_received": received,
        "total_chunks": total_chunks,
        "complete": received >= total_chunks,
    }


@app.post("/ingest/complete")
async def ingest_complete(
    device_id: str = Form(...),
    session_id: str = Form(...),
):
    """Signal that all chunks for a session have been uploaded.

    Reassembles chunks into a single file and moves to incoming/.
    """
    session_dir = CHUNKS_DIR / f"{device_id}_{session_id}"
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=400, detail="Session metadata missing")

    meta = json.loads(meta_path.read_text())
    total = meta["total_chunks"]

    # Verify all chunks present
    chunks = sorted(session_dir.glob("chunk_*.bin"))
    if len(chunks) < total:
        missing = set(range(total)) - {int(c.stem.split("_")[1]) for c in chunks}
        return JSONResponse(
            status_code=400,
            content={
                "status": "incomplete",
                "chunks_received": len(chunks),
                "total_chunks": total,
                "missing_chunks": sorted(missing)[:20],
            },
        )

    # Reassemble
    ts = meta.get("timestamp", datetime.now(timezone.utc).isoformat())
    ts_safe = ts.replace(":", "-").replace("T", "_")[:19]
    ext = meta.get("format", "raw")
    if ext not in ("wav", "opus"):
        ext = "raw"

    assembled = bytearray()
    for chunk in chunks:
        assembled.extend(chunk.read_bytes())

    file_hash = hashlib.sha256(assembled).hexdigest()[:16]
    filename = f"{device_id}_{ts_safe}_{file_hash}.{ext}"
    dest = INCOMING_DIR / filename

    dest.write_bytes(assembled)

    # Write metadata sidecar
    meta["filename"] = filename
    meta["file_hash"] = file_hash
    meta["size_bytes"] = len(assembled)
    meta["received_at"] = datetime.now(timezone.utc).isoformat()
    meta["reassembled_from_chunks"] = len(chunks)
    (INCOMING_DIR / f"{filename}.meta.json").write_text(json.dumps(meta, indent=2))

    # Clean up chunks
    shutil.rmtree(session_dir)

    logger.info(
        f"Reassembled {filename} ({len(assembled)} bytes) "
        f"from {len(chunks)} chunks"
    )

    return {
        "status": "ok",
        "filename": filename,
        "hash": file_hash,
        "size_bytes": len(assembled),
    }


# ---------- Device query ----------


@app.get("/ingest/device/{device_id}")
async def device_uploads(device_id: str, limit: int = 50):
    """List recent uploads from a specific device.

    The ESP32 can call this after connecting to WiFi to check
    which files have already been uploaded successfully.
    """
    files = []
    for meta_file in sorted(INCOMING_DIR.glob(f"{device_id}_*.meta.json"), reverse=True)[:limit]:
        meta = json.loads(meta_file.read_text())
        files.append({
            "filename": meta.get("filename"),
            "hash": meta.get("file_hash"),
            "timestamp": meta.get("timestamp"),
            "size_bytes": meta.get("size_bytes"),
            "received_at": meta.get("received_at"),
        })

    return {
        "device_id": device_id,
        "uploads": files,
        "count": len(files),
    }
