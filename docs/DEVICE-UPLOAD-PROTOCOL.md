# Device Upload Protocol

## Overview

The Aura wearable (ESP32-C6) records audio all day and syncs to the processing server (Otto) over WiFi. USB is charging only.

## Architecture

```
ESP32-C6 (wearable)
    │
    │  WiFi (home network)
    │  HTTP POST multipart/form-data
    ▼
Otto (192.168.1.72 / 100.111.32.113)
    │  Port 8080 — Ingest API (FastAPI)
    │
    ├─→ /data/uploads/        ← incoming audio files
    │       └── *.meta.json   ← metadata sidecars
    │
    ├─→ Watcher service       ← polls uploads/ for new files
    │       └── Pipeline      ← VAD → transcribe → diarize → extract
    │
    └─→ PostgreSQL            ← results persisted
```

## Ingest API

Base URL: `http://<server>:8080`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/ingest/status` | Queue depth, disk space |
| POST | `/ingest` | Upload complete audio file |
| POST | `/ingest/chunk` | Upload one chunk (resumable) |
| POST | `/ingest/complete` | Signal all chunks uploaded |
| GET | `/ingest/device/{id}` | List uploads from a device |

### Simple Upload (recommended for files < 1MB)

```
POST /ingest
Content-Type: multipart/form-data

Fields:
  file          — audio file (required)
  device_id     — unique device ID, e.g. MAC address (required)
  timestamp     — ISO 8601 recording start time (optional)
  duration_seconds — recording duration in seconds (optional)
  sample_rate   — default 16000
  channels      — default 1
  bit_depth     — default 16
  format        — "wav", "opus", "raw", "pcm", "adpcm" (default "wav")
```

Response (200):
```json
{
  "status": "ok",
  "filename": "AA:BB:CC:DD:EE:FF_2026-03-05_22-30-00_a1b2c3d4.wav",
  "hash": "a1b2c3d4e5f6g7h8",
  "size_bytes": 1234567
}
```

Duplicate response (200):
```json
{
  "status": "duplicate",
  "message": "File already uploaded",
  "filename": "...",
  "hash": "..."
}
```

### Chunked Upload (for files > 1MB or limited RAM)

The ESP32-C6 has ~320KB SRAM. For large recordings, split the file into chunks (e.g., 32KB or 64KB) and upload sequentially.

**Step 1: Upload chunks**
```
POST /ingest/chunk
Content-Type: multipart/form-data

Fields:
  file          — chunk data (required)
  device_id     — unique device ID (required)
  session_id    — unique per-recording (required, generate on device)
  chunk_index   — 0-based index (required)
  total_chunks  — total expected chunks (required)
  timestamp     — ISO 8601 recording start (optional, send with first chunk)
  sample_rate   — default 16000
  channels      — default 1
  bit_depth     — default 16
  format        — default "wav"
```

Response:
```json
{
  "status": "ok",
  "chunk_index": 0,
  "chunks_received": 1,
  "total_chunks": 100,
  "complete": false
}
```

**Step 2: Signal completion**
```
POST /ingest/complete
Content-Type: multipart/form-data

Fields:
  device_id     — (required)
  session_id    — (required)
```

Response:
```json
{
  "status": "ok",
  "filename": "...",
  "hash": "...",
  "size_bytes": 12345678
}
```

If chunks are missing:
```json
{
  "status": "incomplete",
  "chunks_received": 98,
  "total_chunks": 100,
  "missing_chunks": [42, 87]
}
```

### Check Previous Uploads

Before uploading, the device can check what's already been received:

```
GET /ingest/device/{device_id}?limit=50
```

Response:
```json
{
  "device_id": "AA:BB:CC:DD:EE:FF",
  "uploads": [
    {
      "filename": "...",
      "hash": "a1b2c3d4e5f6g7h8",
      "timestamp": "2026-03-05T22:30:00Z",
      "size_bytes": 1234567,
      "received_at": "2026-03-05T22:35:00Z"
    }
  ],
  "count": 1
}
```

The device can use the `hash` field to skip re-uploading files it has already sent.

## Audio Format Recommendations

| Format | Size/hr (16kHz mono) | ESP32 Complexity | Notes |
|--------|---------------------|------------------|-------|
| Raw PCM | ~115MB | None | Simplest, largest |
| WAV | ~115MB | Trivial header | Standard, pipeline handles natively |
| ADPCM | ~29MB | Low (IMA ADPCM) | 4:1 compression, easy to decode |
| Opus | ~3-5MB | High | Best compression, needs library |

**Recommendation:** Start with WAV for simplicity. Switch to ADPCM or Opus once the basic flow works. The pipeline's VAD step accepts all formats via ffmpeg conversion.

## ESP32-C6 Implementation Notes

### Storage
- Record to external SPI flash or SD card (internal flash is limited)
- Name files with incrementing index or timestamp: `rec_000001.wav`
- Keep a simple index file tracking which recordings have been uploaded

### WiFi Sync Trigger
Options (implement one or more):
1. **On charger + WiFi** — best for nightly batch sync
2. **Periodic** — every N hours while WiFi is available
3. **Manual** — button press to trigger sync

### Upload Flow (pseudocode)
```c
// 1. Connect to WiFi
wifi_connect(SSID, PASSWORD);

// 2. Check server health
http_get("http://otto:8080/");

// 3. Get list of already-uploaded files
http_get("http://otto:8080/ingest/device/MY_DEVICE_ID");

// 4. For each unsynced recording:
for (file in unsynced_recordings) {
    if (file.size < MAX_SINGLE_UPLOAD) {
        // Simple upload
        http_post_multipart("http://otto:8080/ingest", file, metadata);
    } else {
        // Chunked upload
        session_id = generate_uuid();
        for (chunk in file.chunks(CHUNK_SIZE)) {
            http_post_multipart("http://otto:8080/ingest/chunk", chunk, metadata);
        }
        http_post("http://otto:8080/ingest/complete", session_id);
    }
    mark_as_uploaded(file);
}

// 5. Disconnect WiFi (save power)
wifi_disconnect();
```

### Recommended Chunk Size
- **32KB** — safe for ESP32 with limited free heap
- **64KB** — faster upload, needs more RAM
- Match your HTTP client buffer size

### Error Handling
- If upload fails mid-transfer, retry from the failed chunk
- The `/ingest/complete` endpoint reports missing chunks
- Server deduplicates by content hash, so retries are safe

## Server Setup

The ingest server runs as a Docker container on Otto:

```bash
# Start all services (DB + watcher + ingest)
docker compose up -d

# Or just the ingest server
docker compose up -d ingest

# Check status
curl http://localhost:8080/ingest/status

# Test upload
curl -X POST http://localhost:8080/ingest \
  -F "file=@test.wav" \
  -F "device_id=test-device" \
  -F "format=wav"
```

The watcher service automatically picks up files from `/data/uploads/` and processes them through the full pipeline.

## Network Configuration

Otto's addresses:
- Tailscale: `100.111.32.113`
- LAN: `192.168.1.72`
- Hostname: `otto`

The ESP32 should use the LAN IP (`192.168.1.72`) for local network uploads. For remote access (if needed later), use Tailscale.

Port: **8080** (mapped in docker-compose)
