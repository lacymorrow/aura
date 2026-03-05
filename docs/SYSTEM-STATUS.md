# Aura System Status

Last updated: 2026-03-05

## Repositories

### `~/repo/aura` — Processing Pipeline (Python)
- **GitHub:** github.com/lacymorrow/aura
- **Purpose:** Audio processing, speaker identification, knowledge extraction, database persistence
- **Runtime:** Docker on Otto (Windows PC, RTX 3090)

### `~/repo/aura-web` — Web Dashboard (Next.js)
- **GitHub:** github.com/lacymorrow/shipkit (aura-web is a ShipKit-scaffolded app)
- **Purpose:** Dashboard UI for browsing conversations, people, speakers, knowledge
- **Runtime:** Local dev (port 3333), reads from Otto's postgres over Tailscale

## Infrastructure

### Otto (Processing Server)
- **OS:** Windows + Docker Desktop
- **GPU:** NVIDIA RTX 3090 (24GB VRAM)
- **Tailscale IP:** 100.111.32.113
- **LAN IP:** 192.168.1.72
- **SSH:** port 22 (user: io, pass: io)
- **PostgreSQL:** port 5432 (user: aura, pass: aura_secret, db: aura)
- **Ingest API:** port 8080 (FastAPI)

### Docker Services on Otto
| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| db | pgvector/pgvector:pg16 | 5432 | PostgreSQL + pgvector |
| aura | aura-aura:latest | — | One-shot pipeline (manual runs) |
| watcher | aura-aura:latest | — | Auto-process new uploads |
| ingest | aura-aura:latest | 8080 | HTTP API for device uploads |

### Wearable Device
- **MCU:** ESP32-C6 (RISC-V, WiFi 6, BLE 5.0, 320KB SRAM)
- **Audio:** External microphone recording to flash/external memory
- **Upload:** WiFi sync to Otto's ingest API (HTTP POST)
- **Charging:** USB (dumb charger, no data transfer)
- **Firmware:** Lacy is building this; separate agent will assist

## Pipeline Stages

```
Audio File
    │
    ▼
1. VAD (Silero) — detect speech segments, strip silence
    │
    ▼
2. Transcription (faster-whisper large-v3) — speech to text with timestamps
    │
    ▼
3. Diarization (pyannote 3.1) — "who spoke when" speaker segmentation
    │
    ▼
4. Speaker Embedding (ECAPA-TDNN) — 192-dim voiceprint per speaker
    │
    ▼
5. Speaker Registry Match — cosine similarity against known speakers
    │   Creates new speaker profile if unknown (threshold: 0.75)
    │
    ▼
6. Alignment — merge transcript + diarization into labeled turns
    │
    ▼
7. Knowledge Extraction (GPT-4o) — people, facts, commitments, events, topics
    │
    ▼
8. Database Persistence — recordings, conversations, speakers, people, knowledge_entries
```

## Database Schema

### Tables (PostgreSQL + pgvector)

**recordings**
- id (UUID PK), file_hash, file_path, file_name, duration_seconds, sample_rate, channels, num_speakers, language, processing_status, processing_time_seconds, stage_times (JSONB), created_at

**speakers**
- id (UUID PK), label, name, is_owner, embedding (vector(192)), total_speech_seconds, embedding_count, first_seen, last_seen, created_at

**conversations**
- id (UUID PK), recording_id (FK), summary, topics (varchar[]), transcript_text, transcript_json (JSONB), extraction_json (JSONB), sentiment, num_speakers, duration_seconds, language, created_at

**people**
- id (UUID PK), name, relationship_to_owner, facts (JSONB), first_mentioned, last_mentioned, mention_count, created_at

**knowledge_entries**
- id (UUID PK), conversation_id (FK), kind (fact/commitment/event/preference/relationship), subject, content, speaker_label, confidence, deadline, event_date, created_at

## CLI Commands

```bash
# Full pipeline
python -m src.cli process <audio.wav> [--db] [--no-extract] [-v]

# Individual stages
python -m src.cli vad <audio.wav>
python -m src.cli transcribe <audio.wav>
python -m src.cli diarize <audio.wav>
python -m src.cli speakers <audio.wav>

# Batch / watch
python -m src.cli batch [--upload-dir <dir>]
python -m src.cli watch [--interval 60]

# Speaker management
python -m src.cli speaker list
python -m src.cli speaker name <id> "Name"
python -m src.cli speaker set-owner <id>

# Database
python -m src.cli db init
python -m src.cli db reset [--yes]
python -m src.cli db status

# Ingest server
python -m src.cli serve [--port 8080]

# Status
python -m src.cli status
```

## Docker Commands (on Otto)

```bash
# Start everything
docker compose up -d

# Process a single file
docker compose run --rm aura process /app/data/uploads/test.wav --db

# Check DB
docker compose run --rm aura db status

# View watcher logs
docker compose logs -f watcher

# Rebuild after code changes
git pull && docker compose build && docker compose up -d
```

## Environment Variables (.env)

```env
# Required
AURA_HF_TOKEN=hf_xxxxx          # HuggingFace token (for pyannote)
AURA_LLM_API_KEY=sk-xxxxx       # OpenAI API key (for knowledge extraction)

# Optional
AURA_DB_PASSWORD=aura_secret     # PostgreSQL password
AURA_WHISPER_MODEL=large-v3      # Whisper model size
AURA_LLM_PROVIDER=openai         # openai, anthropic, local
AURA_LLM_MODEL=gpt-4o            # LLM model name
AURA_SPEAKER_MATCH_THRESHOLD=0.75
```

## Web Dashboard

Routes (Next.js, ShipKit scaffold):
- `/aura` — Overview dashboard with stats + morning briefing
- `/aura/conversations` — List of processed conversations
- `/aura/conversations/[id]` — Detail: transcript, speakers, knowledge
- `/aura/people` — All people mentioned across conversations
- `/aura/people/[id]` — Person detail: facts, mentions
- `/aura/speakers` — Voice profiles (identified speakers)
- `/aura/knowledge` — All extracted facts, commitments, events

Database connection: `AURA_DATABASE_URL=postgresql://aura:aura_secret@100.111.32.113:5432/aura`

## What's Working

- [x] Full 7-stage pipeline (VAD through knowledge extraction)
- [x] Speaker registry with voiceprint matching + re-identification
- [x] PostgreSQL persistence (recordings, conversations, people, knowledge)
- [x] Docker Compose deployment on Otto
- [x] File watcher for auto-processing
- [x] Web dashboard with live data
- [x] Ingest API for device uploads

## What's Next

- [ ] Deploy ingest service on Otto
- [ ] Test end-to-end: ESP32 upload -> ingest -> watcher -> pipeline -> dashboard
- [ ] Add WAV header parsing for raw PCM uploads
- [ ] Speaker embedding drift handling (update embeddings over time)
- [ ] Morning briefing generation (daily summary cron)
