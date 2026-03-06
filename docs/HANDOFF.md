# Aura — Agent Handoff
**Date:** 2026-03-05 9:16 PM EST
**From:** io (dev agent)

---

## What Is Aura?
A necklace wearable (ESP32-C6) that records audio all day, syncs via WiFi to a local processing server (Otto), and builds a knowledge graph of the user's life. Focus is **knowledge generation**, not real-time features.

**Wear → Upload → Process → Accumulate Knowledge → Build Portrait**

## Repos
| Repo | Location | Description |
|------|----------|-------------|
| `lacymorrow/aura` | `~/repo/aura` + `C:\Users\io\aura` on Otto | Python pipeline (VAD, transcription, diarization, speaker ID, knowledge extraction) |
| `lacymorrow/aura-web` | `~/repo/aura-web` | Next.js dashboard (ShipKit-based, Drizzle ORM, Tailwind, shadcn/ui) |
| `lacymorrow/shipkit` | `~/repo/shipkit` | Base framework — clean, no Aura code (was accidentally polluted, reverted) |

## Infrastructure
| Component | Details |
|-----------|---------|
| **Otto** (processing server) | Windows PC, RTX 3090 24GB, Tailscale `100.111.32.113`, LAN `192.168.1.72` |
| **SSH** | User `io`, pass `io`, port 22. sshd set to `Automatic` startup |
| **PostgreSQL** | Port 5432, user `aura`, pass `aura_secret`, db `aura` (runs in Docker on Otto) |
| **Ingest API** | FastAPI on port 8080 (Docker container `aura-ingest-1`) |
| **Watcher** | Auto-processes new uploads (Docker container `aura-watcher-1`) |

## What Works (Fully Tested)
- [x] **Full 6-stage pipeline** on Otto's 3090 — 25s for 30s audio, zero errors
  - Silero VAD → faster-whisper large-v3 → pyannote 3.1 diarization → ECAPA-TDNN embeddings → alignment → Claude extraction
- [x] **Speaker registry** with voiceprint matching (cosine similarity, re-identification confirmed)
- [x] **DB persistence** — recordings, conversations, knowledge, people, speakers all written to Postgres
- [x] **Ingest API** — single upload, chunked upload, device query, status endpoints
- [x] **Web dashboard pages** — `/aura`, `/conversations`, `/people`, `/speakers`, `/knowledge` (HTTP 200)
- [x] **Drizzle schema** matches Python SQLAlchemy models perfectly (`aura-schema.ts`)
- [x] **Server actions** — `getDashboardStats()`, `getConversations()`, `getPeople()`, `getSpeakers()`, `getKnowledge()`, `getMorningBriefing()`
- [x] **API routes** — `/api/aura/latest`, `/api/aura/upload` (proxy), `/api/aura/stats`
- [x] **Demo page** with drag-and-drop upload, uses proxy to avoid CORS
- [x] **CORS middleware** on ingest API
- [x] **Device upload protocol** documented (`docs/DEVICE-UPLOAD-PROTOCOL.md`)

## What's Broken: Docker on Otto
**This is the #1 blocker.** Docker Desktop on Otto (Windows + WSL2) is deeply unstable:
- Engine starts, works for ~30 seconds, then hangs
- `docker ps` intermittently works, then times out
- Creating new containers returns 502 Bad Gateway
- Named pipe `//./pipe/dockerDesktopLinuxEngine` disappears
- Multiple restart cycles, full reboot, kill/relaunch all tried — same result
- **The DB container (`aura-db-1`) was briefly healthy but the engine kills it**

### Options (pick one):
1. **Reinstall Docker Desktop** — Uninstall from Settings → Apps, download fresh from docker.com, reinstall. DB data in WSL volumes should survive.
2. **Go bare-metal** — Install Postgres + Python natively on Otto, skip Docker entirely. More reliable on Windows, but means managing deps on the host.
3. **Switch to WSL-native Docker** — Install `docker.io` inside WSL Ubuntu instead of Docker Desktop. More stable than Docker Desktop but needs WSL Ubuntu distro setup.

## What Needs Doing Next (Priority Order)
1. **Fix Docker or go bare-metal on Otto** — nothing works without a running DB + ingest server
2. **Production hardening** — error handling, retry logic, health checks, graceful degradation in the pipeline
3. **Test end-to-end browser flow** — demo page upload → ingest → watcher → pipeline → dashboard
4. **Wire dashboard pages** to real data (queries exist, pages need to call them)
5. **Morning briefing** — daily summary cron job
6. **ESP32 firmware** — Lacy's second agent builds upload client using `docs/DEVICE-UPLOAD-PROTOCOL.md`
7. **Speaker embedding drift** — handle voiceprint changes over time
8. **WAV header parsing** — for raw PCM uploads from ESP32

## Key Files
```
~/repo/aura/
├── src/pipeline/          # 6-stage processing pipeline
│   ├── vad.py             # Silero VAD
│   ├── transcribe.py      # faster-whisper
│   ├── diarize.py         # pyannote 4.0.4
│   ├── speaker_embed.py   # ECAPA-TDNN (speechbrain)
│   ├── align.py           # speaker-transcript alignment
│   ├── extract.py         # Claude knowledge extraction
│   └── processor.py       # orchestrator
├── src/api/ingest.py      # FastAPI ingest server
├── src/speakers/registry.py  # voiceprint matching
├── src/db/
│   ├── models.py          # SQLAlchemy models (source of truth)
│   ├── persist.py         # write pipeline results to DB
│   └── engine.py          # DB connection
├── src/cli.py             # CLI: process, serve, speaker commands
├── docker-compose.yml     # db, aura, watcher, ingest services
├── Dockerfile
└── docs/
    ├── DEVICE-UPLOAD-PROTOCOL.md
    ├── SYSTEM-STATUS.md
    └── (9 more planning docs)

~/repo/aura-web/
├── src/server/db/
│   ├── aura-schema.ts     # Drizzle schema (mirrors SQLAlchemy)
│   └── aura-db.ts         # DB connection (AURA_DATABASE_URL)
├── src/server/actions/aura/
│   └── queries.ts         # all read queries
├── src/app/api/aura/
│   ├── latest/route.ts    # latest conversation API
│   ├── upload/route.ts    # proxy to Otto ingest (no CORS)
│   └── stats/route.ts     # dashboard stats
└── src/app/(app)/(dashboard)/aura/
    ├── page.tsx            # main dashboard
    ├── demo/page.tsx       # upload demo
    ├── conversations/      # conversation list
    ├── people/             # people list
    ├── speakers/           # speaker list
    └── knowledge/          # knowledge entries
```

## Environment Variables
```bash
# Otto (.env in C:\Users\io\aura)
DATABASE_URL=postgresql://aura:aura_secret@db:5432/aura
ANTHROPIC_API_KEY=sk-ant-...  # for knowledge extraction
AURA_LLM_PROVIDER=anthropic
HF_TOKEN=hf_...  # for pyannote gated models

# aura-web (.env)
AURA_DATABASE_URL=postgresql://aura:aura_secret@100.111.32.113:5432/aura
AURA_INGEST_URL=http://100.111.32.113:8080
```

## Key Decisions Already Made
- **Local RTX 3090 over cloud** — $0/mo, 24GB VRAM fits all models
- **WiFi sync over USB** — ESP32-C6 uploads via HTTP POST
- **Chunked uploads** — 32-64KB chunks for ESP32's 320KB SRAM
- **Claude for extraction** — Anthropic API, not local LLM
- **speechbrain from git develop** — pinned to fix torchaudio 2.9+ compat
- **Docker-only on Otto** — but this decision is being tested by Docker instability
- **No emdashes (—)** in vibe.rehab content (unrelated project, but noted)

## Test Data
- `data/uploads/test_conversation.wav` — 30s two-speaker conversation (Diane from NJ, Sheila from TX)
- Successfully processed: 2 speakers identified, 6 facts extracted, summary + topics generated
