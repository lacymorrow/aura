# Roadmap

## Phase 0: Foundation (Weeks 1-3) 🔴 CURRENT
**Goal**: Get the core audio pipeline working end-to-end with test files.

### Deliverables
- [ ] Project scaffolding (FastAPI app, Docker, config)
- [ ] Audio ingest endpoint (accept upload, store to disk/S3)
- [ ] VAD stage (Silero) — strip silence from audio files
- [ ] Transcription stage (faster-whisper) — generate timestamped text
- [ ] Speaker diarization (pyannote 3.1) — "who spoke when"
- [ ] Transcript assembly — merge transcription + diarization
- [ ] Basic CLI to run pipeline on a .wav file
- [ ] Test suite with sample audio files

### Success Criteria
Drop a .wav file in → get a speaker-labeled transcript out.

---

## Phase 1: Speaker Identity (Weeks 4-6)
**Goal**: Persistent speaker identification across multiple audio sessions.

### Deliverables
- [ ] Speaker embedding extraction (ECAPA-TDNN)
- [ ] Voiceprint database (PostgreSQL + pgvector)
- [ ] Owner enrollment flow
- [ ] Speaker matching (cosine similarity against known voiceprints)
- [ ] Unknown speaker clustering (cross-session)
- [ ] Speaker labeling API (assign name to voiceprint)
- [ ] Retroactive relabeling when a speaker is identified

### Success Criteria
Process 5 different recordings → system recognizes "this is the same person as recording #2" without being told.

---

## Phase 2: Knowledge Extraction (Weeks 7-10)
**Goal**: Extract structured information from labeled transcripts.

### Deliverables
- [ ] LLM extraction pipeline (facts, commitments, topics, entities)
- [ ] Knowledge graph schema (Neo4j or PostgreSQL AGE)
- [ ] Entity resolution (match extracted names to speaker IDs)
- [ ] Conversation segmentation (split day into discrete conversations)
- [ ] Daily summary generation
- [ ] Semantic search index (embed conversations → pgvector)
- [ ] Basic query API ("who is X", "what did I discuss with Y")

### Success Criteria
After processing a week of audio, query "what do I know about Sarah" → get a structured profile with facts and conversation history.

---

## Phase 3: Mobile App MVP (Weeks 11-16)
**Goal**: Functional mobile app for reviewing and searching.

### Deliverables
- [ ] React Native app scaffolding
- [ ] Auth flow (device pairing, user account)
- [ ] Morning briefing screen
- [ ] People list with person detail views
- [ ] Conversation history browser
- [ ] Natural language search
- [ ] Commitments tracker
- [ ] Unknown speaker labeling UI
- [ ] Settings (privacy, retention, consent mode)
- [ ] Push notifications for commitments/reminders

### Success Criteria
User opens app in the morning → sees yesterday's conversations, people, and action items.

---

## Phase 4: Device Integration (Weeks 17-22)
**Goal**: Connect the hardware to the processing pipeline.

### Deliverables
- [ ] Upload protocol (device → charger → WiFi → server)
- [ ] Audio format handling (Opus decode, chunking)
- [ ] Device auth & pairing
- [ ] Upload progress & status in app
- [ ] Battery/storage monitoring
- [ ] Automatic nightly processing trigger
- [ ] Error handling & retry (partial uploads, network issues)

### Success Criteria
Wear device all day → dock at night → processing runs automatically → results in app by morning.

---

## Phase 5: Real-Time Whisper Mode (Weeks 23-30)
**Goal**: Live speaker identification and contextual whispers.

### Deliverables
- [ ] BLE audio streaming (device → phone)
- [ ] On-device VAD (phone)
- [ ] On-device speaker embedding (lightweight model)
- [ ] Local voiceprint cache on phone (top 50 known speakers)
- [ ] Speaker match → notification pipeline
- [ ] Context card generation (name + last conversation + key facts)
- [ ] Spatial memory ("you came here for X")
- [ ] Whisper delivery (notification, watch, bone conduction)
- [ ] Latency optimization (target: < 3 seconds from speech to notification)

### Success Criteria
Walk up to someone you've met before → within 3 seconds, get a silent notification with their name and context.

---

## Phase 6: Essence & Legacy (Months 8-12+)
**Goal**: Long-term memory archival and personality preservation.

### Deliverables
- [ ] Speech pattern modeling (vocabulary, cadence, expressions)
- [ ] Opinion/belief extraction and tracking over time
- [ ] Story detection and archival (anecdotes the user tells repeatedly)
- [ ] Personal voice synthesis (optional, with explicit consent)
- [ ] Legacy mode UI (for family members after user's passing)
- [ ] Cognitive tracking (detect changes in speech patterns over time)
- [ ] Therapeutic integrations (dementia care, grief support)

### Success Criteria
TBD — this phase requires extensive ethical review and user research.

---

## Milestones Summary

| Milestone | Target | Key Metric |
|-----------|--------|-----------|
| Pipeline works on test audio | Week 3 | End-to-end labeled transcript |
| Speaker ID across sessions | Week 6 | Cross-session recognition accuracy > 80% |
| Knowledge graph populated | Week 10 | Structured facts from 1 week of audio |
| App MVP | Week 16 | Daily briefing + search functional |
| Hardware integrated | Week 22 | Automatic dock-to-briefing flow |
| Whisper mode | Week 30 | < 3s name recall in real-time |
| Essence features | Month 12 | Speech model + legacy mode prototype |
