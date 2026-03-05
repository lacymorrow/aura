# Aura

Audio processing pipeline for wearable memory augmentation. Records conversations, identifies speakers, transcribes speech, and extracts structured knowledge into a persistent graph.

## Architecture

```
Audio File → VAD → Transcription → Diarization → Speaker ID → Alignment → Knowledge Extraction
                                                                                    ↓
                                                              PostgreSQL + pgvector (voiceprints, knowledge graph)
```

**Pipeline stages:**

| Stage | Model | What it does |
|-------|-------|-------------|
| VAD | Silero VAD | Strips silence and non-speech |
| Transcribe | Whisper large-v3 (CTranslate2) | Word-level timestamped transcription |
| Diarize | pyannote 3.1 | Who spoke when |
| Embed | ECAPA-TDNN (192-dim) | Voiceprint per speaker |
| Align | Custom | Merges transcript + diarization |
| Extract | Claude / GPT-4o | People, facts, commitments, events |

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your HuggingFace token and LLM API key

# 2. Build and run
docker compose build
docker compose run --rm aura process /app/data/uploads/your_audio.wav

# 3. Start the watcher (processes new files automatically)
docker compose up -d watcher
```

## CLI Commands

```bash
# Full pipeline
aura process <audio_file> [-o output_dir] [--no-extract] [--owner SPEAKER_00]

# Batch process all unprocessed files
aura batch [-d upload_dir] [-o output_dir]

# Watch for new files (daemon mode)
aura watch [--interval 60]

# Individual stages
aura vad <audio_file>
aura transcribe <audio_file>
aura diarize <audio_file>
aura speakers <audio_file>

# Database
aura db init
aura db status

# System info
aura status
```

## Requirements

- NVIDIA GPU with CUDA support (tested on RTX 3090, 24GB VRAM)
- Docker with nvidia-container-toolkit
- HuggingFace account with accepted model terms:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- Anthropic or OpenAI API key (for knowledge extraction)

## Performance

On RTX 3090 (30s test audio):
- VAD: ~1.4s
- Transcription: ~10s
- Diarization: ~6s
- Embeddings: ~1s
- Extraction: ~6s (API latency)
- **Total: ~25s for 30s of audio (~1.2x realtime)**

Estimated for 8hr recording day: ~2-2.5 hours processing time.

## Data Flow

```
data/
├── uploads/          # Drop audio files here
│   └── 2024-03-04_recording.wav
└── processed/        # Results appear here
    └── 2024-03-04_recording/
        ├── *_transcript.json    # Timestamped, speaker-labeled
        ├── *_transcript.txt     # Human-readable
        ├── *_speakers.json      # Speaker metadata
        ├── *_knowledge.json     # Extracted knowledge graph
        └── *_full.json          # Complete processing metadata
```

## Speaker Identification

Speakers are identified via 192-dimensional ECAPA-TDNN voiceprints. The system maintains a persistent speaker registry:

- **Cosine similarity > 0.75**: Confident match (auto-linked)
- **0.55 - 0.75**: Candidate match (flagged for review)
- **< 0.55**: New speaker (new profile created)

Embeddings improve over time via running-weighted-average updates.

## License

Private.
