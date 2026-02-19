# podcast-pipeline

End-to-end podcast transcription pipeline: ingest audio → transcribe on Modal (Whisper) → publish to HuggingFace Dataset.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Audio URLs  │────▶│  Modal (Whisper)  │────▶│  HF Dataset     │
│  (RSS/local) │     │  GPU transcribe   │     │  (push on merge)│
└─────────────┘     └──────────────────┘     └─────────────────┘
       │                     │                        ▲
       │                     ▼                        │
       │              ┌──────────────┐                │
       └─────────────▶│  GitHub repo  │───────────────┘
                      │  CI/CD (GHA)  │
                      └──────────────┘
```

### Flow

1. **Ingest** — Fetch audio from RSS feeds or local files
2. **Transcribe** — Run Whisper large-v3 on Modal GPUs (chunked for long episodes)
3. **Post-process** — Clean up, segment, generate metadata
4. **Review** — CodeRabbit auto-reviews PRs; Luna auto-merges when approved
5. **Publish** — On merge to `main`, sync transcripts to HuggingFace Dataset

## Project Structure

```
src/
├── ingest/        # Audio fetching and preprocessing
├── transcribe/    # Modal Whisper transcription
├── postprocess/   # Transcript cleanup and segmentation
└── publish/       # HuggingFace Dataset sync
```

## Setup

```bash
pip install -e ".[dev]"
modal setup  # configure Modal credentials
```

## Usage

```bash
# Transcribe a single audio file
python -m src.transcribe.run --input audio.mp3

# Ingest from RSS feed
python -m src.ingest.fetch --feed-url https://example.com/feed.xml

# Publish to HuggingFace
python -m src.publish.sync --repo carlnoah6/podcast-transcripts
```

## License

MIT
