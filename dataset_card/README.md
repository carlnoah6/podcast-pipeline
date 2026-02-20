---
license: cc-by-nc-4.0
language:
  - zh
tags:
  - podcast
  - transcription
  - whisper
  - chinese
pretty_name: Podcast Pipeline Transcripts
size_categories:
  - n<1K
---

# Podcast Pipeline Transcripts

Whisper large-v3 transcriptions of Chinese podcast episodes, produced by
[podcast-pipeline](https://github.com/carlnoah6/podcast-pipeline).

## Dataset Structure

Each row represents one episode:

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | string | Unique episode identifier |
| `title` | string | Episode title |
| `date` | string | Publication date (ISO 8601) |
| `duration` | float | Duration in seconds |
| `transcription` | string | Full transcript text |
| `word_count` | int | Character/word count |
| `segments` | list | Timestamped segments (`start`, `end`, `text`) |
| `language` | string | Detected language code |
| `model` | string | Whisper model used |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("carlnoah6/podcast-pipeline-transcripts")
print(ds["train"][0]["title"])
```

## Pipeline

Audio → Modal Whisper (large-v3, A10G GPU) → JSON → Parquet → HuggingFace

Sync is automated via GitHub Actions on every push to `data/` in the main branch.
