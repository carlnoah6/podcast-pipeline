#!/usr/bin/env python3
"""Build a HuggingFace Dataset from data/transcripts/ and push it.

Reads all JSON transcript files, builds a datasets.Dataset with typed
columns, exports to Parquet, and pushes to the specified HF repo.

Usage:
    HF_TOKEN=hf_xxx python scripts/sync_hf.py --repo user/dataset-name
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sync_hf")

DATA_DIR = Path("data/transcripts")

FEATURES = Features(
    {
        "episode_id": Value("string"),
        "title": Value("string"),
        "date": Value("string"),
        "duration": Value("float64"),
        "transcription": Value("string"),
        "word_count": Value("int64"),
        "segments": Sequence(
            {
                "start": Value("float64"),
                "end": Value("float64"),
                "text": Value("string"),
            }
        ),
        "language": Value("string"),
        "model": Value("string"),
    }
)


def load_transcripts(data_dir: Path) -> list[dict]:
    """Load all JSON transcript files from the data directory."""
    records = []
    for json_path in sorted(data_dir.glob("*.json")):
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            # Convert segments to columnar format for datasets Sequence
            raw_segments = raw.get("segments", [])
            segments = {
                "start": [float(s.get("start", 0.0)) for s in raw_segments],
                "end": [float(s.get("end", 0.0)) for s in raw_segments],
                "text": [str(s.get("text", "")) for s in raw_segments],
            }
            records.append(
                {
                    "episode_id": str(raw.get("episode_id", json_path.stem)),
                    "title": str(raw.get("title", "")),
                    "date": str(raw.get("date", "")),
                    "duration": float(raw.get("duration", 0.0)),
                    "transcription": str(raw.get("transcription", "")),
                    "word_count": int(raw.get("word_count", 0)),
                    "segments": segments,
                    "language": str(raw.get("language", "zh")),
                    "model": str(raw.get("model", "large-v3")),
                }
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Skipping %s: %s", json_path.name, exc)
    return records


def build_dataset(records: list[dict]) -> Dataset:
    """Build a typed HuggingFace Dataset from transcript records."""
    if not records:
        # from_list([]) with complex features can fail; use from_dict instead
        empty = {col: [] for col in FEATURES}
        return Dataset.from_dict(empty, features=FEATURES)
    return Dataset.from_list(records, features=FEATURES)


def sync(repo_id: str, *, export_jsonl: bool = True) -> None:
    """Build dataset and push to HuggingFace."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN not set")
        sys.exit(1)

    if not repo_id:
        logger.error("No repo specified. Pass --repo user/dataset-name")
        sys.exit(1)

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    if not DATA_DIR.exists():
        logger.info("No data directory found at %s. Nothing to sync.", DATA_DIR)
        return

    records = load_transcripts(DATA_DIR)
    if not records:
        logger.info("No transcript JSON files found. Nothing to sync.")
        return

    logger.info("Building dataset from %d transcripts", len(records))
    ds = build_dataset(records)

    # Push as Parquet (default HF format)
    logger.info("Pushing dataset to %s", repo_id)
    ds.push_to_hub(repo_id, token=token, commit_message=f"sync: {len(records)} episodes")

    # Also export JSONL alongside Parquet for easy consumption
    if export_jsonl:
        jsonl_path = Path("data/dataset.jsonl")
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_json(str(jsonl_path))
        api.upload_file(
            path_or_fileobj=str(jsonl_path),
            path_in_repo="data/dataset.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="sync: add JSONL export",
        )
        logger.info("Uploaded JSONL to %s/data/dataset.jsonl", repo_id)

    logger.info("Sync complete: %d episodes pushed to %s", len(records), repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync transcripts to HuggingFace Dataset")
    parser.add_argument("--repo", required=True, help="HF dataset repo (user/name)")
    parser.add_argument("--no-jsonl", action="store_true", help="Skip JSONL export")
    args = parser.parse_args()
    sync(args.repo, export_jsonl=not args.no_jsonl)


if __name__ == "__main__":
    main()
