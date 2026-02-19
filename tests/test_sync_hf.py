"""Tests for scripts/sync_hf.py dataset building logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add project root to path so we can import the script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.sync_hf import FEATURES, build_dataset, load_transcripts


@pytest.fixture()
def sample_transcript(tmp_path: Path) -> Path:
    """Create a sample transcript JSON file."""
    data = {
        "episode_id": "ep001",
        "title": "Test Episode",
        "date": "2026-01-15",
        "duration": 1800.0,
        "transcription": "This is a test transcript with some content.",
        "word_count": 42,
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "This is a test"},
            {"start": 5.0, "end": 10.0, "text": "transcript with some content."},
        ],
        "language": "en",
        "model": "large-v3",
    }
    json_path = tmp_path / "ep001.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def test_load_transcripts(sample_transcript: Path) -> None:
    records = load_transcripts(sample_transcript)
    assert len(records) == 1
    r = records[0]
    assert r["episode_id"] == "ep001"
    assert r["title"] == "Test Episode"
    assert r["duration"] == 1800.0
    assert len(r["segments"]["start"]) == 2
    assert r["segments"]["text"][0] == "This is a test"


def test_load_transcripts_skips_invalid(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    records = load_transcripts(tmp_path)
    assert len(records) == 0


def test_build_dataset(sample_transcript: Path) -> None:
    records = load_transcripts(sample_transcript)
    ds = build_dataset(records)
    assert len(ds) == 1
    assert ds.column_names == list(FEATURES.keys())
    assert ds[0]["episode_id"] == "ep001"
    assert ds[0]["word_count"] == 42


def test_build_dataset_empty() -> None:
    ds = build_dataset([])
    assert len(ds) == 0
    assert ds.column_names == list(FEATURES.keys())


def test_dataset_to_parquet(sample_transcript: Path, tmp_path: Path) -> None:
    records = load_transcripts(sample_transcript)
    ds = build_dataset(records)
    parquet_path = tmp_path / "test.parquet"
    ds.to_parquet(str(parquet_path))
    assert parquet_path.exists()
    assert parquet_path.stat().st_size > 0
