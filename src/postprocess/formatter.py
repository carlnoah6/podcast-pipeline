"""Format transcription results into JSON and Markdown files.

Reads TranscriptionResult dicts (from Modal) and writes structured output
to the data/transcripts/ directory, ready for HuggingFace Dataset sync.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.models import TranscriptionResult

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/transcripts")


def dict_to_result(raw: dict) -> TranscriptionResult:
    """Convert a raw dict (from Modal JSON) into a TranscriptionResult."""
    return TranscriptionResult(
        episode_id=raw["episode_id"],
        title=raw.get("title", ""),
        date=raw.get("date", ""),
        duration=raw.get("duration", 0.0),
        transcription=raw.get("transcription", ""),
        word_count=raw.get("word_count", 0),
        segments=raw.get("segments", []),
        language=raw.get("language", "zh"),
        model=raw.get("model", "large-v3"),
    )


def save_transcript(
    result: TranscriptionResult | dict,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Write both JSON and Markdown files for a transcription result.

    File naming: {episode_id}.json / {episode_id}.md

    Returns:
        Tuple of (json_path, md_path).
    """
    if isinstance(result, dict):
        result = dict_to_result(result)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{result.episode_id}.json"
    md_path = output_dir / f"{result.episode_id}.md"

    result.to_json(json_path)
    result.to_markdown(md_path)

    logger.info("Saved transcript: %s (.json + .md)", result.episode_id)
    return json_path, md_path


def save_batch(
    results: list[dict],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[tuple[Path, Path]]:
    """Save multiple transcription results at once."""
    paths = []
    for raw in results:
        paths.append(save_transcript(raw, output_dir))
    logger.info("Saved %d transcripts to %s", len(paths), output_dir)
    return paths
