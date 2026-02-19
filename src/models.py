"""Data models for podcast episodes and transcription results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Episode:
    """Represents a single podcast episode from an RSS feed."""

    episode_id: str
    title: str
    date: str  # ISO 8601 date string
    duration: float  # seconds
    audio_url: str
    audio_size: int = 0  # bytes, 0 if unknown
    description: str = ""

    @property
    def duration_minutes(self) -> float:
        return self.duration / 60


@dataclass
class TranscriptionResult:
    """Structured output from Whisper transcription."""

    episode_id: str
    title: str
    date: str
    duration: float
    transcription: str
    word_count: int = 0
    segments: list[dict] = field(default_factory=list)
    language: str = "zh"
    model: str = "large-v3"

    def __post_init__(self) -> None:
        if self.word_count == 0 and self.transcription:
            self.word_count = len(self.transcription.replace(" ", ""))

    def to_json(self, path: Path | None = None) -> str:
        """Serialize to JSON string. Optionally write to file."""
        data = asdict(self)
        text = json.dumps(data, ensure_ascii=False, indent=2)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    def to_markdown(self, path: Path | None = None) -> str:
        """Render a human-readable Markdown transcript."""
        lines = [
            f"# {self.title}",
            "",
            f"- **Episode ID**: {self.episode_id}",
            f"- **Date**: {self.date}",
            f"- **Duration**: {self.duration:.0f}s ({self.duration / 60:.1f} min)",
            f"- **Word count**: {self.word_count}",
            f"- **Language**: {self.language}",
            f"- **Model**: {self.model}",
            "",
            "---",
            "",
            self.transcription,
            "",
        ]
        text = "\n".join(lines)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text
