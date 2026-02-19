"""Modal-based Whisper transcription using large-v3.

Runs on Modal GPUs (A10G by default). Each episode is transcribed in a
single function call; Modal handles cold-start, scaling, and teardown.

Usage (local):
    modal run src/transcribe/modal_whisper.py::transcribe_episode \
        --audio-url "https://..." --episode-id "abc123"

Usage (from Python):
    from src.transcribe.modal_whisper import transcribe_episode
    result = transcribe_episode.remote(audio_url="...", episode_id="...")
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modal app & image
# ---------------------------------------------------------------------------

app = modal.App("podcast-pipeline")

whisper_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("openai-whisper>=20240930", "torch>=2.1")
)

# ---------------------------------------------------------------------------
# Transcription function
# ---------------------------------------------------------------------------


@app.function(
    image=whisper_image,
    gpu="A10G",
    timeout=3600,
    retries=1,
)
def transcribe_episode(
    audio_url: str,
    episode_id: str,
    title: str = "",
    date: str = "",
    duration: float = 0.0,
    model_name: str = "large-v3",
    language: str = "zh",
) -> dict:
    """Download audio and transcribe with Whisper on a Modal GPU.

    Returns a dict matching the TranscriptionResult schema:
        episode_id, title, date, duration, transcription, word_count,
        segments, language, model
    """
    import whisper  # imported inside the Modal container

    # Download audio to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = _download_audio(audio_url, tmpdir)

        logger.info("Loading Whisper model: %s", model_name)
        model = whisper.load_model(model_name)

        logger.info("Transcribing: %s (%s)", episode_id, title or "untitled")
        result = model.transcribe(
            str(audio_path),
            language=language,
            verbose=False,
        )

    text: str = result.get("text", "")
    segments = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        }
        for seg in result.get("segments", [])
    ]

    word_count = len(text.replace(" ", ""))

    return {
        "episode_id": episode_id,
        "title": title,
        "date": date,
        "duration": duration,
        "transcription": text,
        "word_count": word_count,
        "segments": segments,
        "language": result.get("language", language),
        "model": model_name,
    }


def _download_audio(url: str, dest_dir: str) -> Path:
    """Download audio file via curl (available in the Modal container)."""
    # Determine extension from URL
    ext = ".mp3"
    for candidate in (".m4a", ".wav", ".ogg", ".flac", ".mp3"):
        if candidate in url.lower():
            ext = candidate
            break

    out_path = Path(dest_dir) / f"audio{ext}"
    logger.info("Downloading audio: %s → %s", url, out_path)

    subprocess.run(
        ["curl", "-fsSL", "-o", str(out_path), url],
        check=True,
        timeout=600,
    )

    size = out_path.stat().st_size
    logger.info("Downloaded %.1f MB", size / 1_000_000)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point for `modal run`
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    audio_url: str = "",
    episode_id: str = "test",
    title: str = "",
    language: str = "zh",
):
    """CLI wrapper: `modal run src/transcribe/modal_whisper.py -- --audio-url URL`"""
    if not audio_url:
        print("Usage: modal run src/transcribe/modal_whisper.py -- --audio-url URL")
        return

    result = transcribe_episode.remote(
        audio_url=audio_url,
        episode_id=episode_id,
        title=title,
        language=language,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
