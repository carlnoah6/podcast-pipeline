"""Modal function to download episode audio and upload to HuggingFace.

Runs on Modal CPU containers. Downloads audio from RSS feed URLs and
uploads directly to a HuggingFace dataset repo, avoiding local storage.

Usage (from Python):
    from src.transcribe.modal_audio import download_and_upload_audio
    download_and_upload_audio.remote(audio_url="...", episode_id="...", hf_repo="...")
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

app = modal.App("podcast-audio-upload")

audio_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("huggingface_hub>=0.20")
)


@app.function(
    image=audio_image,
    timeout=600,
    retries=1,
)
def download_and_upload_audio(
    audio_url: str,
    episode_id: str,
    title: str = "",
    hf_repo: str = "Adam429/podcast-audio",
    hf_token: str = "",
) -> dict:
    """Download audio and upload to HuggingFace dataset repo.

    Returns dict with episode_id, filename, size_mb, hf_path.
    """
    from huggingface_hub import HfApi

    # Determine extension
    ext = ".mp3"
    for candidate in (".m4a", ".wav", ".ogg", ".flac", ".mp3"):
        if candidate in audio_url.lower():
            ext = candidate
            break

    filename = f"{episode_id}{ext}"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / filename
        logger.info("Downloading: %s", audio_url)

        subprocess.run(
            ["curl", "-fsSL", "-o", str(out_path), audio_url],
            check=True,
            timeout=300,
        )

        size_mb = out_path.stat().st_size / 1_000_000
        logger.info("Downloaded %.1f MB: %s", size_mb, filename)

        # Upload to HuggingFace
        token = hf_token or os.environ.get("HF_TOKEN", "")
        api = HfApi(token=token)

        hf_path = f"audio/{filename}"
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=hf_path,
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message=f"Add audio: {title or episode_id}",
        )
        logger.info("Uploaded to HF: %s/%s", hf_repo, hf_path)

    return {
        "episode_id": episode_id,
        "filename": filename,
        "size_mb": round(size_mb, 1),
        "hf_path": hf_path,
    }
