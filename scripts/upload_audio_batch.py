#!/usr/bin/env python3
"""One-off: upload audio for episodes that already have transcripts but no audio on HF.

Usage:
    python scripts/upload_audio_batch.py --batch-size 10
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.filter import filter_episodes
from src.ingest.rss import parse_feed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("upload_audio")

DEFAULT_FEED_URL = "https://feed.xyzfm.space/y9qnpfdrctnx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed-url", default=DEFAULT_FEED_URL)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--transcript-dir", type=Path, default=Path("data/transcripts"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token and not args.dry_run:
        logger.error("HF_TOKEN not set")
        sys.exit(1)

    # Find episodes that have transcripts
    episodes = parse_feed(args.feed_url, max_episodes=0)
    episodes = filter_episodes(episodes)

    transcript_ids = set()
    if args.transcript_dir.exists():
        transcript_ids = {f.stem for f in args.transcript_dir.glob("*.json")}

    # Only episodes WITH transcripts (we want to backfill their audio)
    batch = [ep for ep in episodes if ep.episode_id in transcript_ids]
    batch.sort(key=lambda e: e.date)
    batch = batch[: args.batch_size]

    logger.info("Found %d episodes with transcripts, uploading %d", len(transcript_ids), len(batch))

    if args.dry_run:
        for ep in batch:
            logger.info("  Would upload: %s — %s", ep.episode_id, ep.title)
        return

    from src.transcribe.modal_audio import app as audio_app, download_and_upload_audio

    # Ensure HF repo exists
    try:
        from huggingface_hub import HfApi
        HfApi(token=hf_token).create_repo(
            "carlnoah6/podcast-audio", repo_type="dataset", exist_ok=True
        )
        logger.info("HF repo ready")
    except Exception:
        logger.warning("Could not verify HF repo")

    audio_args = [
        (ep.audio_url, ep.episode_id, ep.title, "carlnoah6/podcast-audio", hf_token)
        for ep in batch
    ]

    logger.info("Launching %d audio uploads in parallel...", len(batch))
    success = 0
    with audio_app.run():
        for i, (result, ep) in enumerate(
            zip(download_and_upload_audio.starmap(audio_args), batch)
        ):
            try:
                success += 1
                logger.info(
                    "[%d/%d] Uploaded: %s — %s (%.1f MB)",
                    i + 1, len(batch), result["filename"], ep.title, result["size_mb"],
                )
            except Exception:
                logger.exception("[%d/%d] Failed: %s", i + 1, len(batch), ep.episode_id)

    logger.info("Done: %d/%d uploaded", success, len(batch))


if __name__ == "__main__":
    main()
