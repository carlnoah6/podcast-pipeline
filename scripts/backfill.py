#!/usr/bin/env python3
"""Backfill: download and transcribe historical episodes in batches.

Processes episodes in batches to avoid timeouts. Skips episodes that
already have transcripts. Designed to be run repeatedly until all
episodes are processed.

Usage:
    python scripts/backfill.py --batch-size 10 --output-dir data/transcripts
    python scripts/backfill.py --dry-run  # preview what would be processed
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
from src.postprocess.formatter import save_transcript

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill")

# Apple Podcast RSS feed for 独树不成林 (289 episodes, full history)
DEFAULT_FEED_URL = "https://feed.xyzfm.space/y9qnpfdrctnx"


def existing_ids(output_dir: Path) -> set[str]:
    """Return episode IDs that already have transcripts."""
    ids: set[str] = set()
    if not output_dir.exists():
        return ids
    for f in output_dir.glob("*.json"):
        ids.add(f.stem)
    return ids


def run(
    feed_url: str,
    batch_size: int,
    output_dir: Path,
    dry_run: bool = False,
    oldest_first: bool = True,
) -> int:
    # 1. Fetch all episodes from RSS
    logger.info("Fetching RSS feed: %s", feed_url)
    episodes = parse_feed(feed_url, max_episodes=0)
    logger.info("Total episodes in feed: %d", len(episodes))

    # 2. Filter
    episodes = filter_episodes(episodes)
    logger.info("After filter: %d episodes", len(episodes))

    # 3. Skip already-transcribed
    done = existing_ids(output_dir)
    episodes = [ep for ep in episodes if ep.episode_id not in done]
    logger.info("After dedup: %d new episodes (skipped %d existing)", len(episodes), len(done))

    if not episodes:
        logger.info("All episodes already transcribed. Nothing to do.")
        return 0

    # 4. Sort: oldest first for backfill (chronological order)
    if oldest_first:
        episodes.sort(key=lambda e: e.date)

    # 5. Batch
    batch = episodes[:batch_size]
    logger.info(
        "Processing batch of %d/%d episodes (oldest_first=%s)",
        len(batch),
        len(episodes),
        oldest_first,
    )

    if dry_run:
        logger.info("DRY RUN — would process:")
        for ep in batch:
            logger.info("  %s — %s (%.0fs)", ep.episode_id, ep.title, ep.duration)
        logger.info("Remaining after this batch: %d", len(episodes) - len(batch))
        return 0

    # 6. Transcribe (parallel via Modal) + save audio to HuggingFace
    try:
        from src.transcribe.modal_whisper import app as modal_app, transcribe_episode
        from src.transcribe.modal_audio import app as audio_app, download_and_upload_audio
    except ImportError:
        logger.error("Modal not available. Install with: pip install modal")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    success = 0

    logger.info("Launching %d transcriptions + audio uploads in parallel...", len(batch))

    # Transcription args
    transcribe_args = [
        (ep.audio_url, ep.episode_id, ep.title, ep.date, ep.duration)
        for ep in batch
    ]

    # Audio upload args (include HF token from env)
    hf_token = os.environ.get("HF_TOKEN", "")
    audio_args = [
        (ep.audio_url, ep.episode_id, ep.title, "Adam429/podcast-audio", hf_token)
        for ep in batch
    ]

    with modal_app.run():
        # Launch transcriptions
        for i, (result, ep) in enumerate(
            zip(
                transcribe_episode.starmap(transcribe_args),
                batch,
            )
        ):
            try:
                save_transcript(result, output_dir=output_dir)
                success += 1
                logger.info(
                    "[%d/%d] Done: %s — %s (%d words)",
                    i + 1,
                    len(batch),
                    ep.episode_id,
                    ep.title,
                    result.get("word_count", 0),
                )
            except Exception:
                logger.exception("[%d/%d] Failed to save: %s", i + 1, len(batch), ep.episode_id)

    # Upload audio to HuggingFace (separate Modal app, CPU only)
    # Wrapped in try/except so audio failures don't block transcript saving
    audio_success = 0
    if hf_token:
        logger.info("Uploading %d audio files to HuggingFace...", len(batch))
        # Ensure HF dataset repo exists
        try:
            from huggingface_hub import HfApi
            HfApi(token=hf_token).create_repo(
                "Adam429/podcast-audio", repo_type="dataset", exist_ok=True
            )
        except Exception:
            logger.warning("Could not create/verify HF repo (will try upload anyway)")

        try:
            with audio_app.run():
                for i, (result, ep) in enumerate(
                    zip(
                        download_and_upload_audio.starmap(audio_args),
                        batch,
                    )
                ):
                    try:
                        audio_success += 1
                        logger.info(
                            "[%d/%d] Audio uploaded: %s (%.1f MB)",
                            i + 1,
                            len(batch),
                            result["filename"],
                            result["size_mb"],
                        )
                    except Exception:
                        logger.exception("[%d/%d] Audio upload failed: %s", i + 1, len(batch), ep.episode_id)
        except Exception:
            logger.exception("Audio upload batch failed (transcripts are still saved)")
    else:
        logger.warning("HF_TOKEN not set — skipping audio upload to HuggingFace")

    # Upload transcripts to HuggingFace (direct, no Modal needed)
    if hf_token and success > 0:
        logger.info("Uploading %d transcripts to HuggingFace...", success)
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            api.create_repo("Adam429/podcast-transcripts", repo_type="dataset", exist_ok=True)

            transcript_uploaded = 0
            for f in output_dir.iterdir():
                if f.suffix in (".json", ".md"):
                    ep_id = f.stem
                    # Only upload files from this batch
                    if any(ep.episode_id == ep_id for ep in batch):
                        api.upload_file(
                            path_or_fileobj=str(f),
                            path_in_repo=f"transcripts/{f.name}",
                            repo_id="Adam429/podcast-transcripts",
                            repo_type="dataset",
                            commit_message=f"Add transcript: {f.name}",
                        )
                        transcript_uploaded += 1
            logger.info("Uploaded %d transcript files to HuggingFace", transcript_uploaded)
        except Exception:
            logger.exception("Transcript upload to HF failed (files are still in git)")

    remaining = len(episodes) - len(batch)
    logger.info(
        "Batch complete: %d/%d transcribed, %d/%d audio uploaded. Remaining: %d episodes.",
        success,
        len(batch),
        audio_success,
        len(batch),
        remaining,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical episode transcriptions")
    parser.add_argument("--feed-url", default=DEFAULT_FEED_URL, help="RSS feed URL")
    parser.add_argument("--batch-size", type=int, default=10, help="Episodes per batch")
    parser.add_argument("--output-dir", type=Path, default=Path("data/transcripts"))
    parser.add_argument("--dry-run", action="store_true", help="Preview without transcribing")
    parser.add_argument("--newest-first", action="store_true", help="Process newest episodes first")
    args = parser.parse_args()

    sys.exit(
        run(
            args.feed_url,
            args.batch_size,
            args.output_dir,
            dry_run=args.dry_run,
            oldest_first=not args.newest_first,
        )
    )


if __name__ == "__main__":
    main()
