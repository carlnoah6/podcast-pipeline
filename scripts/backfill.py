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

    # 6. Transcribe
    try:
        from src.transcribe.modal_whisper import transcribe_episode
    except ImportError:
        logger.error("Modal not available. Install with: pip install modal")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    for i, ep in enumerate(batch):
        logger.info("[%d/%d] Transcribing: %s — %s", i + 1, len(batch), ep.episode_id, ep.title)
        try:
            result = transcribe_episode.remote(
                audio_url=ep.audio_url,
                episode_id=ep.episode_id,
                title=ep.title,
                date=ep.date,
                duration=ep.duration,
            )
            save_transcript(result, output_dir=output_dir)
            success += 1
            logger.info("  Done: %s (%d words)", ep.episode_id, result.get("word_count", 0))
        except Exception:
            logger.exception("  Failed: %s", ep.episode_id)

    remaining = len(episodes) - len(batch)
    logger.info(
        "Batch complete: %d/%d transcribed. Remaining: %d episodes.",
        success,
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
