#!/usr/bin/env python3
"""Pipeline runner: RSS → filter → Modal transcribe → save.

Called by the check-new-episodes GitHub Action.
Skips episodes that already have transcripts in the output directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.filter import filter_episodes
from src.ingest.rss import parse_feed
from src.postprocess.formatter import save_transcript

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")

DEFAULT_PODCAST_ID = os.environ.get("PODCAST_ID", "")


def existing_ids(output_dir: Path) -> set[str]:
    """Return episode IDs that already have transcripts."""
    ids: set[str] = set()
    if not output_dir.exists():
        return ids
    for f in output_dir.glob("*.json"):
        ids.add(f.stem)
    return ids


def run(
    podcast_id: str,
    max_episodes: int,
    output_dir: Path,
) -> int:
    if not podcast_id:
        logger.error("No podcast ID provided. Set PODCAST_ID env var or pass --podcast-id.")
        return 1

    # 1. Fetch RSS
    logger.info("Fetching RSS for: %s", podcast_id)
    episodes = parse_feed(podcast_id, max_episodes=0)
    logger.info("Found %d episodes in feed", len(episodes))

    # 2. Filter
    episodes = filter_episodes(episodes)
    logger.info("After filter: %d episodes", len(episodes))

    # 3. Skip already-transcribed
    done = existing_ids(output_dir)
    episodes = [ep for ep in episodes if ep.episode_id not in done]
    logger.info("After dedup: %d new episodes (skipped %d existing)", len(episodes), len(done))

    if not episodes:
        logger.info("Nothing to transcribe. Done.")
        return 0

    # 4. Limit
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
        logger.info("Processing %d episodes (max_episodes=%d)", len(episodes), max_episodes)

    # 5. Transcribe via Modal
    try:
        from src.transcribe.modal_whisper import app as modal_app, transcribe_episode
    except ImportError:
        logger.error("Modal not available. Install with: pip install modal")
        return 1

    success = 0
    with modal_app.run():
        for ep in episodes:
            logger.info("Transcribing: %s — %s", ep.episode_id, ep.title)
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
                logger.info("Done: %s (%d words)", ep.episode_id, result.get("word_count", 0))
            except Exception:
                logger.exception("Failed to transcribe: %s", ep.episode_id)

    logger.info("Pipeline complete: %d/%d transcribed", success, len(episodes))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Podcast transcription pipeline")
    parser.add_argument("--podcast-id", default=DEFAULT_PODCAST_ID)
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("data/transcripts"))
    args = parser.parse_args()

    sys.exit(run(args.podcast_id, args.max_episodes, args.output_dir))


if __name__ == "__main__":
    main()
