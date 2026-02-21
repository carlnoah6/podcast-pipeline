#!/usr/bin/env python3
"""Batch audio slicing via Modal.

Usage:
    python scripts/slice_batch.py --batch-size 10
    python scripts/slice_batch.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("slice_batch")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
AUDIO_REPO = "Adam429/podcast-audio"
OUTPUT_REPO = "Adam429/podcast-audio"  # slices/ subfolder in same repo


def get_done_episodes() -> set[str]:
    """Check HF for already-sliced episodes."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        files = list(api.list_repo_tree(OUTPUT_REPO, repo_type="dataset", path_in_repo="slices"))
        # Each sliced episode has a folder slices/<episode_id>/
        return {f.rfilename.split("/")[1] for f in files if "/" in f.rfilename and f.rfilename.endswith("analysis.json")}
    except Exception:
        return set()


def get_all_episodes() -> list[str]:
    """Get all episode IDs from local transcripts."""
    transcript_dir = Path("data/transcripts")
    return sorted(f.stem for f in transcript_dir.glob("*.json"))


def run(batch_size: int = 10, dry_run: bool = False) -> int:
    all_eps = get_all_episodes()
    logger.info("Total episodes: %d", len(all_eps))

    done = get_done_episodes()
    pending = [ep for ep in all_eps if ep not in done]
    logger.info("Already sliced: %d, pending: %d", len(done), len(pending))

    if not pending:
        logger.info("All episodes already sliced. Nothing to do.")
        return 0

    batch = pending[:batch_size]
    remaining = len(pending) - len(batch)
    logger.info("Processing batch of %d/%d", len(batch), len(pending))

    if dry_run:
        for ep in batch:
            logger.info("  Would slice: %s", ep)
        logger.info("Remaining: %d episodes", remaining)
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.slice.modal_slice import app, slice_episode

    args = [(ep, AUDIO_REPO, OUTPUT_REPO, HF_TOKEN) for ep in batch]

    success = 0
    total_segments = 0
    with app.run():
        for i, result in enumerate(slice_episode.starmap(args)):
            ep = batch[i]
            if "error" in result:
                logger.error("[%d/%d] %s: %s", i + 1, len(batch), ep, result["error"])
            else:
                success += 1
                total_segments += result.get("segments", 0)
                logger.info(
                    "[%d/%d] %s: %d segments, top quality %.1f",
                    i + 1, len(batch), ep,
                    result.get("segments", 0),
                    result.get("top_quality", 0),
                )

    logger.info(
        "Batch complete: %d/%d sliced, %d total segments. Remaining: %d episodes",
        success, len(batch), total_segments, remaining,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sys.exit(run(batch_size=args.batch_size, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
