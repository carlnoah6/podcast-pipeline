#!/usr/bin/env python3
"""Batch analyze episodes: style + entity extraction (parallel).

Usage:
    python scripts/analyze_batch.py --batch-size 30 --workers 10 --output-dir data/analysis
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("analyze_batch")


def process_one(transcript_path: Path, output_dir: Path, proofread_dir: Path) -> tuple[str, int | None, str | None]:
    """Process a single episode. Returns (episode_id, entity_count, error)."""
    from src.analyze import analyze_transcript

    episode_id = transcript_path.stem
    try:
        result = analyze_transcript(
            transcript_path,
            output_dir=output_dir,
            use_proofread=True,
            proofread_dir=proofread_dir,
        )
        if result.get("skipped"):
            return episode_id, None, "skipped"
        return episode_id, len(result.get("entities", [])), None
    except Exception as e:
        logger.exception("Failed: %s", episode_id)
        return episode_id, None, str(e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument("--transcript-dir", type=Path, default=Path("data/transcripts"))
    parser.add_argument("--proofread-dir", type=Path, default=Path("data/proofread"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find episodes not yet analyzed
    all_transcripts = sorted(args.transcript_dir.glob("*.json"))
    done = {f.stem for f in args.output_dir.glob("*.json")}
    pending = [f for f in all_transcripts if f.stem not in done]

    logger.info("Already analyzed: %d, pending: %d", len(done), len(pending))

    if not pending:
        logger.info("All transcripts already analyzed!")
        return

    batch = pending[: args.batch_size]
    logger.info("Processing batch of %d episodes with %d workers...", len(batch), args.workers)

    success = 0
    total_entities = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, t, args.output_dir, args.proofread_dir): t
            for t in batch
        }
        for i, future in enumerate(as_completed(futures), 1):
            episode_id, entity_count, error = future.result()
            if error == "skipped":
                # Write marker so we don't retry forever
                marker = args.output_dir / f"{episode_id}.json"
                if not marker.exists():
                    with open(marker, "w") as f:
                        json.dump({"episode_id": episode_id, "skipped": True, "reason": "text too short"}, f)
                success += 1
                logger.warning("[%d/%d] %s: skipped (text too short)", i, len(batch), episode_id)
            elif error:
                logger.warning("[%d/%d] %s: %s", i, len(batch), episode_id, error)
            else:
                success += 1
                total_entities += entity_count or 0
                logger.info("[%d/%d] %s: %d entities", i, len(batch), episode_id, entity_count)

    remaining = len(pending) - success
    logger.info(
        "Batch complete: %d/%d analyzed, %d total entities. Remaining: %d",
        success, len(batch), total_entities, remaining,
    )


if __name__ == "__main__":
    main()
