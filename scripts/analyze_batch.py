#!/usr/bin/env python3
"""Batch analyze episodes: style + entity extraction.

Usage:
    python scripts/analyze_batch.py --batch-size 10 --output-dir data/analysis
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("analyze_batch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10)
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
    logger.info("Processing batch of %d episodes...", len(batch))

    from src.analyze import analyze_transcript

    success = 0
    for i, transcript_path in enumerate(batch, 1):
        try:
            result = analyze_transcript(
                transcript_path,
                output_dir=args.output_dir,
                use_proofread=True,
                proofread_dir=args.proofread_dir,
            )
            if not result.get("skipped"):
                entity_count = len(result.get("entities", []))
                logger.info("[%d/%d] %s: %d entities", i, len(batch), transcript_path.stem, entity_count)
                success += 1
            else:
                logger.warning("[%d/%d] %s: skipped", i, len(batch), transcript_path.stem)
        except Exception:
            logger.exception("[%d/%d] Failed: %s", i, len(batch), transcript_path.stem)

    remaining = len(pending) - success
    logger.info("Batch complete: %d/%d analyzed. Remaining: %d", success, len(batch), remaining)


if __name__ == "__main__":
    main()
