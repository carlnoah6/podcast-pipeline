#!/usr/bin/env python3
"""Batch proofread transcripts with LLM, chain-style.

Usage:
    python scripts/proofread_batch.py --batch-size 10 --output-dir data/proofread
    python scripts/proofread_batch.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("proofread_batch")


def existing_ids(output_dir: Path) -> set[str]:
    """Return episode IDs already proofread."""
    ids: set[str] = set()
    if not output_dir.exists():
        return ids
    for f in output_dir.glob("*.json"):
        ids.add(f.stem)
    return ids


def run(
    transcript_dir: Path,
    output_dir: Path,
    batch_size: int = 10,
    dry_run: bool = False,
    api_base: str = "",
    api_key: str = "",
    model: str = "",
) -> int:
    # 1. Find all transcripts
    all_transcripts = sorted(transcript_dir.glob("*.json"))
    logger.info("Total transcripts: %d", len(all_transcripts))

    # 2. Skip already proofread
    done = existing_ids(output_dir)
    pending = [t for t in all_transcripts if t.stem not in done]
    logger.info("Already proofread: %d, pending: %d", len(done), len(pending))

    if not pending:
        logger.info("All transcripts already proofread. Nothing to do.")
        return 0

    # 3. Batch
    batch = pending[:batch_size]
    logger.info("Processing batch of %d/%d", len(batch), len(pending))

    if dry_run:
        logger.info("DRY RUN — would proofread:")
        for t in batch:
            logger.info("  %s", t.name)
        logger.info("Remaining: %d", len(pending) - len(batch))
        return 0

    # 4. Proofread
    from src.proofread import proofread_transcript

    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if model:
        kwargs["model"] = model

    success = 0
    total_changes = 0
    for i, t in enumerate(batch):
        try:
            result = proofread_transcript(t, output_dir=output_dir, **kwargs)
            success += 1
            total_changes += len(result.get("changes", []))
            logger.info(
                "[%d/%d] %s: %d changes",
                i + 1, len(batch), t.stem, len(result.get("changes", [])),
            )
        except Exception:
            logger.exception("[%d/%d] Failed: %s", i + 1, len(batch), t.stem)

    remaining = len(pending) - len(batch)
    logger.info(
        "Batch complete: %d/%d proofread, %d total changes. Remaining: %d",
        success, len(batch), total_changes, remaining,
    )
    # Output remaining for chain trigger
    print(f"::set-output name=remaining::{remaining}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript-dir", type=Path, default=Path("data/transcripts"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/proofread"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-base", default=os.environ.get("API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("PROOFREAD_MODEL", ""))
    args = parser.parse_args()

    sys.exit(run(
        args.transcript_dir,
        args.output_dir,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
    ))


if __name__ == "__main__":
    main()
