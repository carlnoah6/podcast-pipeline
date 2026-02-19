#!/usr/bin/env python3
"""Sync data/transcripts/ to a HuggingFace Dataset repo.

Uploads all JSON + Markdown files from data/transcripts/ to the
specified HF dataset repository. Creates the repo if it doesn't exist.

Usage:
    HF_TOKEN=hf_xxx python scripts/sync_hf.py --repo user/dataset-name
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sync_hf")

DATA_DIR = Path("data/transcripts")


def sync(repo_id: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN not set")
        sys.exit(1)

    if not repo_id:
        logger.error("No repo specified. Pass --repo user/dataset-name")
        sys.exit(1)

    api = HfApi(token=token)

    # Ensure repo exists
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    if not DATA_DIR.exists():
        logger.info("No data directory found. Nothing to sync.")
        return

    files = list(DATA_DIR.glob("*"))
    if not files:
        logger.info("No files to upload.")
        return

    logger.info("Uploading %d files to %s", len(files), repo_id)
    api.upload_folder(
        folder_path=str(DATA_DIR),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="transcripts",
        commit_message=f"sync: {len(files)} transcript files",
    )
    logger.info("Sync complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync transcripts to HuggingFace")
    parser.add_argument("--repo", required=True, help="HF dataset repo (user/name)")
    args = parser.parse_args()
    sync(args.repo)


if __name__ == "__main__":
    main()
