#!/usr/bin/env python3
"""One-time setup: create the HuggingFace Dataset repo and upload the README.

Usage:
    HF_TOKEN=hf_xxx python scripts/create_hf_repo.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "carlnoah6/podcast-pipeline-transcripts"
CARD_PATH = Path("dataset_card/README.md")


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN first: export HF_TOKEN=hf_xxx")
        sys.exit(1)

    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    print(f"Repo ready: https://huggingface.co/datasets/{REPO_ID}")

    if CARD_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(CARD_PATH),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        print("Dataset card uploaded.")


if __name__ == "__main__":
    main()
