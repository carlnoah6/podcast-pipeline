#!/usr/bin/env python3
"""One-off: upload existing transcripts to HuggingFace."""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "")
if not token:
    print("HF_TOKEN not set"); sys.exit(1)

api = HfApi(token=token)
api.create_repo("Adam429/podcast-transcripts", repo_type="dataset", exist_ok=True)

transcript_dir = Path("data/transcripts")
files = sorted(transcript_dir.glob("*.*"))
print(f"Uploading {len(files)} files...")

for f in files:
    if f.suffix in (".json", ".md"):
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"transcripts/{f.name}",
            repo_id="Adam429/podcast-transcripts",
            repo_type="dataset",
            commit_message=f"Add transcript: {f.name}",
        )
        print(f"  ✅ {f.name}")

print("Done!")
