#!/usr/bin/env python3
"""One-off: sync missing audio and transcripts to HuggingFace.

Compares local data/transcripts with HF repos and uploads anything missing.
Also downloads and uploads audio for episodes that have transcripts but no audio on HF.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sync_hf")


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        logger.error("HF_TOKEN not set")
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)

    # --- Sync proofread ---
    sync_proofread(api, hf_token)

    # --- Sync transcripts ---
    transcript_dir = Path("data/transcripts")
    local_files = sorted(transcript_dir.glob("*.*")) if transcript_dir.exists() else []

    # Get existing HF transcript files
    try:
        hf_tr = {f.rfilename for f in api.list_repo_tree(
            "Adam429/podcast-transcripts", repo_type="dataset", path_in_repo="transcripts"
        ) if hasattr(f, "rfilename")}
    except Exception:
        hf_tr = set()

    missing_tr = [f for f in local_files if f"transcripts/{f.name}" not in hf_tr]
    logger.info("Transcripts: %d local, %d on HF, %d missing", len(local_files), len(hf_tr), len(missing_tr))

    for f in missing_tr:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"transcripts/{f.name}",
            repo_id="Adam429/podcast-transcripts",
            repo_type="dataset",
            commit_message=f"Sync: {f.name}",
        )
        logger.info("  Uploaded transcript: %s", f.name)

    # --- Sync audio (ep1-10 missing) ---
    # Get existing HF audio files
    try:
        hf_audio = {f.rfilename.split("/")[-1].split(".")[0] for f in api.list_repo_tree(
            "Adam429/podcast-audio", repo_type="dataset", path_in_repo="audio"
        ) if hasattr(f, "rfilename")}
    except Exception:
        hf_audio = set()

    # Get episode IDs that have transcripts
    local_ids = {f.stem for f in transcript_dir.glob("*.json")} if transcript_dir.exists() else set()
    missing_audio_ids = local_ids - hf_audio
    logger.info("Audio: %d on HF, %d local episodes, %d missing audio", len(hf_audio), len(local_ids), len(missing_audio_ids))

    if missing_audio_ids:
        # Need RSS feed to get audio URLs
        from src.ingest.rss import parse_feed
        episodes = parse_feed("https://feed.xyzfm.space/y9qnpfdrctnx", max_episodes=0)
        ep_map = {ep.episode_id: ep for ep in episodes}

        # Upload via Modal
        from src.transcribe.modal_audio import app as audio_app, download_and_upload_audio

        missing_eps = [ep_map[eid] for eid in sorted(missing_audio_ids) if eid in ep_map]
        logger.info("Uploading %d missing audio files via Modal...", len(missing_eps))

        audio_args = [
            (ep.audio_url, ep.episode_id, ep.title, "Adam429/podcast-audio", hf_token)
            for ep in missing_eps
        ]

        with audio_app.run():
            for i, (result, ep) in enumerate(
                zip(download_and_upload_audio.starmap(audio_args), missing_eps)
            ):
                try:
                    logger.info(
                        "[%d/%d] Audio uploaded: %s (%.1f MB)",
                        i + 1, len(missing_eps), result["filename"], result["size_mb"],
                    )
                except Exception:
                    logger.exception("[%d/%d] Audio failed: %s", i + 1, len(missing_eps), ep.episode_id)

    logger.info("Sync complete!")


def sync_proofread(api, hf_token: str) -> None:
    """Sync proofread files to HF."""
    proofread_dir = Path("data/proofread")
    if not proofread_dir.exists():
        return

    local_files = sorted(proofread_dir.glob("*.*"))
    try:
        hf_files = {f for f in api.list_repo_files(
            "Adam429/podcast-transcripts", repo_type="dataset"
        ) if f.startswith("proofread/")}
    except Exception:
        hf_files = set()

    missing = [f for f in local_files if f"proofread/{f.name}" not in hf_files]
    logger.info("Proofread: %d local, %d on HF, %d missing", len(local_files), len(hf_files), len(missing))

    for f in missing:
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"proofread/{f.name}",
                repo_id="Adam429/podcast-transcripts",
                repo_type="dataset",
                commit_message=f"Sync proofread: {f.name}",
            )
            logger.info("  Uploaded: %s", f.name)
        except Exception:
            logger.exception("  Failed: %s", f.name)


if __name__ == "__main__":
    main()
