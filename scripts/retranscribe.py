#!/usr/bin/env python3
"""Re-transcribe specific episodes that had bad Whisper output.

Downloads audio from HF, re-runs Whisper on Modal, saves results.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("retranscribe")

# Episodes with garbage transcriptions (repeated ad text)
BAD_EPISODES = [
    "673176a6f373fe5d4d44f6a7",
    "67dc65c878103db3bddad15b",
    "6918d089cbba038b42b59a3e",
    "65da72a035dd8780ed2d4269",
    "67d48f33e924d4525ab33382",
    "69684e07109824f9e188fdcd",
    "6828fee2457b22ce0d33748a",
    "6983a489c78b823892b31407",
]


def main() -> None:
    from huggingface_hub import HfApi
    import modal

    api = HfApi()
    transcript_dir = Path("data/transcripts")

    # Get audio URLs from HF
    episodes = []
    for eid in BAD_EPISODES:
        # Load original metadata
        with open(transcript_dir / f"{eid}.json") as f:
            meta = json.load(f)

        # Find audio file on HF
        for ext in ("m4a", "mp3"):
            try:
                url = f"https://huggingface.co/datasets/Adam429/podcast-audio/resolve/main/audio/{eid}.{ext}"
                episodes.append({
                    "episode_id": eid,
                    "audio_url": url,
                    "title": meta.get("title", ""),
                    "date": meta.get("date", ""),
                    "duration": meta.get("duration", 0),
                })
                break
            except Exception:
                continue

    logger.info("Re-transcribing %d episodes on Modal...", len(episodes))

    from src.transcribe.modal_whisper import app as whisper_app, transcribe_episode

    with whisper_app.run():
        for i, ep in enumerate(episodes, 1):
            try:
                result = transcribe_episode.remote(
                    audio_url=ep["audio_url"],
                    episode_id=ep["episode_id"],
                    title=ep["title"],
                    date=ep["date"],
                    duration=ep["duration"],
                )
                # Save
                out_path = transcript_dir / f"{ep['episode_id']}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                logger.info(
                    "[%d/%d] %s: %d chars (%s)",
                    i, len(episodes), ep["episode_id"],
                    len(result.get("transcription", "")),
                    ep["title"][:40],
                )
            except Exception:
                logger.exception("[%d/%d] Failed: %s", i, len(episodes), ep["episode_id"])


if __name__ == "__main__":
    main()
