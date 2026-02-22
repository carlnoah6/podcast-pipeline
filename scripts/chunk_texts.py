#!/usr/bin/env python3
"""K-006: Chunk proofread transcripts for RAG embedding.

Splits each episode into overlapping chunks of ~500-800 chars.
Output: data/knowledge/chunks.jsonl
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("chunk_texts")

CHUNK_SIZE = 500  # target chars per chunk
OVERLAP = 80      # overlap between chunks


def split_into_chunks(text: str, episode_id: str, title: str) -> list[dict]:
    """Split text into overlapping chunks."""
    # Try to split on sentence boundaries (。！？)
    sentences = re.split(r"(?<=[。！？])", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""
    chunk_idx = 0

    for sent in sentences:
        if len(current) + len(sent) > CHUNK_SIZE and current:
            chunks.append({
                "chunk_id": f"{episode_id}_{chunk_idx:04d}",
                "episode_id": episode_id,
                "title": title,
                "text": current.strip(),
                "chunk_index": chunk_idx,
                "char_count": len(current.strip()),
            })
            # Keep overlap
            overlap_text = current[-OVERLAP:] if len(current) > OVERLAP else current
            current = overlap_text + sent
            chunk_idx += 1
        else:
            current += sent

    # Last chunk
    if current.strip():
        chunks.append({
            "chunk_id": f"{episode_id}_{chunk_idx:04d}",
            "episode_id": episode_id,
            "title": title,
            "text": current.strip(),
            "chunk_index": chunk_idx,
            "char_count": len(current.strip()),
        })

    return chunks


def main() -> None:
    proofread_dir = Path("data/proofread")
    paid_path = Path("data/paid_episodes.json")
    output_dir = Path("data/knowledge")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load paid episode list
    paid_ids = set()
    if paid_path.exists():
        with open(paid_path) as f:
            paid_ids = set(json.load(f))

    all_chunks = []
    episode_count = 0

    for f in sorted(proofread_dir.glob("*.json")):
        episode_id = f.stem
        if episode_id in paid_ids:
            continue

        with open(f) as fh:
            data = json.load(fh)

        text = data.get("transcription", "")
        title = data.get("title", "")

        if len(text) < 100:
            continue

        chunks = split_into_chunks(text, episode_id, title)
        all_chunks.extend(chunks)
        episode_count += 1

    # Save as JSONL
    out_path = output_dir / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(
        "Chunked %d episodes into %d chunks (avg %.0f chars/chunk)",
        episode_count, len(all_chunks),
        sum(c["char_count"] for c in all_chunks) / max(len(all_chunks), 1),
    )
    logger.info("Output: %s", out_path)


if __name__ == "__main__":
    main()
