#!/usr/bin/env python3
"""K-007 + K-008: Generate embeddings with bge-m3 on Modal and build FAISS index.

Reads chunks.jsonl + enriched_chunks.jsonl, generates embeddings, saves FAISS index.
Output: data/knowledge/faiss.index + data/knowledge/chunk_ids.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import modal

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("build_faiss")

app = modal.App("podcast-embedding")

embed_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("sentence-transformers>=2.2", "faiss-cpu", "torch>=2.1", "numpy")
)


@app.function(image=embed_image, gpu="A10G", timeout=1800)
def embed_and_build_index(chunks: list[dict]) -> dict:
    """Generate embeddings and build FAISS index on Modal GPU."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3")

    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    dim = embeddings.shape[1]
    print(f"Building FAISS index (dim={dim})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    import io
    buf = io.BytesIO()
    faiss.write_index(index, faiss.BufferedIOWriter(faiss.PyCallbackIOWriter(buf.write)))

    return {
        "index_bytes": buf.getvalue(),
        "dim": dim,
        "count": len(texts),
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }


def main() -> None:
    output_dir = Path("data/knowledge")

    # Load podcast transcript chunks
    chunks = []
    chunks_path = output_dir / "chunks.jsonl"
    if chunks_path.exists():
        with open(chunks_path) as f:
            for line in f:
                chunks.append(json.loads(line))
    logger.info("Podcast chunks: %d", len(chunks))

    # Load enriched knowledge chunks
    enriched_path = output_dir / "enriched_chunks.jsonl"
    enriched_count = 0
    if enriched_path.exists():
        with open(enriched_path) as f:
            for line in f:
                chunks.append(json.loads(line))
                enriched_count += 1
    logger.info("Enriched chunks: %d", enriched_count)
    logger.info("Total chunks to index: %d", len(chunks))

    # Run on Modal
    with app.run():
        result = embed_and_build_index.remote(chunks)

    # Save FAISS index
    index_path = output_dir / "faiss.index"
    with open(index_path, "wb") as f:
        f.write(result["index_bytes"])

    # Save chunk ID mapping
    ids_path = output_dir / "chunk_ids.json"
    with open(ids_path, "w") as f:
        json.dump(result["chunk_ids"], f)

    logger.info(
        "FAISS index: %d vectors, dim=%d, saved to %s",
        result["count"], result["dim"], index_path,
    )


if __name__ == "__main__":
    main()
