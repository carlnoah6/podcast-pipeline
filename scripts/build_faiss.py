#!/usr/bin/env python3
"""K-007 + K-008: Generate embeddings with bge-m3 on Modal and build FAISS index.

Reads chunks.jsonl, generates embeddings via Modal GPU, saves FAISS index.
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


@app.function(image=embed_image, gpu="T4", timeout=1800)
def embed_and_build_index(chunks: list[dict]) -> dict:
    """Generate embeddings and build FAISS index on Modal GPU."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    logger.info("Loading bge-m3 model...")
    model = SentenceTransformer("BAAI/bge-m3")

    texts = [c["text"] for c in chunks]
    logger.info("Encoding %d chunks...", len(texts))

    # Batch encode
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Build FAISS index
    dim = embeddings.shape[1]
    logger.info("Building FAISS index (dim=%d)...", dim)
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim since normalized)
    index.add(embeddings.astype(np.float32))

    # Serialize
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
    chunks_path = Path("data/knowledge/chunks.jsonl")
    output_dir = Path("data/knowledge")

    # Load chunks
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            chunks.append(json.loads(line))

    logger.info("Loaded %d chunks", len(chunks))

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
        "FAISS index built: %d vectors, dim=%d, saved to %s",
        result["count"], result["dim"], index_path,
    )


if __name__ == "__main__":
    main()
