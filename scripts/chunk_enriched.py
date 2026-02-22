#!/usr/bin/env python3
"""K-005b: Chunk enriched entity knowledge for FAISS indexing.

Converts enriched/*.json into chunks suitable for embedding.
Output: data/knowledge/enriched_chunks.jsonl
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("chunk_enriched")


def main() -> None:
    enriched_dir = Path("data/knowledge/enriched")
    output_dir = Path("data/knowledge")
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for f in sorted(enriched_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)

        name = data.get("name", f.stem)
        entity_type = data.get("type", "unknown")
        normalized = data.get("entity_normalized", f.stem)

        # Chunk 1: Summary + core ideas
        summary = data.get("summary", "")
        core_ideas = data.get("core_ideas", [])
        if summary:
            text = f"【{name}】{summary}"
            if core_ideas:
                text += "\n核心思想：" + "；".join(
                    idea.replace("**", "") for idea in core_ideas
                )
            chunks.append({
                "chunk_id": f"enriched_{normalized}_summary",
                "episode_id": f"enriched_{normalized}",
                "title": f"[知识库] {name}",
                "text": text,
                "char_count": len(text),
                "chunk_type": "enriched_summary",
                "entity_name": name,
                "entity_type": entity_type,
            })

        # Chunk 2: Quotes (if any)
        quotes = data.get("quotes", [])
        if quotes:
            quote_texts = []
            for q in quotes:
                if isinstance(q, str):
                    quote_texts.append(q)
                elif isinstance(q, dict):
                    qt = q.get("text", "")
                    src = q.get("source", "")
                    quote_texts.append(f"{qt}——{src}" if src else qt)
            if quote_texts:
                text = f"【{name}·名言】" + "\n".join(f"• {q}" for q in quote_texts)
                chunks.append({
                    "chunk_id": f"enriched_{normalized}_quotes",
                    "episode_id": f"enriched_{normalized}",
                    "title": f"[名言] {name}",
                    "text": text,
                    "char_count": len(text),
                    "chunk_type": "enriched_quotes",
                    "entity_name": name,
                    "entity_type": entity_type,
                })

        # Chunk 3: Key facts
        facts = data.get("key_facts", [])
        if facts:
            text = f"【{name}·事实】" + "；".join(facts)
            chunks.append({
                "chunk_id": f"enriched_{normalized}_facts",
                "episode_id": f"enriched_{normalized}",
                "title": f"[事实] {name}",
                "text": text,
                "char_count": len(text),
                "chunk_type": "enriched_facts",
                "entity_name": name,
                "entity_type": entity_type,
            })

    # Save
    out_path = output_dir / "enriched_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(
        "Chunked %d enriched entities into %d chunks (summary=%d, quotes=%d, facts=%d)",
        len(list(enriched_dir.glob("*.json"))),
        len(chunks),
        sum(1 for c in chunks if c["chunk_type"] == "enriched_summary"),
        sum(1 for c in chunks if c["chunk_type"] == "enriched_quotes"),
        sum(1 for c in chunks if c["chunk_type"] == "enriched_facts"),
    )


if __name__ == "__main__":
    main()
