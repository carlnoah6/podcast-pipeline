#!/usr/bin/env python3
"""K-003: Aggregate entities from per-episode analysis into a unified entity database.

Deduplicates by name, merges contexts, counts episode appearances.
Output: data/knowledge/entities.json
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("aggregate_entities")


def normalize_name(name: str) -> str:
    """Normalize entity name for dedup."""
    # Remove parenthetical foreign names for matching
    base = re.sub(r"[（(].+?[）)]", "", name).strip()
    # Normalize whitespace
    base = re.sub(r"\s+", "", base)
    return base


def main() -> None:
    analysis_dir = Path("data/analysis")
    output_dir = Path("data/knowledge")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all entities
    raw_entities: list[dict] = []
    episode_count = 0

    for f in sorted(analysis_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        if data.get("skipped") or data.get("parse_error"):
            continue
        episode_count += 1
        episode_id = data.get("episode_id", f.stem)
        title = data.get("title", "")
        for ent in data.get("entities", []):
            ent["_episode_id"] = episode_id
            ent["_episode_title"] = title
            raw_entities.append(ent)

    logger.info("Loaded %d raw entities from %d episodes", len(raw_entities), episode_count)

    # Group by normalized name
    groups: dict[str, list[dict]] = defaultdict(list)
    for ent in raw_entities:
        key = normalize_name(ent.get("name", ""))
        if key:
            groups[key].append(ent)

    # Build unified entity database
    entities = []
    for key, group in sorted(groups.items(), key=lambda x: -len(x[1])):
        # Pick the most common full name (with foreign name if present)
        name_counts: dict[str, int] = defaultdict(int)
        for ent in group:
            name_counts[ent["name"]] += 1
        canonical_name = max(name_counts, key=name_counts.get)

        # Most common type
        type_counts: dict[str, int] = defaultdict(int)
        for ent in group:
            type_counts[ent.get("type", "other")] += 1
        canonical_type = max(type_counts, key=type_counts.get)

        # Collect unique episodes
        episodes = []
        seen_eps = set()
        for ent in group:
            ep_id = ent["_episode_id"]
            if ep_id not in seen_eps:
                seen_eps.add(ep_id)
                episodes.append({
                    "episode_id": ep_id,
                    "title": ent["_episode_title"],
                    "context": ent.get("context", ""),
                    "importance": ent.get("importance", "low"),
                })

        # Highest importance across episodes
        imp_order = {"high": 3, "medium": 2, "low": 1}
        max_importance = max(
            (ent.get("importance", "low") for ent in group),
            key=lambda x: imp_order.get(x, 0),
        )

        entities.append({
            "name": canonical_name,
            "normalized": key,
            "type": canonical_type,
            "episode_count": len(episodes),
            "importance": max_importance,
            "episodes": episodes,
        })

    # Sort by episode count (most referenced first)
    entities.sort(key=lambda x: -x["episode_count"])

    # Save
    out_path = output_dir / "entities.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

    # Stats
    logger.info("Unique entities: %d", len(entities))
    logger.info("Top 20 most referenced:")
    for ent in entities[:20]:
        logger.info(
            "  [%s] %s — %d episodes (%s)",
            ent["type"], ent["name"], ent["episode_count"], ent["importance"],
        )

    # Type distribution
    from collections import Counter
    type_dist = Counter(e["type"] for e in entities)
    logger.info("Type distribution: %s", dict(type_dist.most_common()))


if __name__ == "__main__":
    main()
