#!/usr/bin/env python3
"""D-020: Aggregate style analysis from all episodes into a structured style profile.

Output: data/knowledge/style_profile.json
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("aggregate_style")


def main() -> None:
    analysis_dir = Path("data/analysis")
    output_dir = Path("data/knowledge")
    output_dir.mkdir(parents=True, exist_ok=True)

    styles = []
    for f in sorted(analysis_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        if data.get("skipped") or data.get("parse_error"):
            continue
        style = data.get("style", {})
        if style:
            styles.append(style)

    logger.info("Loaded style data from %d episodes", len(styles))

    # Aggregate opening patterns
    openings = Counter()
    for s in styles:
        op = s.get("opening_pattern", "")
        if op:
            openings[op] += 1

    # Aggregate structures
    structures = Counter()
    for s in styles:
        st = s.get("structure", "")
        if st:
            structures[st] += 1

    # Aggregate tones
    tones = Counter()
    for s in styles:
        t = s.get("tone", "")
        if t:
            tones[t] += 1

    # Aggregate catchphrases
    catchphrases = Counter()
    for s in styles:
        for cp in s.get("catchphrases", []):
            catchphrases[cp] += 1

    # Aggregate closing patterns
    closings = Counter()
    for s in styles:
        cl = s.get("closing_pattern", "")
        if cl:
            closings[cl] += 1

    # Aggregate narrative techniques
    techniques = Counter()
    for s in styles:
        for t in s.get("narrative_techniques", []):
            # Normalize: remove parenthetical examples
            import re
            base = re.sub(r"[（(].+?[）)]", "", t).strip()
            techniques[base] += 1

    profile = {
        "episode_count": len(styles),
        "opening_patterns": [
            {"pattern": k, "count": v}
            for k, v in openings.most_common(20)
        ],
        "structures": [
            {"pattern": k, "count": v}
            for k, v in structures.most_common(20)
        ],
        "tones": [
            {"tone": k, "count": v}
            for k, v in tones.most_common(20)
        ],
        "catchphrases": [
            {"phrase": k, "count": v}
            for k, v in catchphrases.most_common(50)
        ],
        "closing_patterns": [
            {"pattern": k, "count": v}
            for k, v in closings.most_common(20)
        ],
        "narrative_techniques": [
            {"technique": k, "count": v}
            for k, v in techniques.most_common(30)
        ],
    }

    out_path = output_dir / "style_profile.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    logger.info("Style profile saved to %s", out_path)
    logger.info("Top 5 catchphrases: %s", [cp["phrase"] for cp in profile["catchphrases"][:5]])
    logger.info("Top 5 techniques: %s", [t["technique"] for t in profile["narrative_techniques"][:5]])


if __name__ == "__main__":
    main()
