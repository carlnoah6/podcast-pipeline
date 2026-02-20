"""RSS feed parser for podcast feeds.

Supports:
  - Xiaoyuzhou (小宇宙) RSS: https://api.xiaoyuzhoufm.com/v1/podcast/rss/{id}
  - Apple Podcast RSS (via xyzfm.space or any standard podcast RSS)
  - Any standard podcast RSS feed URL

The parser extracts episode metadata and audio URLs from the feed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime

import feedparser

from src.models import Episode

logger = logging.getLogger(__name__)

XIAOYUZHOU_RSS_TEMPLATE = "https://api.xiaoyuzhoufm.com/v1/podcast/rss/{podcast_id}"


def build_rss_url(podcast_id: str) -> str:
    """Build the RSS feed URL for a Xiaoyuzhou podcast.

    Accepts either a bare podcast ID or a full URL.
    """
    if podcast_id.startswith("http"):
        return podcast_id
    return XIAOYUZHOU_RSS_TEMPLATE.format(podcast_id=podcast_id)


def _parse_duration(entry: dict) -> float:
    """Extract duration in seconds from an RSS entry.

    Tries itunes:duration (HH:MM:SS or seconds) then falls back to 0.
    """
    raw = entry.get("itunes_duration", "0")
    if raw is None:
        return 0.0
    raw = str(raw).strip()
    if ":" in raw:
        parts = raw.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
        except ValueError:
            return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _extract_audio(entry: dict) -> tuple[str, int]:
    """Return (audio_url, size_bytes) from enclosures."""
    for enc in entry.get("enclosures", []):
        url = enc.get("href", enc.get("url", ""))
        length = int(enc.get("length", 0) or 0)
        if url:
            return url, length
    # Fallback: look for links with audio type
    for link in entry.get("links", []):
        if "audio" in link.get("type", ""):
            return link["href"], int(link.get("length", 0) or 0)
    return "", 0


def _episode_id(entry: dict, podcast_id: str) -> str:
    """Derive a stable episode ID.

    Prefers the guid; falls back to a hash of the title + pub date.
    """
    guid = entry.get("id", "")
    if guid:
        # Xiaoyuzhou guids look like full URLs; extract the trailing ID
        match = re.search(r"[a-f0-9]{24}$", guid)
        if match:
            return match.group(0)
        return hashlib.sha256(guid.encode()).hexdigest()[:16]
    fallback = f"{podcast_id}:{entry.get('title', '')}:{entry.get('published', '')}"
    return hashlib.sha256(fallback.encode()).hexdigest()[:16]


def parse_feed(podcast_id: str, max_episodes: int = 0) -> list[Episode]:
    """Parse a Xiaoyuzhou RSS feed and return a list of Episodes.

    Args:
        podcast_id: Xiaoyuzhou podcast ID or full RSS URL.
        max_episodes: Limit the number of episodes returned (0 = all).

    Returns:
        List of Episode objects sorted by date descending.
    """
    url = build_rss_url(podcast_id)
    logger.info("Fetching RSS feed: %s", url)
    feed = feedparser.parse(url)

    if feed.bozo and not feed.entries:
        raise ValueError(f"Failed to parse RSS feed: {feed.bozo_exception}")

    episodes: list[Episode] = []
    for entry in feed.entries:
        audio_url, audio_size = _extract_audio(entry)
        if not audio_url:
            logger.warning("Skipping entry without audio: %s", entry.get("title", "?"))
            continue

        pub = entry.get("published_parsed") or entry.get("updated_parsed")
        date_str = datetime(*pub[:6]).strftime("%Y-%m-%d") if pub else ""

        ep = Episode(
            episode_id=_episode_id(entry, podcast_id),
            title=entry.get("title", ""),
            date=date_str,
            duration=_parse_duration(entry),
            audio_url=audio_url,
            audio_size=audio_size,
            description=entry.get("summary", ""),
        )
        episodes.append(ep)

    episodes.sort(key=lambda e: e.date, reverse=True)

    if max_episodes > 0:
        episodes = episodes[:max_episodes]

    logger.info("Parsed %d episodes from feed", len(episodes))
    return episodes
