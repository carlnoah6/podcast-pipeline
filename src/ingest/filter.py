"""Episode filters for the podcast pipeline.

Filters out episodes that should not be transcribed:
  1. Paid/preview content — audio file < 1 MB or duration < 60s
  2. Conversation/interview episodes — title contains dialogue keywords
"""

from __future__ import annotations

import logging
import re

from src.models import Episode

logger = logging.getLogger(__name__)

# Minimum thresholds for valid free episodes
MIN_AUDIO_BYTES = 1_000_000  # 1 MB
MIN_DURATION_SECONDS = 60  # 1 minute

# Chinese keywords indicating conversation/interview format.
# Matches: 对话, 对谈, 访谈, 聊聊, 圆桌, 连麦, 串台, 嘉宾, 特邀, 专访
DIALOGUE_KEYWORDS = re.compile(r"对话|对谈|访谈|聊聊|圆桌|连麦|串台|嘉宾|特邀|专访")


def is_paid_or_preview(episode: Episode) -> bool:
    """Return True if the episode looks like paid/preview content.

    Heuristic: very small file size or very short duration usually means
    the RSS entry only contains a teaser clip for a paid episode.
    """
    if episode.audio_size > 0 and episode.audio_size < MIN_AUDIO_BYTES:
        logger.debug("Filtered (small file %d B): %s", episode.audio_size, episode.title)
        return True
    if episode.duration > 0 and episode.duration < MIN_DURATION_SECONDS:
        logger.debug("Filtered (short %.0fs): %s", episode.duration, episode.title)
        return True
    return False


def is_dialogue(episode: Episode) -> bool:
    """Return True if the episode title suggests a conversation format."""
    if DIALOGUE_KEYWORDS.search(episode.title):
        logger.debug("Filtered (dialogue keyword): %s", episode.title)
        return True
    return False


def filter_episodes(
    episodes: list[Episode],
    *,
    skip_paid: bool = True,
    skip_dialogue: bool = True,
) -> list[Episode]:
    """Apply all filters and return episodes suitable for transcription.

    Args:
        episodes: Raw episode list from the RSS parser.
        skip_paid: Drop episodes that look like paid previews.
        skip_dialogue: Drop episodes with dialogue keywords in the title.

    Returns:
        Filtered list of episodes.
    """
    result: list[Episode] = []
    paid_count = 0
    dialogue_count = 0

    for ep in episodes:
        if skip_paid and is_paid_or_preview(ep):
            paid_count += 1
            continue
        if skip_dialogue and is_dialogue(ep):
            dialogue_count += 1
            continue
        result.append(ep)

    logger.info(
        "Filter: %d input → %d kept (paid/preview=%d, dialogue=%d)",
        len(episodes),
        len(result),
        paid_count,
        dialogue_count,
    )
    return result
