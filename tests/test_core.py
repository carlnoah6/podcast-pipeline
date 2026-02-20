"""Tests for RSS parser, episode filters, and formatter.

These tests use mocked data and do not require network access or Modal.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.ingest.filter import (
    filter_episodes,
    is_dialogue,
    is_paid_or_preview,
    is_special,
)
from src.models import Episode, TranscriptionResult
from src.postprocess.formatter import dict_to_result, save_transcript

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_episode(**overrides) -> Episode:
    defaults = dict(
        episode_id="ep001",
        title="普通单口节目",
        date="2025-01-15",
        duration=1800.0,
        audio_url="https://example.com/ep001.mp3",
        audio_size=30_000_000,
    )
    defaults.update(overrides)
    return Episode(**defaults)


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------


class TestPaidFilter:
    def test_small_file_is_paid(self):
        ep = _make_episode(audio_size=500_000)  # 500 KB < 1 MB
        assert is_paid_or_preview(ep) is True

    def test_short_duration_is_paid(self):
        ep = _make_episode(duration=30.0)  # 30s < 60s
        assert is_paid_or_preview(ep) is True

    def test_normal_episode_passes(self):
        ep = _make_episode()
        assert is_paid_or_preview(ep) is False

    def test_zero_size_passes(self):
        """When size is unknown (0), don't filter."""
        ep = _make_episode(audio_size=0)
        assert is_paid_or_preview(ep) is False

    def test_zero_duration_passes(self):
        """When duration is unknown (0), don't filter."""
        ep = _make_episode(duration=0)
        assert is_paid_or_preview(ep) is False


class TestDialogueFilter:
    @pytest.mark.parametrize(
        "keyword",
        [
            "对话",
            "对谈",
            "访谈",
            "聊聊",
            "圆桌",
            "连麦",
            "串台",
            "嘉宾",
            "特邀",
            "专访",
        ],
    )
    def test_dialogue_keywords(self, keyword):
        ep = _make_episode(title=f"和张三{keyword}创业那些事")
        assert is_dialogue(ep) is True

    def test_normal_title_passes(self):
        ep = _make_episode(title="我的2024年度总结")
        assert is_dialogue(ep) is False


class TestSpecialFilter:
    def test_intro_episode(self):
        ep = _make_episode(title="0-播客介绍")
        assert is_special(ep) is True

    def test_trailer(self):
        ep = _make_episode(title="Trailer: 新节目预告")
        assert is_special(ep) is True

    def test_normal_passes(self):
        ep = _make_episode(title="293-现代哲学如何在重复中寻找意义")
        assert is_special(ep) is False


class TestFilterEpisodes:
    def test_combined_filter(self):
        episodes = [
            _make_episode(episode_id="ok1"),
            _make_episode(episode_id="paid", audio_size=100_000),
            _make_episode(episode_id="short", duration=10.0),
            _make_episode(episode_id="talk", title="和朋友聊聊AI"),
            _make_episode(episode_id="intro", title="0-播客介绍"),
            _make_episode(episode_id="ok2"),
        ]
        result = filter_episodes(episodes)
        ids = [e.episode_id for e in result]
        assert ids == ["ok1", "ok2"]

    def test_skip_flags(self):
        episodes = [
            _make_episode(episode_id="paid", audio_size=100_000),
            _make_episode(episode_id="talk", title="专访某大佬"),
            _make_episode(episode_id="intro", title="0-播客介绍"),
        ]
        # Disable all filters
        result = filter_episodes(
            episodes, skip_paid=False, skip_dialogue=False, skip_special=False
        )
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestTranscriptionResult:
    def test_auto_word_count(self):
        tr = TranscriptionResult(
            episode_id="ep001",
            title="Test",
            date="2025-01-01",
            duration=600.0,
            transcription="你好世界 hello",
        )
        # Chinese chars + English letters, spaces removed
        assert tr.word_count == len("你好世界hello")

    def test_to_json_roundtrip(self):
        tr = TranscriptionResult(
            episode_id="ep001",
            title="Test Episode",
            date="2025-01-01",
            duration=600.0,
            transcription="测试内容",
        )
        data = json.loads(tr.to_json())
        assert data["episode_id"] == "ep001"
        assert data["word_count"] == 4

    def test_to_markdown(self):
        tr = TranscriptionResult(
            episode_id="ep001",
            title="My Episode",
            date="2025-01-01",
            duration=600.0,
            transcription="Hello world",
        )
        md = tr.to_markdown()
        assert "# My Episode" in md
        assert "Hello world" in md


# ---------------------------------------------------------------------------
# Formatter tests
# ---------------------------------------------------------------------------


class TestFormatter:
    def test_save_transcript_creates_files(self):
        raw = {
            "episode_id": "ep_test",
            "title": "Test",
            "date": "2025-06-01",
            "duration": 300.0,
            "transcription": "测试转录内容",
            "word_count": 6,
            "segments": [],
            "language": "zh",
            "model": "large-v3",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            json_path, md_path = save_transcript(raw, output_dir=out)
            assert json_path.exists()
            assert md_path.exists()

            data = json.loads(json_path.read_text())
            assert data["episode_id"] == "ep_test"
            assert data["word_count"] == 6

            md = md_path.read_text()
            assert "# Test" in md

    def test_dict_to_result(self):
        raw = {
            "episode_id": "x",
            "title": "T",
            "date": "2025-01-01",
            "duration": 100.0,
            "transcription": "abc",
        }
        r = dict_to_result(raw)
        assert r.episode_id == "x"
        assert r.word_count == 3
