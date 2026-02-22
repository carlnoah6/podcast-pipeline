"""Combined style analysis + entity extraction for podcast episodes.

One LLM pass per episode: extracts style patterns and named entities simultaneously.
Uses Kimi K2.5 via api-proxy.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = os.environ.get("API_BASE", "http://localhost:8180/v1")
DEFAULT_API_KEY = os.environ.get("API_KEY", "sk-podcast-proofread")
DEFAULT_MODEL = os.environ.get("ANALYZE_MODEL", "kimi-k2.5")

SYSTEM_PROMPT = """你是一个播客内容分析专家。你将分析一档中文知识类播客《独树不成林》的单集文本。

请从以下两个维度进行分析，输出严格的 JSON 格式。

## 1. 风格分析 (style)

分析这一集的表达风格：
- opening_pattern: 开场方式（如"以新闻事件引入"、"直接提出问题"、"讲一个故事"）
- structure: 论述结构（如"故事→分析→引用→总结"、"问题→历史→哲学→现实"）
- tone: 整体语气（如"理性客观"、"轻松幽默"、"严肃深沉"）
- catchphrases: 这一集中出现的口头禅或高频表达（数组，最多10个）
- closing_pattern: 结尾方式（如"开放式提问"、"总结观点"、"引用名言"）
- narrative_techniques: 使用的叙事技巧（如"类比"、"反问"、"举例"、"对比"）

## 2. 实体提取 (entities)

提取文本中提到的所有重要实体。不要预设类型，根据内容自然归类。常见类型包括但不限于：
- person: 人物（哲学家、政治家、作家、艺术家、历史人物等）
- work: 作品（书籍、论文、音乐、电影、艺术品等）
- concept: 概念/思想（哲学概念、政治理论、学术术语等）
- event: 历史事件
- place: 地点/国家/地区
- organization: 组织/机构/政党
- other: 其他重要实体

每个实体包含：
- name: 实体名称（标准中文名，如有外文名可附在括号里）
- type: 实体类型
- context: 在本集中的上下文（一句话说明这个实体在本集中是怎么被提到的）
- importance: high/medium/low（在本集中的重要程度）

## 输出格式

严格输出 JSON，不要有任何其他文字：

```json
{
  "style": {
    "opening_pattern": "...",
    "structure": "...",
    "tone": "...",
    "catchphrases": ["...", "..."],
    "closing_pattern": "...",
    "narrative_techniques": ["...", "..."]
  },
  "entities": [
    {"name": "...", "type": "person", "context": "...", "importance": "high"},
    {"name": "...", "type": "work", "context": "...", "importance": "medium"}
  ]
}
```"""


def analyze_episode(
    text: str,
    title: str = "",
    episode_id: str = "",
    api_base: str = DEFAULT_API_BASE,
    api_key: str = DEFAULT_API_KEY,
    model: str = DEFAULT_MODEL,
    timeout: float = 600.0,
    retries: int = 2,
) -> dict:
    """Analyze a single episode: style + entities in one LLM call.

    Returns:
        {
            "style": {...},
            "entities": [...],
            "model": str,
            "episode_id": str,
            "title": str,
        }
    """
    user_msg = f"标题：{title}\n\n以下是播客文本：\n\n{text}"

    last_err = None
    for attempt in range(1 + retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 8192,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
            return _parse_response(content, model, episode_id, title)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
            last_err = e
            if attempt < retries:
                import time
                wait = 10 * (attempt + 1)
                logger.warning("Attempt %d/%d failed (%s), retrying in %ds...", attempt + 1, 1 + retries, e, wait)
                time.sleep(wait)
    raise last_err  # type: ignore[misc]


def _parse_response(content: str, model: str, episode_id: str, title: str) -> dict:
    """Parse LLM JSON response."""
    # Extract JSON from markdown code block if present
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = content.strip()

    # Try parsing as-is first
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find the outermost JSON object with brace matching
        start = json_str.find("{")
        if start >= 0:
            depth = 0
            end = start
            for i, ch in enumerate(json_str[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            try:
                data = json.loads(json_str[start:end])
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON for %s. Raw (first 500): %s", episode_id, json_str[:500])
                data = {"style": {}, "entities": [], "parse_error": True}
        else:
            logger.error("No JSON object found for %s. Raw (first 500): %s", episode_id, json_str[:500])
            data = {"style": {}, "entities": [], "parse_error": True}
        data = {"style": {}, "entities": [], "parse_error": True}

    data["model"] = model
    data["episode_id"] = episode_id
    data["title"] = title
    return data


def analyze_transcript(
    transcript_path: Path,
    output_dir: Path | None = None,
    use_proofread: bool = True,
    proofread_dir: Path | None = None,
    **kwargs,
) -> dict:
    """Analyze a transcript file (or its proofread version).

    Args:
        transcript_path: Path to original transcript JSON.
        output_dir: Where to save analysis results.
        use_proofread: If True, prefer proofread text over original.
        proofread_dir: Directory containing proofread JSONs.
    """
    episode_id = transcript_path.stem

    # Prefer proofread text
    text = ""
    title = ""
    if use_proofread and proofread_dir:
        pr_path = proofread_dir / f"{episode_id}.json"
        if pr_path.exists():
            with open(pr_path) as f:
                data = json.load(f)
            text = data.get("transcription", "")
            title = data.get("title", "")

    if not text:
        with open(transcript_path) as f:
            data = json.load(f)
        text = data.get("transcription", "")
        title = data.get("title", "")

    if not text or len(text) < 50:
        logger.warning("Skipping %s: text too short (%d chars)", episode_id, len(text))
        return {"episode_id": episode_id, "skipped": True}

    logger.info("Analyzing: %s — %s (%d chars)", episode_id, title, len(text))
    result = analyze_episode(text, title=title, episode_id=episode_id, **kwargs)

    # Save
    out_dir = output_dir or transcript_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{episode_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    entity_count = len(result.get("entities", []))
    logger.info("  Done: %d entities, saved to %s", entity_count, out_path)
    return result
