"""LLM proofreading for Whisper transcripts.

Fixes ASR errors in Chinese philosophy podcast transcripts:
- Philosopher names (海德格尔, 阿伦特, 尼采, 萨特...)
- Book titles (《存在与时间》, 《理想国》...)
- Academic terms (存在主义, 现象学, 本体论...)
- Common Whisper mishearings

Uses Kimi K2.5 via api-proxy for best Chinese accuracy.
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
DEFAULT_MODEL = os.environ.get("PROOFREAD_MODEL", "kimi-k2.5")

SYSTEM_PROMPT = """你是一个中文播客转录校对专家。这是一档哲学播客《独树不成林》的 Whisper 语音识别文本。

你的任务是修正 ASR（语音识别）错误，只修正以下类型：

1. **人名错误**：哲学家、思想家、作家的名字
   - 常见人物：苏格拉底、柏拉图、亚里士多德、康德、黑格尔、尼采、海德格尔、阿伦特、萨特、波伏娃、福柯、德勒兹、维特根斯坦、罗尔斯、哈耶克、卢梭、洛克、霍布斯、马基雅维利、简·奥斯丁、张爱玲、鲁迅
   
2. **书名/作品名错误**：
   - 常见作品：《理想国》《存在与时间》《人的境况》《查拉图斯特拉如是说》《纯粹理性批判》《社会契约论》《利维坦》《君主论》《正义论》《通往奴役之路》《傲慢与偏见》

3. **学术术语错误**：
   - 常见术语：存在主义、现象学、本体论、认识论、形而上学、辩证法、解构主义、后现代主义、功利主义、自由主义、虚无主义、犬儒主义、斯多葛主义

4. **明显的同音字错误**：语音识别把正确的词听成了同音/近音的错误词

**规则**：
- 只修正明显的 ASR 错误，不改变原文的表述方式和口语风格
- 不修正语法、不润色文字、不改变句子结构
- 保留口语化表达（"就是说"、"然后呢"、"对吧"等）
- 如果不确定是否是错误，保持原文不变
- 输出完整的修正后文本

**输出格式**：
先输出修正后的完整文本，然后在末尾用 `---CHANGES---` 分隔，列出所有修改：
```
修正后的完整文本...

---CHANGES---
- "原文片段" → "修正后片段" (原因)
- "原文片段" → "修正后片段" (原因)
```

如果没有需要修正的内容，在 `---CHANGES---` 后写 "无修改"。"""


def proofread_text(
    text: str,
    title: str = "",
    api_base: str = DEFAULT_API_BASE,
    api_key: str = DEFAULT_API_KEY,
    model: str = DEFAULT_MODEL,
    timeout: float = 600.0,
    retries: int = 2,
) -> dict:
    """Proofread a transcript text using LLM.

    Returns:
        {
            "corrected": str,       # Full corrected text
            "changes": list[dict],  # [{original, corrected, reason}]
            "has_changes": bool,
            "model": str,
        }
    """
    user_msg = f"标题：{title}\n\n以下是需要校对的转录文本：\n\n{text}"

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
                        "max_tokens": 16384,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
            return _parse_response(content, model)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
            last_err = e
            if attempt < retries:
                import time
                wait = 10 * (attempt + 1)
                logger.warning("Attempt %d/%d timed out, retrying in %ds...", attempt + 1, 1 + retries, wait)
                time.sleep(wait)
    raise last_err  # type: ignore[misc]


def _parse_response(content: str, model: str) -> dict:
    """Parse LLM response into corrected text and change list."""
    if "---CHANGES---" in content:
        parts = content.split("---CHANGES---", 1)
        corrected = parts[0].strip()
        changes_text = parts[1].strip()
    else:
        corrected = content.strip()
        changes_text = ""

    changes = []
    if changes_text and "无修改" not in changes_text:
        for line in changes_text.split("\n"):
            line = line.strip().lstrip("- ")
            match = re.match(r'"(.+?)"\s*→\s*"(.+?)"\s*\((.+?)\)', line)
            if match:
                changes.append({
                    "original": match.group(1),
                    "corrected": match.group(2),
                    "reason": match.group(3),
                })

    return {
        "corrected": corrected,
        "changes": changes,
        "has_changes": len(changes) > 0,
        "model": model,
    }


def proofread_transcript(
    transcript_path: Path,
    output_dir: Path | None = None,
    **kwargs,
) -> dict:
    """Proofread a transcript JSON file.

    Reads the transcript, sends text to LLM, saves corrected version.
    Returns the proofread result dict.
    """
    with open(transcript_path) as f:
        data = json.load(f)

    text = data.get("transcription", "")
    title = data.get("title", "")

    if not text:
        logger.warning("Empty transcript: %s", transcript_path)
        return {"has_changes": False, "changes": []}

    logger.info("Proofreading: %s — %s (%d chars)", data.get("episode_id", ""), title, len(text))
    result = proofread_text(text, title=title, **kwargs)

    # Save proofread result
    out_dir = output_dir or transcript_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_id = data.get("episode_id", transcript_path.stem)

    # Save corrected transcript (same format as original + proofread metadata)
    corrected_data = {
        **data,
        "transcription": result["corrected"],
        "proofread": {
            "model": result["model"],
            "has_changes": result["has_changes"],
            "changes": result["changes"],
            "original_transcription": text,
        },
    }

    out_path = out_dir / f"{episode_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corrected_data, f, ensure_ascii=False, indent=2)

    # Save corrected markdown
    md_path = out_dir / f"{episode_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"- Episode ID: {episode_id}\n")
        f.write(f"- Date: {data.get('date', '')}\n")
        f.write(f"- Duration: {data.get('duration', 0):.0f}s\n")
        f.write(f"- Words: {len(result['corrected'].replace(' ', ''))}\n")
        f.write(f"- Proofread: {result['model']}\n")
        if result["changes"]:
            f.write(f"- Changes: {len(result['changes'])}\n")
        f.write(f"\n---\n\n{result['corrected']}\n")

    logger.info(
        "  Done: %d changes, saved to %s",
        len(result["changes"]),
        out_path,
    )
    return result
