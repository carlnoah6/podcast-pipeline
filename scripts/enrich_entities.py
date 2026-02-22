#!/usr/bin/env python3
"""K-004 v2: Enrich entities with real external knowledge.

Three layers:
1. WikiQuote (zh) — quotes with source attribution
2. SEP (Stanford Encyclopedia of Philosophy) — deep academic analysis
3. LLM synthesis — structure and summarize the collected knowledge

Output: data/knowledge/enriched/<entity_normalized>.json
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("enrich_v2")

API_BASE = os.environ.get("API_BASE", "http://localhost:8180/v1")
API_KEY = os.environ.get("API_KEY", "sk-luna-2026-openclaw")
MODEL = os.environ.get("ENRICH_MODEL", "gemini-2.5-flash")
WORKERS = int(os.environ.get("WORKERS", "20"))

TARGET_TYPES = {"person", "work", "event", "organization"}
MIN_EPISODES = 3


# ── Layer 1: WikiQuote ──────────────────────────────────────────────

def fetch_wikiquote(name: str) -> list[dict]:
    """Fetch quotes from Chinese WikiQuote."""
    encoded = urllib.parse.quote(name)
    url = f"https://zh.wikiquote.org/w/api.php?action=parse&page={encoded}&prop=wikitext&format=json&redirects=1"
    try:
        resp = httpx.get(
            url, timeout=15, follow_redirects=True,
            headers={"User-Agent": "PodcastKnowledgeBot/1.0 (research project)"},
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
        if not wikitext or "missingtitle" in str(data.get("error", "")):
            return []
        return _parse_wikiquote(wikitext)
    except Exception as e:
        logger.debug("WikiQuote error for %s: %s", name, e)
        return []


def _parse_wikiquote(wikitext: str) -> list[dict]:
    """Parse WikiQuote wikitext into structured quotes."""
    quotes = []
    current_section = ""
    for line in wikitext.split("\n"):
        line = line.strip()
        # Section headers
        m = re.match(r"^={2,3}(.+?)={2,3}$", line)
        if m:
            current_section = m.group(1).strip()
            continue
        # Quote lines (start with *)
        if line.startswith("*") and not line.startswith("**"):
            text = line.lstrip("*").strip()
            # Remove wiki markup
            text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
            text = re.sub(r"'''?", "", text)
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\{\{[^}]*\}\}", "", text)
            text = re.sub(r"<ref[^>]*>.*?</ref>", "", text)
            text = text.strip()
            if len(text) > 10 and len(text) < 500:
                # Extract source (often after ——)
                source = ""
                if "——" in text:
                    parts = text.split("——", 1)
                    text = parts[0].strip()
                    source = parts[1].strip().strip("''")
                quotes.append({
                    "text": text,
                    "source": source,
                    "section": current_section,
                })
    return quotes


# ── Layer 2: SEP ────────────────────────────────────────────────────

# Mapping of Chinese names to SEP entry slugs
SEP_SLUGS = {
    "卢梭": "rousseau",
    "亚里士多德": "aristotle",
    "柏拉图": "plato",
    "康德": "kant-moral",
    "尼采": "nietzsche",
    "黑格尔": "hegel",
    "马克思": "marx",
    "苏格拉底": "socrates",
    "托克维尔": "tocqueville",
    "孟德斯鸠": "montesquieu",
    "霍布斯": "hobbes",
    "洛克": "locke-political",
    "笛卡尔": "descartes",
    "休谟": "hume",
    "密尔": "mill",
    "伏尔泰": "voltaire",
    "斯宾诺莎": "spinoza",
    "莱布尼茨": "leibniz",
    "叔本华": "schopenhauer",
    "萨特": "sartre",
    "加缪": "camus",
    "福柯": "foucault",
    "汉娜·阿伦特": "arendt",
    "阿伦特": "arendt",
    "韦伯": "weber",
    "马基雅维利": "machiavelli",
    "陀思妥耶夫斯基": "dostoevsky",
    "普鲁塔克": "plutarch",
    "西塞罗": "cicero",
    "奥古斯丁": "augustine",
    "阿奎那": "aquinas",
    "维特根斯坦": "wittgenstein",
    "海德格尔": "heidegger",
    "罗尔斯": "rawls",
    "哈贝马斯": "habermas",
    "伯克": "burke",
    "法国大革命": "french-revolution",
    "社会契约论": "contractarianism",
    "功利主义": "utilitarianism",
    "存在主义": "existentialism",
    "自由主义": "liberalism",
    "共和主义": "republicanism",
    "民主": "democracy",
}


def fetch_sep(name: str) -> str:
    """Fetch SEP article summary (first ~2000 chars)."""
    slug = SEP_SLUGS.get(name)
    if not slug:
        return ""
    url = f"https://plato.stanford.edu/entries/{slug}/"
    try:
        resp = httpx.get(
            url, timeout=15, follow_redirects=True,
            headers={"User-Agent": "PodcastKnowledgeBot/1.0 (research project)"},
        )
        if resp.status_code != 200:
            return ""
        # Extract text content (rough)
        html = resp.text
        # Get preamble (before first <h2>)
        m = re.search(r'<div id="preamble">(.*?)</div>', html, re.DOTALL)
        if not m:
            m = re.search(r'<div id="aueditable">(.*?)<h2', html, re.DOTALL)
        if m:
            text = re.sub(r"<[^>]+>", "", m.group(1))
            text = re.sub(r"\s+", " ", text).strip()
            return text[:3000]
        return ""
    except Exception as e:
        logger.debug("SEP error for %s: %s", name, e)
        return ""


# ── Layer 3: LLM Synthesis ──────────────────────────────────────────

SYNTHESIS_PROMPT = """你是一个人文知识库助手。请根据以下收集到的真实资料，为这个实体生成结构化的知识卡片。

实体名称: {name}
实体类型: {type}
在播客中出现 {episode_count} 次

{wikiquote_section}

{sep_section}

{context_section}

请输出 JSON（只输出 JSON，不要其他文字）：
{{
  "name": "{name}",
  "type": "{type}",
  "summary": "200-400字的深度介绍，重点关注人文/哲学/政治维度",
  "key_facts": ["5-8个具体事实"],
  "quotes": ["从上面的名言中精选最经典的5-10条，保留原文和出处"],
  "core_ideas": ["3-5个核心思想/概念，每个用一句话解释"],
  "related_entities": ["相关的人物/作品/事件"],
  "time_period": "相关时间段",
  "significance": "为什么在人文领域重要（一句话）"
}}

要求：
1. quotes 必须来自上面提供的真实名言，不要编造
2. 如果没有 WikiQuote 数据，quotes 字段留空数组
3. summary 和 core_ideas 可以结合 SEP 资料和你的知识
4. 保持学术准确性"""


def synthesize(entity: dict, quotes: list[dict], sep_text: str) -> dict | None:
    """Use LLM to synthesize collected knowledge into structured card."""
    # Build prompt sections
    wikiquote_section = ""
    if quotes:
        lines = [f"- 「{q['text']}」" + (f" ——{q['source']}" if q['source'] else "") for q in quotes[:20]]
        wikiquote_section = "【WikiQuote 名言】\n" + "\n".join(lines)
    else:
        wikiquote_section = "【WikiQuote 名言】无数据"

    sep_section = ""
    if sep_text:
        sep_section = f"【Stanford Encyclopedia of Philosophy】\n{sep_text[:2000]}"
    else:
        sep_section = "【SEP】无数据"

    contexts = []
    for ep in entity.get("episodes", [])[:3]:
        ctx = ep.get("context", "")
        if ctx:
            contexts.append(f"[{ep.get('title', '')}] {ctx}")
    context_section = "【播客上下文】\n" + "\n".join(contexts) if contexts else "【播客上下文】无"

    prompt = SYNTHESIS_PROMPT.format(
        name=entity["name"],
        type=entity["type"],
        episode_count=entity["episode_count"],
        wikiquote_section=wikiquote_section,
        sep_section=sep_section,
        context_section=context_section,
    )

    try:
        resp = httpx.post(
            f"{API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4096,
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _parse_json(content)
    except Exception as e:
        logger.error("LLM error for %s: %s", entity["name"], e)
        return None


def _parse_json(text: str) -> dict | None:
    """Extract JSON from LLM response."""
    # Try code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Brace matching
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ── Main Pipeline ───────────────────────────────────────────────────

def enrich_one(entity: dict, output_dir: Path) -> dict | None:
    """Full enrichment pipeline for one entity."""
    normalized = entity["normalized"]
    out_path = output_dir / f"{normalized}.json"
    if out_path.exists():
        return None

    name = entity["name"]

    # Layer 1: WikiQuote
    quotes = fetch_wikiquote(name)

    # Layer 2: SEP
    sep_text = fetch_sep(name)

    # Layer 3: LLM synthesis
    result = synthesize(entity, quotes, sep_text)

    if result:
        result["_sources"] = {
            "wikiquote_count": len(quotes),
            "sep_available": bool(sep_text),
        }
        result["entity_normalized"] = normalized
        result["episode_count"] = entity["episode_count"]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-episodes", type=int, default=MIN_EPISODES)
    args = parser.parse_args()

    entities_path = Path("data/knowledge/entities.json")
    output_dir = Path("data/knowledge/enriched")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(entities_path) as f:
        entities = json.load(f)

    targets = [
        e for e in entities
        if e["type"] in TARGET_TYPES
        and (e["importance"] == "high" or e["episode_count"] >= args.min_episodes)
    ]

    existing = {f.stem for f in output_dir.glob("*.json")}
    pending = [e for e in targets if e["normalized"] not in existing]
    if args.limit > 0:
        pending = pending[: args.limit]

    logger.info(
        "Entities: %d total, %d target, %d done, %d pending",
        len(entities), len(targets), len(existing), len(pending),
    )

    if not pending:
        logger.info("Nothing to do!")
        return

    # Phase 1: Fetch external data sequentially (WikiQuote rate-limited)
    logger.info("Phase 1: Fetching external data (WikiQuote + SEP)...")
    prefetched = {}
    for i, entity in enumerate(pending):
        name = entity["name"]
        quotes = fetch_wikiquote(name)
        time.sleep(0.5)  # Rate limit WikiQuote
        sep_text = fetch_sep(name)
        prefetched[entity["normalized"]] = {"quotes": quotes, "sep": sep_text}
        if (i + 1) % 50 == 0:
            wq = sum(1 for v in prefetched.values() if v["quotes"])
            sp = sum(1 for v in prefetched.values() if v["sep"])
            logger.info("  Fetched %d/%d (WikiQuote: %d, SEP: %d)", i + 1, len(pending), wq, sp)

    wq_total = sum(1 for v in prefetched.values() if v["quotes"])
    sep_total = sum(1 for v in prefetched.values() if v["sep"])
    logger.info(
        "Phase 1 done: WikiQuote=%d, SEP=%d out of %d entities",
        wq_total, sep_total, len(pending),
    )

    # Phase 2: LLM synthesis in parallel
    logger.info("Phase 2: LLM synthesis (%d workers)...", args.workers)
    done = 0
    failed = 0

    def synth_one(entity: dict) -> dict | None:
        normalized = entity["normalized"]
        out_path = output_dir / f"{normalized}.json"
        if out_path.exists():
            return None
        data = prefetched[normalized]
        result = synthesize(entity, data["quotes"], data["sep"])
        if result:
            result["_sources"] = {
                "wikiquote_count": len(data["quotes"]),
                "sep_available": bool(data["sep"]),
            }
            result["entity_normalized"] = normalized
            result["episode_count"] = entity["episode_count"]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return result
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(synth_one, e): e for e in pending}
        for future in as_completed(futures):
            entity = futures[future]
            try:
                result = future.result()
                if result:
                    done += 1
                    if done % 50 == 0:
                        logger.info("  Synthesized: %d/%d", done, len(pending))
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logger.error("Exception for %s: %s", entity["name"], e)

    logger.info(
        "Complete: %d enriched, %d failed. Sources: WikiQuote=%d, SEP=%d",
        done, failed, wq_total, sep_total,
    )


if __name__ == "__main__":
    main()
