import json
import os
import time
from openai import OpenAI
from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    KEYWORD_BATCH_SIZE, KEYWORDS_CACHE_FILE,
)


client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


SYSTEM_PROMPT = """你是一个专业的中文关键词提取助手，专门处理短视频标题。

请从给定的短视频标题中提取关键词，提取规则：
1. 优先提取：时间词（如"昨天"、"2024年"、"春节"）、地点词（如"北京"、"某地"）、人物/角色（如"警察"、"学生"）、核心事件词（如"爆炸"、"救援"、"获奖"）
2. 每个标题提取 3-8 个关键词
3. 关键词应简洁，一般为 2-4 个汉字
4. 去除停用词和无意义的词

返回严格的 JSON 格式，键为标题的序号（从0开始），值为关键词列表：
{"0": ["关键词1", "关键词2"], "1": ["关键词A", "关键词B"], ...}

只返回 JSON，不要有任何其他内容。"""


# ── 缓存 I/O ──────────────────────────────────────────────────────────────────

def _load_cache() -> dict[str, list[str]]:
    """从磁盘加载缓存，返回 {标题: 关键词列表}"""
    if os.path.exists(KEYWORDS_CACHE_FILE):
        try:
            with open(KEYWORDS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict[str, list[str]]) -> None:
    """将缓存写回磁盘"""
    os.makedirs(os.path.dirname(KEYWORDS_CACHE_FILE) or ".", exist_ok=True)
    with open(KEYWORDS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── API 调用 ──────────────────────────────────────────────────────────────────

def _extract_batch_from_api(titles: list[str]) -> dict[int, list[str]]:
    """
    对一批标题调用 API 提取关键词。
    返回 {批内序号: [关键词]} 的字典。
    """
    numbered_titles = "\n".join(
        f"{i}: {title}" for i, title in enumerate(titles)
    )
    user_prompt = f"请从以下短视频标题中提取关键词：\n\n{numbered_titles}"

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    batch_result = json.loads(raw)

    return {
        int(key): [kw.strip() for kw in kws if kw.strip()]
        for key, kws in batch_result.items()
    }


# ── 对外接口 ──────────────────────────────────────────────────────────────────

def extract_keywords(titles: list[str], verbose: bool = True) -> list[list[str]]:
    """
    对所有标题提取关键词，命中缓存的直接返回，未命中的批量调用 API。
    缓存以标题文本为 key，持久化到 KEYWORDS_CACHE_FILE。
    返回与 titles 等长的列表。
    """
    cache = _load_cache()

    # 找出需要调用 API 的标题（去重后再查缓存）
    uncached_titles: list[str] = []
    seen: set[str] = set()
    for title in titles:
        if title not in cache and title not in seen:
            uncached_titles.append(title)
            seen.add(title)

    cached_count = len(titles) - len(uncached_titles)
    if verbose:
        print(f"  缓存命中 {cached_count} 条，需调用 API {len(uncached_titles)} 条")

    # 批量提取未缓存的标题
    if uncached_titles:
        batch_size = KEYWORD_BATCH_SIZE
        total = len(uncached_titles)

        for start in range(0, total, batch_size):
            batch = uncached_titles[start: start + batch_size]
            if verbose:
                end = min(start + batch_size, total)
                print(f"  提取关键词: {start + 1}-{end} / {total}（新增）")

            try:
                batch_result = _extract_batch_from_api(batch)
                for local_idx, keywords in batch_result.items():
                    cache[batch[local_idx]] = keywords
                # 未在返回结果中的标题标记为空列表
                for local_idx in range(len(batch)):
                    if local_idx not in batch_result:
                        cache[batch[local_idx]] = []
            except Exception as e:
                print(f"  [警告] 批次 {start}-{start + batch_size} 提取失败: {e}")
                for title in batch:
                    cache.setdefault(title, [])

            # 每批完成后立即保存，防止中途中断丢失进度
            _save_cache(cache)

            if start + batch_size < total:
                time.sleep(0.5)

        if verbose:
            print(f"  缓存已更新并保存至 {KEYWORDS_CACHE_FILE}")

    return [cache.get(title, []) for title in titles]
