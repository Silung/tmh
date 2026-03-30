"""
KuaiRec 成对标注测试集构建：关键词共现 + LLM 1–5 分 + SQLite 断点。

评测口径说明（阈值可据人工标定调整）：
  - score 1–5：LLM 输出；5＝确定同一热点事件/话题，1＝明确无关。
  - 正例阈值示例：score >= 4 视为「同属一事件」用于 pairwise F1。
  - 主集 coverage：仅「提取关键词交集非空」的候选对；若有 negative 分层，指标分开报。

用法:
  python build_pairwise_evalset.py sample      # 仅生成候选对写入 SQLite（status=pending）
  python build_pairwise_evalset.py run         # 对 pending/error 调 LLM（可反复执行，断点续跑）
  python build_pairwise_evalset.py export      # 从 SQLite 导出 pairs_eval.jsonl（及可选弱聚类）
  python build_pairwise_evalset.py topics       # 建簇 + LLM 概括；写 topics_eval.json 与 topics_list.json
  python build_pairwise_evalset.py topics-list   # 仅从 topics_eval.json 生成精简 topics_list.json
  python build_pairwise_evalset.py list-models  # 查看本机服务注册的 model id（与 --model 须一致）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from config import INPUT_FILE, KEYWORDS_CACHE_FILE, OUTPUT_DIR, TITLE_COLUMN


# ── 默认与服务（可用环境变量覆盖）────────────────────────────────────────────
# 默认直连本机 OpenAI 兼容服务；需要代理时设置环境变量 PAIRWISE_HTTP_PROXY 或传 --proxy

DEFAULT_PROXY = os.getenv("PAIRWISE_HTTP_PROXY") or ""
DEFAULT_BASE_URL = os.getenv("PAIRWISE_BASE_URL", "http://127.0.0.1:8012/v1")
# vLLM 默认用 HuggingFace 模型 id；若在启动脚本里设置了 --served-model-name，须与之完全一致
DEFAULT_MODEL = os.getenv("PAIRWISE_MODEL", "qwen")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("PAIRWISE_API_KEY") or "EMPTY"

SQLITE_PATH = os.path.join(OUTPUT_DIR, "pair_scores.sqlite")
JSONL_PATH = os.path.join(OUTPUT_DIR, "pair_scores.jsonl")
EVAL_JSONL_PATH = os.path.join(OUTPUT_DIR, "pairs_eval.jsonl")
CLUSTERS_JSON_PATH = os.path.join(OUTPUT_DIR, "pairs_eval_clusters.json")
TOPICS_JSON_PATH = os.path.join(OUTPUT_DIR, "topics_eval.json")
TOPICS_LIST_JSON_PATH = os.path.join(OUTPUT_DIR, "topics_list.json")

SYSTEM_PROMPT = """你是短视频内容评测助手。给定两条视频的 caption（标题文案），请判断是否描述**同一热点事件或同一可合并的话题**（同一新闻事件、同一活动、同一人物热点等算同一事件；仅词语碰巧相同但事实无关算不同）。

请只输出一个 JSON 对象，不要 markdown，不要其它文字。字段：
- score: 整数 1–5（5=确定同一事件/话题；4=很大概率同一；3=不确定或部分相关；2=多半不同；1=明确无关）
- same_event: 布尔值，是否与 score>=4 的判定一致即可
- brief_reason: 不超过40字的中文简短理由"""


USER_TEMPLATE = """caption A：
{caption_a}

caption B：
{caption_b}

输出 JSON：{{"score": <1-5>, "same_event": <true|false>, "brief_reason": "..."}}"""


TOPIC_SYSTEM_PROMPT = """你是新闻与热点事实归纳员。输入是多条短视频 caption，它们在标注阶段被粗略判为「可能同一事件」，但你需要严格复审。

【目标】只概括**一件具体的事实性事件**，读者应能回答「谁/什么 + 做了什么 / 发生了什么」（如同一条社会或文娱热点新闻的主谓宾），例如：「某某出席某活动」「某地发生某事」「某剧官宣定档」。

【必须 skip（skip=true）】出现任一则不要硬编概括：
- 标题明显不是同一桩事，只是领域相近、关键词碰巧相同、或单纯刷屏话术混在一组；
- 只能归纳成领域大类或风格标签，而没有共同的具体事实主体与行为（见下方禁止项）。

【禁止】topic_summary 与 topic_description 中**不得**使用或等同于以下内容作主概括：商业、娱乐、美食、搞笑、颜值、情感、三农、时尚、运动、摄影、音乐、生活、日常、好物、直播、上热门、作品推广等**领域/频道类名词**；也不得用「某某领域热点」这类空洞说法敷衍。

若无法写成具体事件句，一律 skip=true，topic_summary/topic_description 为 null，topic_keywords 为 []。

只输出一个 JSON 对象，不要 markdown，不要其它文字。字段：
- skip: 布尔值
- topic_summary: 字符串，8–24 字以内的**具体事件短句**（主谓清晰）；skip 为 true 时为 null
- topic_description: 字符串，一句话补充人物/行为/时间地点等可核实要素；skip 为 true 时为 null
- topic_keywords: 字符串数组；具体实体与事件词 3–8 个（人名、地名、剧名、活动名等）；skip 为 true 时为 []"""


TOPIC_USER_TEMPLATE = """本组共 {total} 条 caption（算法粗聚类），以下列出前 {shown} 条供你判断是否为**同一具体事件**：

{titles_block}

若明显不是同一事件，或只能概括成领域类别，请 skip=true；否则给出可当作新闻标题的事实型概括。输出 JSON。"""

_SENTINEL = object()

# 话题概括时，送入模型的 titles 拼接块最大字符数（与 CLI --max-prompt-chars 默认一致）
DEFAULT_MAX_PROMPT_TITLES_CHARS = 5000


def titles_sample_for_llm(
    titles_full: list[str],
    max_count: int,
    max_total_chars: int,
) -> list[str]:
    """
    按顺序选取 caption，拼接为「- 行1\\n- 行2」形式时总长度不超过 max_total_chars。
    max_count<=0 表示条数不设上限，仅受字符数约束。
    """
    if not titles_full or max_total_chars <= 0:
        return []
    out: list[str] = []
    for t in titles_full:
        if max_count > 0 and len(out) >= max_count:
            break
        candidate = out + [t]
        block = "\n".join(f"- {x}" for x in candidate)
        if len(block) <= max_total_chars:
            out.append(t)
            continue
        if not out:
            head = max_total_chars - 2
            if head < 1:
                break
            suffix = "…" if len(t) > head else ""
            out.append(t[:head] + suffix)
        break
    return out


@dataclass(frozen=True)
class Item:
    video_id: int
    caption: str
    keywords: frozenset[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pair_key(a: int, b: int) -> str:
    if a == b:
        raise ValueError("pair_key: identical ids")
    x, y = (a, b) if a < b else (b, a)
    return f"{x}_{y}"


def load_keywords_cache(path: str) -> dict[str, list[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"关键词缓存不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("keywords_cache 根类型应为 JSON 对象")
    out: dict[str, list[str]] = {}
    for k, v in raw.items():
        if not isinstance(v, list):
            raise ValueError(f"缓存键 {k!r} 的值必须是列表")
        out[str(k)] = [str(x).strip() for x in v if str(x).strip()]
    return out


def load_items(csv_path: str, cache: dict[str, list[str]]) -> list[Item]:
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    if TITLE_COLUMN not in df.columns or "video_id" not in df.columns:
        raise ValueError(f"CSV 需含 {TITLE_COLUMN} 与 video_id")

    cap = df[TITLE_COLUMN]
    valid = cap.notna()
    df = df.loc[valid].copy()
    df["_cap"] = df[TITLE_COLUMN].astype(str).str.strip()
    df = df[df["_cap"] != ""]
    df["_vid"] = pd.to_numeric(df["video_id"], errors="coerce")
    df = df.dropna(subset=["_vid"])
    df["_vid"] = df["_vid"].astype("int64")

    items: list[Item] = []
    for _, row in df.iterrows():
        caption = row["_cap"]
        kws = cache.get(caption)
        if not kws:
            continue
        vid = int(row["_vid"])
        items.append(Item(video_id=vid, caption=caption, keywords=frozenset(kws)))

    # 同一 video_id 多条时保留首条，避免主键冲突
    seen: set[int] = set()
    uniq: list[Item] = []
    for it in items:
        if it.video_id in seen:
            continue
        seen.add(it.video_id)
        uniq.append(it)
    return uniq


def load_vid_to_caption(csv_path: str) -> dict[int, str]:
    """全表非空 caption → video_id（同 id 保留首条）。"""
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    if TITLE_COLUMN not in df.columns or "video_id" not in df.columns:
        raise ValueError(f"CSV 需含 {TITLE_COLUMN} 与 video_id")
    cap = df[TITLE_COLUMN]
    valid = cap.notna()
    df = df.loc[valid].copy()
    df["_cap"] = df[TITLE_COLUMN].astype(str).str.strip()
    df = df[df["_cap"] != ""]
    df["_vid"] = pd.to_numeric(df["video_id"], errors="coerce")
    df = df.dropna(subset=["_vid"])
    df["_vid"] = df["_vid"].astype("int64")
    out: dict[int, str] = {}
    for _, row in df.iterrows():
        vid = int(row["_vid"])
        if vid not in out:
            out[vid] = row["_cap"]
    return out


def clusters_from_ok_pair_rows(
    rows: list[tuple[Any, ...]],
    score_min: int,
    min_cluster_size: int,
) -> list[list[int]]:
    """
    rows 与 export 查询一致：pair_key,i,j,...,score,...
    按 score>=score_min 建无向图，返回连通分量（仅保留成员数>=min_cluster_size），按规模降序。
    """
    adj: dict[int, set[int]] = defaultdict(set)
    nodes: set[int] = set()
    for r in rows:
        _, i, j, _, _, _, _, sc, _ = r
        if sc is not None and sc >= score_min:
            adj[i].add(j)
            adj[j].add(i)
            nodes.add(i)
            nodes.add(j)
    seen: set[int] = set()
    clusters: list[list[int]] = []
    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        comp: list[int] = []
        seen.add(n)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(comp) >= min_cluster_size:
            clusters.append(sorted(comp))
    clusters.sort(key=lambda c: -len(c))
    return clusters


def build_inverted_index(items: list[Item]) -> dict[str, list[int]]:
    inv: dict[str, list[int]] = defaultdict(list)
    for it in items:
        for w in it.keywords:
            inv[w].append(it.video_id)
    return dict(inv)


def sample_pairs_from_keywords(
    inv_index: dict[str, list[int]],
    cap_per_keyword: int,
    max_total_pairs: int,
    rng: random.Random,
) -> set[tuple[int, int]]:
    """每个关键词在其 posting 列表内随机抽对，全局去重，总数封顶。"""
    pairs: set[tuple[int, int]] = set()
    words = list(inv_index.keys())
    rng.shuffle(words)
    for w in words:
        ids = inv_index[w]
        if len(ids) < 2:
            continue
        added = 0
        local_cap = cap_per_keyword
        # 去重后的 id 列表（同一词下同一视频只出现一次）
        id_list = list(dict.fromkeys(ids))
        if len(id_list) < 2:
            continue
        attempts = 0
        max_attempts = local_cap * 30
        while added < local_cap and len(pairs) < max_total_pairs and attempts < max_attempts:
            attempts += 1
            a, b = rng.sample(id_list, 2)
            x, y = (a, b) if a < b else (b, a)
            if (x, y) in pairs:
                continue
            pairs.add((x, y))
            added += 1
        if len(pairs) >= max_total_pairs:
            break
    return pairs


def sample_disjoint_pairs(
    id_to_keywords: dict[int, frozenset[str]],
    n_target: int,
    exclude: set[tuple[int, int]],
    rng: random.Random,
) -> set[tuple[int, int]]:
    """关键词集合交集为空的无向对（与主集分层分开记指标）。"""
    ids = list(id_to_keywords.keys())
    if len(ids) < 2:
        return set()
    out: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(n_target * 80, 1000)
    while len(out) < n_target and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(ids, 2)
        if id_to_keywords[a] & id_to_keywords[b]:
            continue
        x, y = (a, b) if a < b else (b, a)
        if (x, y) in exclude:
            continue
        out.add((x, y))
    return out


# ── SQLite ───────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS pair_scores (
  pair_key TEXT PRIMARY KEY,
  i INTEGER NOT NULL,
  j INTEGER NOT NULL,
  caption_i TEXT NOT NULL,
  caption_j TEXT NOT NULL,
  shared_keywords TEXT,
  layer TEXT NOT NULL DEFAULT 'keyword_overlap',
  score INTEGER,
  same_event INTEGER,
  raw_response TEXT,
  status TEXT NOT NULL,
  error_message TEXT,
  updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_pair_scores_status ON pair_scores(status);
"""


def db_connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(DDL)
    # 旧库补列
    cur = conn.execute("PRAGMA table_info(pair_scores)")
    cols = {row[1] for row in cur.fetchall()}
    if "layer" not in cols:
        conn.execute(
            "ALTER TABLE pair_scores ADD COLUMN layer TEXT NOT NULL DEFAULT 'keyword_overlap'"
        )
        conn.commit()
    conn.commit()
    return conn


def db_upsert_pending(
    conn: sqlite3.Connection,
    pairs: list[tuple[int, int, str, str, str, str]],
) -> None:
    conn.executemany(
        """
        INSERT OR IGNORE INTO pair_scores
        (pair_key, i, j, caption_i, caption_j, shared_keywords, layer, status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """,
        [
            (
                pair_key(i, j),
                i,
                j,
                ci,
                cj,
                sk,
                layer,
                _utc_now_iso(),
            )
            for (i, j, ci, cj, sk, layer) in pairs
        ],
    )
    conn.commit()


def db_update_result(
    conn: sqlite3.Connection,
    pk: str,
    score: int | None,
    same_event: int | None,
    raw: str,
    status: str,
    err: str | None,
) -> None:
    conn.execute(
        """
        UPDATE pair_scores SET
          score = ?, same_event = ?, raw_response = ?, status = ?,
          error_message = ?, updated_at = ?
        WHERE pair_key = ?
        """,
        (
            score,
            same_event,
            raw,
            status,
            err,
            _utc_now_iso(),
            pk,
        ),
    )


def append_jsonl(path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def shared_kw_str(ka: frozenset[str], kb: frozenset[str]) -> str:
    inter = sorted(ka & kb)
    return ",".join(inter)


# ── LLM ──────────────────────────────────────────────────────────────────────

_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)

# API / 网络类错误：可重试。解析与业务校验错误：可重试（换次模型输出）。
_RETRY_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
    httpx.RequestError,
    httpx.TimeoutException,
    json.JSONDecodeError,
    KeyError,
    TypeError,
    ValueError,
)


def parse_llm_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            raise
        data = json.loads(m.group())
    if not isinstance(data, dict):
        raise TypeError("模型输出必须是 JSON 对象")
    if isinstance(data.get("same_event"), str):
        data["same_event"] = data["same_event"].lower() in ("true", "1", "yes")
    return data


def _coerce_score(v: Any) -> int:
    """1–5 整数；拒绝 bool（Python 中 bool 是 int 子类，会误把 true 当成 1）。"""
    if isinstance(v, bool):
        raise ValueError("score 不得为布尔值")
    if isinstance(v, int):
        if 1 <= v <= 5:
            return v
        raise ValueError(f"score 需在 1–5: {v!r}")
    if isinstance(v, float) and v == int(v):
        return _coerce_score(int(v))
    if isinstance(v, str) and v.strip().isdigit():
        return _coerce_score(int(v.strip()))
    raise ValueError(f"无法解析 score: {v!r}")


async def score_one_pair(
    client: AsyncOpenAI,
    model: str,
    caption_a: str,
    caption_b: str,
    max_retries: int = 5,
) -> tuple[dict[str, Any] | None, str, str | None]:
    """返回 (parsed_dict, raw_text, error_msg)。并发由外层 worker 数量限制，此处不再套信号量。"""
    user = USER_TEMPLATE.format(caption_a=caption_a, caption_b=caption_b)
    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = parse_llm_json(raw)
            score = _coerce_score(data["score"])
            se = data.get("same_event")
            if not isinstance(se, bool):
                se = score >= 4
            data["_same_event_coerced"] = se
            return data, raw, None
        except _RETRY_EXCEPTIONS as e:
            last_err = str(e)
            await asyncio.sleep(min(2**attempt, 30))
    return None, "", last_err


async def llm_summarize_topic(
    client: AsyncOpenAI,
    model: str,
    titles_sample: list[str],
    total_members: int,
    max_retries: int = 5,
) -> tuple[dict[str, Any] | None, str, str | None]:
    """对一簇 caption 做话题概括。返回 (解析字典, 原始文本, 错误信息)。"""
    titles_block = "\n".join(f"- {t}" for t in titles_sample)
    user = TOPIC_USER_TEMPLATE.format(
        total=total_members,
        shown=len(titles_sample),
        titles_block=titles_block,
    )
    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": TOPIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = parse_llm_json(raw)
            if not isinstance(data.get("skip"), bool):
                sk = data.get("skip")
                data["skip"] = bool(sk) if sk is not None else False
            kws = data.get("topic_keywords")
            if kws is not None and not isinstance(kws, list):
                raise ValueError("topic_keywords 须为数组")
            return data, raw, None
        except _RETRY_EXCEPTIONS as e:
            last_err = str(e)
            await asyncio.sleep(min(2**attempt, 30))
    return None, "", last_err


async def run_topics_llm(
    clusters: list[list[int]],
    vid2cap: dict[int, str],
    base_url: str,
    api_key: str,
    model: str,
    proxy: str | None,
    concurrency: int,
    max_titles_sample: int,
    max_prompt_titles_chars: int,
) -> list[dict[str, Any]]:
    total = len(clusters)
    print(f"待 LLM 概括的话题簇: {total}，并发={concurrency}", flush=True)
    timeout = httpx.Timeout(120.0, connect=30.0)
    http_client = httpx.AsyncClient(timeout=timeout, proxy=proxy) if proxy else httpx.AsyncClient(timeout=timeout)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=_openai_v1_base(base_url),
        http_client=http_client,
    )

    queue: asyncio.Queue = asyncio.Queue()
    for i, c in enumerate(clusters):
        queue.put_nowait((i, c))
    for _ in range(concurrency):
        queue.put_nowait(_SENTINEL)

    results: list[dict[str, Any] | None] = [None] * total
    lock = asyncio.Lock()
    done = 0

    async def worker() -> None:
        nonlocal done
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            topic_idx, members = item
            titles_full = [vid2cap[v] for v in members if v in vid2cap]
            count_cap = max_titles_sample if max_titles_sample > 0 else 0
            source_for_sample = titles_full
            if not source_for_sample:
                source_for_sample = [
                    f"(无 caption，video_id={v})" for v in members
                ]
            sample = titles_sample_for_llm(
                source_for_sample,
                count_cap,
                max_prompt_titles_chars,
            )
            data, raw, err = await llm_summarize_topic(
                client, model, sample, len(members)
            )
            skip = False
            ts: str | None = None
            td: str | None = None
            tk: list[str] = []
            if data is None:
                llm_error = err or "request_failed"
            else:
                skip = bool(data.get("skip"))
                llm_error = err
                if not skip:
                    ts = data.get("topic_summary")
                    if isinstance(ts, str):
                        ts = ts.strip() or None
                    td = data.get("topic_description")
                    if isinstance(td, str):
                        td = td.strip() or None
                    tk = [
                        str(x).strip()
                        for x in (data.get("topic_keywords") or [])
                        if str(x).strip()
                    ]
            rec: dict[str, Any] = {
                "topic_index": topic_idx,
                "member_count": len(members),
                "skipped_by_llm": skip,
                "topic_summary": ts,
                "topic_description": td,
                "topic_keywords": tk,
                "video_ids": members,
                "titles": titles_full
                if titles_full
                else [vid2cap.get(v, "") for v in members],
                "llm_error": llm_error,
            }
            if data is not None:
                rec["raw_llm_response"] = raw
            async with lock:
                results[topic_idx] = rec
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"话题概括进度 {done}/{total}", flush=True)

    try:
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers)
    finally:
        await http_client.aclose()

    return [r for r in results if r is not None]


async def run_labeling(
    conn: sqlite3.Connection,
    jsonl_path: str,
    sqlite_path: str,
    base_url: str,
    api_key: str,
    model: str,
    proxy: str | None,
    concurrency: int,
    commit_every: int,
    progress_every: int,
) -> None:
    cur = conn.execute(
        """
        SELECT pair_key, caption_i, caption_j FROM pair_scores
        WHERE status IN ('pending', 'error')
        ORDER BY pair_key
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("没有 pending/error 记录，跳过 LLM。")
        return

    total = len(rows)
    pe = max(1, progress_every)
    print(f"待标注 {total} 对，并发 worker={concurrency}（同时最多 {concurrency} 条请求）…", flush=True)

    timeout = httpx.Timeout(120.0, connect=30.0)
    http_client = httpx.AsyncClient(timeout=timeout, proxy=proxy) if proxy else httpx.AsyncClient(timeout=timeout)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=_openai_v1_base(base_url),
        http_client=http_client,
    )

    queue: asyncio.Queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    for _ in range(concurrency):
        queue.put_nowait(_SENTINEL)

    count_lock = asyncio.Lock()
    n_ok = 0
    n_err = 0
    since_commit = 0
    done_count = 0

    async def one(pk: str, ca: str, cb: str) -> None:
        nonlocal n_ok, n_err, since_commit, done_count
        data, raw, err = await score_one_pair(client, model, ca, cb)
        out_score: int | None = None
        out_se = False
        async with count_lock:
            if data is not None:
                n_ok += 1
                out_score = int(data["score"])
                out_se = bool(
                    data.get("_same_event_coerced") or data.get("same_event")
                )
                db_update_result(
                    conn, pk, out_score, 1 if out_se else 0, raw or "", "ok", None
                )
            else:
                n_err += 1
                db_update_result(conn, pk, None, None, raw or "", "error", err)
            since_commit += 1
            if since_commit >= commit_every:
                conn.commit()
                since_commit = 0
            done_count += 1
            d = done_count
            if d % pe == 0 or d == total:
                print(
                    f"进度 {d}/{total}  ok={n_ok}  err={n_err}",
                    flush=True,
                )
        if data is not None and out_score is not None:
            append_jsonl(
                jsonl_path,
                {
                    "pair_key": pk,
                    "score": out_score,
                    "same_event": out_se,
                    "raw_response": raw,
                    "ts": _utc_now_iso(),
                },
            )

    async def worker() -> None:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            pk, ca, cb = item
            await one(pk, ca, cb)

    try:
        worker_tasks = [
            asyncio.create_task(worker()) for _ in range(concurrency)
        ]
        await asyncio.gather(*worker_tasks)
    finally:
        conn.commit()
        await http_client.aclose()

    print(f"完成: ok={n_ok}, error={n_err}，数据库 {sqlite_path}", flush=True)


def cmd_sample(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    cache = load_keywords_cache(args.keywords_cache)
    items = load_items(args.csv, cache)
    print(f"有效样本（非空 caption + 缓存有关键词）: {len(items)}")

    inv = build_inverted_index(items)
    kw_pairs = sample_pairs_from_keywords(
        inv, args.cap_per_keyword, args.max_total_pairs, rng
    )
    print(f"关键词共现候选对（去重）: {len(kw_pairs)}")

    by_id = {it.video_id: it for it in items}
    id2k = {it.video_id: it.keywords for it in items}

    neg_pairs: set[tuple[int, int]] = set()
    if args.negative_fraction > 0:
        n_neg = min(
            max(0, int(len(kw_pairs) * args.negative_fraction)),
            args.max_negative_pairs,
        )
        neg_pairs = sample_disjoint_pairs(id2k, n_neg, kw_pairs, rng)
        print(f"无关键词交集分层（负例/分层）: {len(neg_pairs)}")

    all_pairs = kw_pairs | neg_pairs
    rows_to_insert: list[tuple[int, int, str, str, str, str]] = []
    for i, j in sorted(all_pairs):
        a, b = by_id[i], by_id[j]
        sk = shared_kw_str(a.keywords, b.keywords)
        layer = "keyword_overlap" if (i, j) in kw_pairs else "disjoint_keywords"
        rows_to_insert.append((i, j, a.caption, b.caption, sk, layer))

    conn = db_connect(args.sqlite)
    n_before = conn.execute("SELECT COUNT(*) FROM pair_scores").fetchone()[0]
    db_upsert_pending(conn, rows_to_insert)
    n_after = conn.execute("SELECT COUNT(*) FROM pair_scores").fetchone()[0]
    print(
        f"候选对 {len(rows_to_insert)}；数据库 {args.sqlite} 行数 {n_before} -> {n_after}（新增约 {n_after - n_before}）"
    )
    conn.close()


def cmd_run(args: argparse.Namespace) -> None:
    conn = db_connect(args.sqlite)
    proxy = (args.proxy or "").strip() or None
    try:
        asyncio.run(
            run_labeling(
                conn,
                args.jsonl,
                args.sqlite,
                args.base_url,
                args.api_key,
                args.model,
                proxy,
                args.concurrency,
                args.commit_every,
                args.progress_every,
            )
        )
    except KeyboardInterrupt:
        conn.commit()
        print(
            "\n已中断：已执行 commit，请再次运行 run 续标未完成行。",
            flush=True,
        )
    finally:
        conn.close()


def cmd_export(args: argparse.Namespace) -> None:
    conn = db_connect(args.sqlite)
    cur = conn.execute(
        """
        SELECT pair_key, i, j, caption_i, caption_j, shared_keywords, layer, score, same_event
        FROM pair_scores WHERE status = 'ok' ORDER BY pair_key
        """
    )
    rows = cur.fetchall()
    os.makedirs(os.path.dirname(args.eval_jsonl) or ".", exist_ok=True)
    with open(args.eval_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            pk, i, j, ci, cj, sk, layer, sc, se = r
            rec = {
                "pair_key": pk,
                "item_i": i,
                "item_j": j,
                "caption_i": ci,
                "caption_j": cj,
                "shared_keywords": sk or "",
                "layer": layer or "keyword_overlap",
                "score": sc,
                "same_event": bool(se),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"导出 {len(rows)} 条至 {args.eval_jsonl}")

    if args.export_clusters and rows:
        thr = args.cluster_score_min
        clusters = clusters_from_ok_pair_rows(rows, thr, min_cluster_size=2)
        meta = {
            "generated_at": _utc_now_iso(),
            "cluster_score_min": thr,
            "note": "仅在导出子图的 ok 边上、score>=阈值 的连通分量；弱聚类标签，非严格金标准",
            "components": [{"members": c} for c in clusters],
        }
        cpath = args.clusters_json
        with open(cpath, "w", encoding="utf-8") as cf:
            json.dump(meta, cf, ensure_ascii=False, indent=2)
        print(f"弱聚类分量（>={thr} 分）: {len(clusters)} 写入 {cpath}")

    conn.close()


def compact_topics_for_list(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """去掉 raw 字段，保留 topic 维度和全量 video_ids/titles。"""
    out: list[dict[str, Any]] = []
    for t in topics:
        out.append(
            {
                "topic_index": t.get("topic_index"),
                "member_count": t.get("member_count"),
                "skipped_by_llm": t.get("skipped_by_llm"),
                "topic_summary": t.get("topic_summary"),
                "topic_description": t.get("topic_description"),
                "topic_keywords": t.get("topic_keywords") or [],
                "video_ids": t.get("video_ids") or [],
                "titles": t.get("titles") or [],
            }
        )
    return out


def write_topics_list_file(
    list_out: str,
    topics: list[dict[str, Any]],
    extra_meta: dict[str, Any],
    list_keep_skipped: bool = False,
) -> None:
    compact = compact_topics_for_list(topics)
    full_n = len(compact)
    if not list_keep_skipped:
        compact = [
            x
            for x in compact
            if (not x.get("skipped_by_llm"))
            and (x.get("topic_summary") and str(x.get("topic_summary")).strip())
        ]
    summaries = [x.get("topic_summary") for x in compact]
    payload = {
        "generated_at": _utc_now_iso(),
        "topic_count": len(compact),
        "topic_summaries": summaries,
        "topics": compact,
        "list_note": (
            "仅含模型判定为「具体事件」的 topic；被 skip 或空概括已剔除。"
            if not list_keep_skipped
            else "含被 skip 的簇，便于核对。"
        ),
        "topics_before_list_filter": full_n,
        **extra_meta,
    }
    os.makedirs(os.path.dirname(list_out) or ".", exist_ok=True)
    with open(list_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def cmd_topics_list(args: argparse.Namespace) -> None:
    """从已生成的 topics_eval.json 提取精简列表。"""
    path = args.topics_in
    if not os.path.isfile(path):
        print(f"文件不存在: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    topics = data.get("topics")
    if not topics:
        print(f"{path} 中无 topics 数组。")
        return
    write_topics_list_file(
        args.list_out,
        topics,
        {"source_topics_eval": path},
        list_keep_skipped=args.topics_list_keep_skipped,
    )
    print(
        f"已写入 {args.list_out}，共 {len(topics)} 条话题；"
        f"字段 topic_summaries 为短标题扁平列表。",
        flush=True,
    )


def cmd_topics(args: argparse.Namespace) -> None:
    """由 pair_scores 高分边建簇，LLM 概括 topic，输出 topic→titles。"""
    conn = db_connect(args.sqlite)
    cur = conn.execute(
        """
        SELECT pair_key, i, j, caption_i, caption_j, shared_keywords, layer, score, same_event
        FROM pair_scores WHERE status = 'ok' ORDER BY pair_key
        """
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        print("数据库中无 status=ok 的成对标注，请先完成 run。")
        return

    clusters = clusters_from_ok_pair_rows(
        rows, args.cluster_score_min, args.min_cluster_size
    )
    if not clusters:
        print("无满足条件的连通簇，可尝试降低 --cluster-score-min 或 --min-cluster-size。")
        return

    if args.max_topics > 0:
        clusters = clusters[: args.max_topics]

    vid2cap = load_vid_to_caption(args.csv)
    proxy = (args.proxy or "").strip() or None

    try:
        topics = asyncio.run(
            run_topics_llm(
                clusters,
                vid2cap,
                args.base_url,
                args.api_key,
                args.model,
                proxy,
                args.summarize_concurrency,
                args.max_titles_sample,
                args.max_prompt_chars,
            )
        )
    except KeyboardInterrupt:
        print("\n已中断，未写入 topics 文件。", flush=True)
        return

    payload = {
        "generated_at": _utc_now_iso(),
        "source_sqlite": args.sqlite,
        "csv": args.csv,
        "cluster_score_min": args.cluster_score_min,
        "min_cluster_size": args.min_cluster_size,
        "max_prompt_titles_chars": args.max_prompt_chars,
        "topic_count": len(topics),
        "topics": topics,
    }
    os.makedirs(os.path.dirname(args.topics_out) or ".", exist_ok=True)
    with open(args.topics_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"已写入 {args.topics_out}，共 {len(topics)} 个话题（含 video_ids 与 titles）。")

    if args.no_topics_list:
        return
    write_topics_list_file(
        args.topics_list_out,
        topics,
        {
            "source_sqlite": args.sqlite,
            "csv": args.csv,
            "cluster_score_min": args.cluster_score_min,
            "min_cluster_size": args.min_cluster_size,
            "full_detail_path": args.topics_out,
        },
        list_keep_skipped=args.topics_list_keep_skipped,
    )
    print(
        f"已写入话题列表 {args.topics_list_out}（topic_summaries + 每条含 video_ids/titles）。",
        flush=True,
    )


def _openai_v1_base(base_url: str) -> str:
    bu = base_url.rstrip("/")
    if not bu.endswith("/v1"):
        bu = bu + "/v1"
    return bu


def cmd_list_models(args: argparse.Namespace) -> None:
    """列出服务端注册的 model id，需与 run 时 --model 一致。"""
    bu = _openai_v1_base(args.base_url)
    url = f"{bu}/models"
    proxy = (getattr(args, "proxy", None) or "").strip() or None
    with httpx.Client(timeout=60.0, proxy=proxy) as client:
        r = client.get(url)
    if r.status_code != 200:
        print(f"HTTP {r.status_code} {url}\n{r.text[:500]}")
        raise SystemExit(1)
    data = r.json()
    items = data.get("data") or []
    if not items:
        print(f"{url} 未返回 data 列表: {data!r}")
        raise SystemExit(1)
    print(f"{url}\n")
    for m in items:
        print(m.get("id", m))


def main() -> None:
    p = argparse.ArgumentParser(description="KuaiRec LLM 成对打分测试集")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--csv", default=INPUT_FILE)
        sp.add_argument("--keywords-cache", default=KEYWORDS_CACHE_FILE)
        sp.add_argument("--sqlite", default=SQLITE_PATH)

    sp0 = sub.add_parser("sample", help="生成候选对并写入 SQLite（pending）")
    add_common(sp0)
    sp0.add_argument("--cap-per-keyword", type=int, default=500)
    sp0.add_argument("--max-total-pairs", type=int, default=50_000)
    sp0.add_argument("--negative-fraction", type=float, default=0.05)
    sp0.add_argument("--max-negative-pairs", type=int, default=5_000)
    sp0.add_argument("--seed", type=int, default=42)
    sp0.set_defaults(func=cmd_sample)

    sp1 = sub.add_parser("run", help="LLM 标注（断点续跑）")
    add_common(sp1)
    sp1.add_argument("--jsonl", default=JSONL_PATH)
    sp1.add_argument(
        "--proxy",
        default=DEFAULT_PROXY,
        help="HTTP 代理，如 http://127.0.0.1:7892；默认空（直连）",
    )
    sp1.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sp1.add_argument("--api-key", default=DEFAULT_API_KEY)
    sp1.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="须与 GET .../v1/models 里某个 id 完全一致；不确定时先运行 list-models",
    )
    sp1.add_argument("--concurrency", type=int, default=16)
    sp1.add_argument("--commit-every", type=int, default=100)
    sp1.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="每完成多少条打印一次进度（含首次达总数时）",
    )
    sp1.set_defaults(func=cmd_run)

    sp_lm = sub.add_parser(
        "list-models",
        help="列出本机 OpenAI 兼容服务上的 model id（排查 404）",
    )
    sp_lm.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sp_lm.add_argument(
        "--proxy",
        default=DEFAULT_PROXY,
        help="HTTP 代理；默认空",
    )
    sp_lm.set_defaults(func=cmd_list_models)

    sp2 = sub.add_parser("export", help="导出 pairs_eval.jsonl")
    add_common(sp2)
    sp2.add_argument("--eval-jsonl", default=EVAL_JSONL_PATH)
    sp2.add_argument("--clusters-json", default=CLUSTERS_JSON_PATH)
    sp2.add_argument("--export-clusters", action="store_true", default=True)
    sp2.add_argument("--no-clusters", action="store_true")
    sp2.add_argument("--cluster-score-min", type=int, default=4)
    sp2.set_defaults(func=cmd_export)

    sp3 = sub.add_parser(
        "topics",
        help="建簇 + LLM 概括；输出 topics_eval.json（全量）与 topics_list.json（话题列表）",
    )
    sp3.add_argument("--csv", default=INPUT_FILE)
    sp3.add_argument("--sqlite", default=SQLITE_PATH)
    sp3.add_argument("--topics-out", default=TOPICS_JSON_PATH)
    sp3.add_argument(
        "--topics-list-out",
        default=TOPICS_LIST_JSON_PATH,
        help="精简 topic 列表 JSON 路径（默认 output/topics_list.json）",
    )
    sp3.add_argument(
        "--no-topics-list",
        action="store_true",
        help="不生成精简 topics_list.json",
    )
    sp3.add_argument(
        "--topics-list-keep-skipped",
        action="store_true",
        help="topics_list.json 中保留被模型 skip 的簇（默认只保留具体事件）",
    )
    sp3.add_argument(
        "--cluster-score-min",
        type=int,
        default=4,
        help="与 export 一致：边上 score>=此值才参与建簇",
    )
    sp3.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        dest="min_cluster_size",
        help="成员数少于此的连通分量丢弃（不做概括）",
    )
    sp3.add_argument(
        "--max-topics",
        type=int,
        default=0,
        help="最多概括前若干个簇（按簇大小降序），0 表示不限制",
    )
    sp3.add_argument(
        "--max-titles-sample",
        type=int,
        default=15,
        help="送给 LLM 的 caption 最多条数；0 表示仅按 --max-prompt-chars 截断",
    )
    sp3.add_argument(
        "--max-prompt-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_TITLES_CHARS,
        help="送入模型的 titles 块（按「- 行\\n- 行」拼接）总字符上限，默认 5000",
    )
    sp3.add_argument(
        "--summarize-concurrency",
        type=int,
        default=8,
        help="话题概括 LLM 并发数",
    )
    sp3.add_argument(
        "--proxy",
        default=DEFAULT_PROXY,
        help="HTTP 代理；默认空（直连）",
    )
    sp3.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sp3.add_argument("--api-key", default=DEFAULT_API_KEY)
    sp3.add_argument("--model", default=DEFAULT_MODEL)
    sp3.set_defaults(func=cmd_topics)

    sp4 = sub.add_parser(
        "topics-list",
        help="从 topics_eval.json 生成精简 topics_list.json（不调 LLM）",
    )
    sp4.add_argument("--topics-in", default=TOPICS_JSON_PATH)
    sp4.add_argument("--list-out", default=TOPICS_LIST_JSON_PATH)
    sp4.add_argument(
        "--topics-list-keep-skipped",
        action="store_true",
        help="列表中保留 skip 的簇（默认只输出具体事件）",
    )
    sp4.set_defaults(func=cmd_topics_list)

    args = p.parse_args()
    if args.cmd == "export" and getattr(args, "no_clusters", False):
        args.export_clusters = False
    elif args.cmd == "export":
        args.export_clusters = not getattr(args, "no_clusters", False)

    args.func(args)


if __name__ == "__main__":
    main()
