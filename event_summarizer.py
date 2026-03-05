import json
import math
import time
from openai import OpenAI
from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    MAX_TITLES_PER_CLUSTER, MIN_TITLES_FOR_EVENT, MAX_EVENTS_TO_SUMMARIZE,
)


client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


SYSTEM_PROMPT = """你是一个专业的新闻事件分析师，擅长从短视频标题中识别真实事件。

任务：判断给定的一批短视频标题是否共同描述了一个具体的真实事件，如果是，用一句话概括该事件。

判断标准：
- 必须是具体事件（如"某地发生火灾"、"某人获奖"），而非泛泛的生活日常、情感表达、广告推广
- 概括句必须包含尽可能多的要素：人物（谁）、事件（做了什么/发生了什么）、时间（何时）、地点（何地）
- 若标题混杂、主题分散、无法归纳出一个明确事件，则标记为跳过

返回严格的 JSON 格式：
{
  "skip": true 或 false,
  "event_summary": "不超过20个字的极简标题（skip为true时为null）",
  "event_description": "一句话事件描述，包含尽可能多的人物/事件/时间/地点要素（skip为true时为null）",
  "time": "事件时间（可推断时填写，否则为null）",
  "location": "事件地点（可推断时填写，否则为null）",
  "key_actors": ["相关人物或角色"],
  "summary_keywords": ["代表性关键词，5-10个"]
}

只返回 JSON，不要有任何其他内容。"""


def summarize_cluster(titles: list[str], cluster_id: int) -> dict | None:
    """
    对一个簇的标题调用大模型归纳事件。
    返回事件信息字典；若模型判断无法归纳则返回 None。
    """
    sampled = titles[:MAX_TITLES_PER_CLUSTER]
    titles_text = "\n".join(f"- {t}" for t in sampled)
    user_prompt = (
        f"以下是 {len(sampled)} 条短视频标题（该组共 {len(titles)} 条）：\n\n"
        f"{titles_text}\n\n"
        """请判断这些标题是否描述了同一个具体事件，并按格式返回结果。例如：
十四届全国人大四次会议开幕会举行
建议春节9天假代表收到很多祝贺
建议尽量不要调休
中国2025成绩单
全皮层修护真相揭秘
霍启刚希望明年春晚有香港分会场
建议三孩每月补贴5000元至3岁
伊朗首次使用最快自杀式无人机
李宁品牌代言人白鹿
奶奶你是一块金子放错了地方
鼓励3岁以下婴幼儿父母弹性工作制
方寸之间看见中国
中国硅碳电池获GLOMO大奖
郭晓婷王天辰结婚共创来了
伊朗袭击以国防部大楼
"""
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    result = json.loads(raw)

    if result.get("skip", False):
        return None
    return result


def summarize_all_clusters(
    clusters: dict[int, list[int]],
    titles: list[str],
    keywords_list: list[list[str]] | None = None,
    silhouette_scores: dict[int, float] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    对所有簇归纳事件，跳过标题数不足或无法归纳的簇。
    按轮廓系数降序排列，最多处理 MAX_EVENTS_TO_SUMMARIZE 个簇。
    返回事件列表，最终按标题数量降序排列。
    """
    events = []
    skipped_small = 0
    skipped_vague = 0

    # 过滤标题不足的簇
    eligible = {
        cid: idxs for cid, idxs in clusters.items()
        if len(idxs) >= MIN_TITLES_FOR_EVENT
    }
    skipped_small = len(clusters) - len(eligible)

    # 按 密度分 × log_{MIN_TITLES_FOR_EVENT}(簇大小) 降序排序
    # 兼顾簇的紧密程度与规模，底数用 MIN_TITLES_FOR_EVENT 使最小合格簇的对数因子=1
    log_base = math.log(MIN_TITLES_FOR_EVENT)

    def _rank_score(cluster_id: int, indices: list) -> float:
        sil = silhouette_scores.get(cluster_id, 0.0) if silhouette_scores else 0.0
        size_factor = math.log(max(len(indices), 1)) / log_base
        return sil * size_factor

    sorted_clusters = sorted(
        eligible.items(),
        key=lambda x: _rank_score(x[0], x[1]),
        reverse=True,
    )

    if verbose:
        print(
            f"  共 {len(clusters)} 个簇，过滤后剩 {len(eligible)} 个"
            f"（跳过 {skipped_small} 个标题不足），目标归纳 {MAX_EVENTS_TO_SUMMARIZE} 个有效事件"
        )

    for rank, (cluster_id, indices) in enumerate(sorted_clusters, 1):
        if len(events) >= MAX_EVENTS_TO_SUMMARIZE:
            break

        n = len(indices)
        sil = silhouette_scores.get(cluster_id, 0.0) if silhouette_scores else 0.0

        cluster_titles = [titles[idx] for idx in indices]
        cluster_keywords: list[str] = []
        if keywords_list:
            seen: set[str] = set()
            for idx in indices:
                for kw in keywords_list[idx]:
                    if kw not in seen:
                        seen.add(kw)
                        cluster_keywords.append(kw)

        rank = _rank_score(cluster_id, indices)
        if verbose:
            print(
                f"  [{len(events) + 1}/{MAX_EVENTS_TO_SUMMARIZE}] "
                f"簇 {cluster_id}，{n} 条标题，"
                f"密度={sil:.4f}，综合分={rank:.4f}…"
            )

        try:
            print(cluster_titles)
            event_info = summarize_cluster(cluster_titles, cluster_id)
        except Exception as e:
            print(f"  [警告] 簇 {cluster_id} 归纳失败: {e}")
            event_info = None

        if event_info is None:
            skipped_vague += 1
            if verbose:
                print(f"    → 无法归纳出具体事件，已跳过，继续尝试下一个簇")
            time.sleep(0.3)
            continue

        events.append({
            "cluster_id": cluster_id,
            "title_count": n,
            "titles": cluster_titles,
            "all_keywords": cluster_keywords,
            "event_summary": event_info.get("event_summary", ""),
            "event_description": event_info.get("event_description", ""),
            "time": event_info.get("time"),
            "location": event_info.get("location"),
            "key_actors": event_info.get("key_actors", []),
            "summary_keywords": event_info.get("summary_keywords", []),
        })

        time.sleep(0.3)

    events.sort(key=lambda e: e["title_count"], reverse=True)

    if verbose:
        print(
            f"\n  共归纳出 {len(events)} 个有效事件"
            f"（跳过 {skipped_small} 个标题不足、{skipped_vague} 个主题分散的簇）"
        )
    return events
