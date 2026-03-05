import json
import os
import sys

import numpy as np
import pandas as pd

from config import INPUT_FILE, TITLE_COLUMN, OUTPUT_DIR, OUTPUT_FILE
from keyword_extractor import extract_keywords
from vectorizer import vectorize_keywords
from cluster import cluster_vectors, get_cluster_indices, get_cluster_scores
from event_summarizer import summarize_all_clusters


def load_titles(filepath: str, column: str) -> list[str]:
    print(f"[1/5] 读取数据: {filepath}")
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath, on_bad_lines="skip")
        df.columns = df.columns.str.strip()   # 去除列名首尾空格
    else:
        df = pd.read_excel(filepath)
    if column not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"列 '{column}' 不存在，可用列: {available}\n"
            f"请修改 config.py 中的 TITLE_COLUMN。"
        )
    titles = df[column].dropna().astype(str).str.strip().tolist()
    print(f"  共读取 {len(titles)} 条标题")
    return titles


def save_results(events: list[dict], output_dir: str, output_file: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, output_file)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    return path


def print_summary(events: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"共归纳出 {len(events)} 个事件（按标题数量降序）：")
    print("=" * 60)
    for i, ev in enumerate(events[:20], 1):
        summary = ev.get("event_summary", "")
        desc = ev.get("event_description", "")
        t = ev.get("time") or "—"
        loc = ev.get("location") or "—"
        count = ev["title_count"]
        print(f"\n{i:2d}. [{count} 条] 【{summary}】")
        print(f"    {desc}")
        print(f"    时间: {t}  地点: {loc}")
        kws = ev.get("summary_keywords", [])
        if kws:
            print(f"    关键词: {', '.join(kws[:8])}")
    if len(events) > 20:
        print(f"\n… 共 {len(events)} 个事件，完整结果见输出文件。")


def main():
    # ── 步骤 1：读取数据 ──────────────────────────────────────────
    titles = load_titles(INPUT_FILE, TITLE_COLUMN)

    # ── 步骤 2：提取关键词 ────────────────────────────────────────
    print("\n[2/5] 提取关键词（调用 OpenAI API）…")
    keywords_list = extract_keywords(titles, verbose=True)
    non_empty = sum(1 for kws in keywords_list if kws)
    print(f"  关键词提取完成，{non_empty}/{len(titles)} 条标题成功提取关键词")

    # ── 步骤 3：向量化 ────────────────────────────────────────────
    print("\n[3/5] 向量化关键词…")
    vectors = vectorize_keywords(keywords_list, verbose=True)
    print(f"  向量矩阵形状: {vectors.shape}")

    # 过滤全零向量（关键词提取失败的标题）
    nonzero_mask = np.linalg.norm(vectors, axis=1) > 0
    valid_indices = np.where(nonzero_mask)[0]
    n_zero = (~nonzero_mask).sum()
    if n_zero > 0:
        print(f"  [注意] {n_zero} 条标题关键词为空，将被跳过聚类")
    valid_vectors = vectors[valid_indices]
    valid_titles = [titles[i] for i in valid_indices]
    valid_keywords = [keywords_list[i] for i in valid_indices]

    # ── 步骤 4：聚类 ──────────────────────────────────────────────
    print("\n[4/5] 对向量进行聚类…")
    labels = cluster_vectors(valid_vectors, verbose=True)
    clusters = get_cluster_indices(labels)
    print(f"  得到 {len(clusters)} 个有效簇")
    print("  计算各簇密度/轮廓分数…")
    sil_scores = get_cluster_scores(valid_vectors, labels)

    # ── 步骤 5：归纳事件 ──────────────────────────────────────────
    print("\n[5/5] 调用大模型归纳事件…")
    events = summarize_all_clusters(
        clusters,
        titles=valid_titles,
        keywords_list=valid_keywords,
        silhouette_scores=sil_scores,
        verbose=True,
    )

    # ── 保存结果 ──────────────────────────────────────────────────
    output_path = save_results(events, OUTPUT_DIR, OUTPUT_FILE)
    print(f"\n结果已保存至: {output_path}")

    print_summary(events)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，退出。")
        sys.exit(0)
    except Exception as e:
        print(f"\n[错误] {e}", file=sys.stderr)
        raise
