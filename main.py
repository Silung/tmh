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


def load_titles_and_video_ids(filepath: str, column: str) -> tuple[list[str], list[str]]:
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
    if "video_id" not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"列 'video_id' 不存在，可用列: {available}\n"
            f"请确保输入文件包含 video_id 列。"
        )

    valid_mask = df[column].notna()
    titles = df.loc[valid_mask, column].astype(str).str.strip().tolist()
    video_ids = df.loc[valid_mask, "video_id"].astype(str).str.strip().tolist()
    print(f"  共读取 {len(titles)} 条标题")
    return titles, video_ids


def load_video_timestamp_map(path: str) -> dict[str, float]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"时间数据文件不存在: {path}")

    matrix_df = pd.read_csv(path, on_bad_lines="skip", usecols=["video_id", "timestamp"])
    matrix_df["video_id"] = matrix_df["video_id"].astype(str).str.strip()
    matrix_df["timestamp"] = pd.to_numeric(matrix_df["timestamp"], errors="coerce")
    matrix_df = matrix_df.dropna(subset=["video_id", "timestamp"])
    # 同一个视频可能出现多次，使用中位数时间作为该视频代表时间。
    return matrix_df.groupby("video_id")["timestamp"].median().to_dict()


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
    titles, video_ids = load_titles_and_video_ids(INPUT_FILE, TITLE_COLUMN)
    timestamp_map = load_video_timestamp_map("KuaiRec/data/big_matrix.csv")

    # ── 步骤 2：提取关键词 ────────────────────────────────────────
    print("\n[2/5] 提取关键词（调用 OpenAI API）…")
    keywords_list = extract_keywords(titles, verbose=True)
    non_empty = sum(1 for kws in keywords_list if kws)
    print(f"  关键词提取完成，{non_empty}/{len(titles)} 条标题成功提取关键词")

    # ── 步骤 3：向量化 ────────────────────────────────────────────
    print("\n[3/5] 向量化关键词…")
    vectors = vectorize_keywords(keywords_list, verbose=True)
    print(f"  关键词向量矩阵形状: {vectors.shape}")

    # 把时间加入 embedding 额外 1 个维度（按 video_id 对应 timestamp）。
    raw_timestamps = np.array([timestamp_map.get(str(vid), np.nan) for vid in video_ids], dtype=np.float64)
    has_ts = ~np.isnan(raw_timestamps)
    if has_ts.any():
        ts_min = raw_timestamps[has_ts].min()
        ts_max = raw_timestamps[has_ts].max()
        if ts_max > ts_min:
            ts_scaled = np.where(has_ts, (raw_timestamps - ts_min) / (ts_max - ts_min), 0.0)
        else:
            ts_scaled = np.zeros_like(raw_timestamps, dtype=np.float64)
        vectors = np.concatenate([vectors, ts_scaled[:, None]], axis=1)
        missing_ts = (~has_ts).sum()
        print(
            f"  已加入时间维度: 1（匹配到 {has_ts.sum()} 条，缺失 {missing_ts} 条，缺失按 0 填充）"
        )
    else:
        vectors = np.concatenate([vectors, np.zeros((vectors.shape[0], 1), dtype=vectors.dtype)], axis=1)
        print("  [注意] 未匹配到任何 timestamp，时间维度全部按 0 填充")
    print(f"  最终向量矩阵形状: {vectors.shape}")

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
