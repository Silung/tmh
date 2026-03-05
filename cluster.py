import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from config import (
    CLUSTERING_METHOD, N_CLUSTERS, MIN_CLUSTER_SIZE,
    DBSCAN_EPS, MIN_K, MAX_K, STEP_K,
)


def _best_k(vectors: np.ndarray) -> int:
    """使用轮廓系数（Silhouette Score）自动选取最优 K 值"""
    best_k = MIN_K
    best_score = -1.0

    upper = min(MAX_K, len(vectors) - 1)
    for k in range(MIN_K, upper + 1, STEP_K):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(vectors, labels, sample_size=min(2000, len(vectors)))
        if score > best_score:
            best_score = score
            best_k = k
        print(f"    k={k:3d}  轮廓系数={score:.4f}")

    print(f"  自动选取 k={best_k}（轮廓系数={best_score:.4f}）")
    return best_k


def _auto_eps(normed: np.ndarray, min_samples: int, verbose: bool = True) -> float:
    """
    用 k 距离图的肘部法则自动确定 DBSCAN 的 eps。
    k = min_samples，对每个点取第 k 近邻的余弦距离，排序后找曲率最大点。
    """
    k = min_samples
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nbrs.fit(normed)
    dists, _ = nbrs.kneighbors(normed)
    k_dists = np.sort(dists[:, k - 1])          # 每个点到第 k 近邻的距离，升序

    n = len(k_dists)
    # 将曲线归一化到 [0,1]×[0,1]，计算每点到首尾连线的距离，取最远点为肘部
    x = np.linspace(0, 1, n)
    y = (k_dists - k_dists[0]) / (k_dists[-1] - k_dists[0] + 1e-12)
    # 首尾连线方向向量
    dx, dy = 1.0, 1.0
    length = np.sqrt(dx ** 2 + dy ** 2)
    dist_to_line = np.abs(dy * x - dx * y) / length
    elbow_idx = int(np.argmax(dist_to_line))
    eps = float(k_dists[elbow_idx])

    if verbose:
        print(f"  k 距离图肘部：index={elbow_idx}/{n}，自动 eps={eps:.4f}"
              f"（k_dist 范围 [{k_dists[0]:.4f}, {k_dists[-1]:.4f}]）")
    return eps


def cluster_vectors(vectors: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    对向量矩阵进行聚类。
    返回形状为 (N,) 的标签数组，噪声点标记为 -1（仅 DBSCAN）。
    """
    normed = normalize(vectors, norm="l2")

    if CLUSTERING_METHOD == "dbscan":
        eps = DBSCAN_EPS if DBSCAN_EPS is not None else _auto_eps(normed, MIN_CLUSTER_SIZE, verbose)
        if verbose:
            print(f"  使用 DBSCAN 聚类（eps={eps:.4f}, min_samples={MIN_CLUSTER_SIZE}）")
        db = DBSCAN(eps=eps, min_samples=MIN_CLUSTER_SIZE, metric="cosine")
        labels = db.fit_predict(normed)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        if verbose:
            print(f"  DBSCAN 结果: {n_clusters} 个簇，{n_noise} 个噪声点")

    else:  # kmeans
        if N_CLUSTERS is not None:
            k = N_CLUSTERS
            if verbose:
                print(f"  使用 KMeans 聚类（k={k}，手动指定）")
        else:
            if verbose:
                print(f"  自动确定最优 K 值（范围 {MIN_K}~{min(MAX_K, len(vectors)-1)}）…")
            k = _best_k(normed)

        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(normed)
        if verbose:
            print(f"  KMeans 完成，共 {k} 个簇")

    return labels


def get_cluster_indices(labels: np.ndarray) -> dict[int, list[int]]:
    """
    将标签数组转换为 {cluster_id: [idx, ...]} 字典。
    忽略噪声点（label == -1）。
    """
    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(int(label), []).append(idx)
    return clusters


def get_cluster_silhouette_scores(
    vectors: np.ndarray,
    labels: np.ndarray,
) -> dict[int, float]:
    """
    计算每个簇的平均轮廓系数（适用于 KMeans）。
    返回 {cluster_id: 平均轮廓系数}，忽略噪声点（label == -1）。
    """
    normed = normalize(vectors, norm="l2")
    valid_mask = labels != -1
    if valid_mask.sum() < 2 or len(set(labels[valid_mask])) < 2:
        unique = [l for l in set(labels) if l != -1]
        return {int(l): 0.0 for l in unique}

    sample_scores = silhouette_samples(normed[valid_mask], labels[valid_mask])
    valid_labels = labels[valid_mask]

    scores: dict[int, float] = {}
    for cluster_id in set(valid_labels):
        mask = valid_labels == cluster_id
        scores[int(cluster_id)] = float(sample_scores[mask].mean())
    return scores


def get_cluster_density_scores(
    vectors: np.ndarray,
    labels: np.ndarray,
) -> dict[int, float]:
    """
    计算每个簇的密度分数（适用于 DBSCAN）。
    密度 = 簇内所有样本对的平均余弦相似度，值域 [-1, 1]，越高越密集。
    忽略噪声点（label == -1）。
    """
    normed = normalize(vectors, norm="l2")
    scores: dict[int, float] = {}

    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        pts = normed[mask]          # (m, dim)
        if len(pts) == 1:
            scores[int(cluster_id)] = 0.0
            continue
        # 余弦相似度矩阵 = 归一化向量点积
        sim_matrix = pts @ pts.T    # (m, m)
        # 取上三角（不含对角线）的均值
        m = len(pts)
        triu_sum = (sim_matrix.sum() - np.trace(sim_matrix)) / 2
        n_pairs = m * (m - 1) / 2
        scores[int(cluster_id)] = float(triu_sum / n_pairs)

    return scores


def get_cluster_scores(
    vectors: np.ndarray,
    labels: np.ndarray,
) -> dict[int, float]:
    """
    根据当前聚类方法自动选择评分策略：
    - DBSCAN → 簇内平均余弦相似度（密度）
    - KMeans → 轮廓系数
    """
    if CLUSTERING_METHOD == "dbscan":
        return get_cluster_density_scores(vectors, labels)
    return get_cluster_silhouette_scores(vectors, labels)
