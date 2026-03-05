import os

# 使用 HF-Mirror 加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
from sentence_transformers import SentenceTransformer
from config import SENTENCE_TRANSFORMER_MODEL


_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  加载向量化模型: {SENTENCE_TRANSFORMER_MODEL}")
        _model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    return _model


def keywords_to_vector(keywords: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    将一组关键词各自转换为向量后求和，得到该标题的代表向量。
    若关键词为空，返回零向量。
    """
    if not keywords:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros(dim, dtype=np.float32)

    embeddings = model.encode(keywords, convert_to_numpy=True, show_progress_bar=False)
    summed = embeddings.sum(axis=0)
    # L2 归一化，防止关键词数量不同导致向量幅值差异过大
    norm = np.linalg.norm(summed)
    if norm > 0:
        summed = summed / norm
    return summed.astype(np.float32)


def vectorize_keywords(keywords_list: list[list[str]], verbose: bool = True) -> np.ndarray:
    """
    对所有标题的关键词列表批量向量化。
    返回形状为 (N, dim) 的 numpy 数组。
    """
    model = get_model()

    # 收集所有不重复的关键词，一次性编码，提升效率
    unique_kws = list({kw for kws in keywords_list for kw in kws})
    if verbose:
        print(f"  共 {len(unique_kws)} 个不重复关键词，开始批量编码…")

    if unique_kws:
        all_embeddings = model.encode(
            unique_kws,
            convert_to_numpy=True,
            show_progress_bar=verbose,
            batch_size=256,
        )
        kw_to_vec: dict[str, np.ndarray] = {
            kw: emb for kw, emb in zip(unique_kws, all_embeddings)
        }
    else:
        kw_to_vec = {}

    dim = model.get_sentence_embedding_dimension()
    vectors = []
    for keywords in keywords_list:
        if not keywords:
            vectors.append(np.zeros(dim, dtype=np.float32))
            continue
        embs = np.stack([kw_to_vec[kw] for kw in keywords], axis=0)
        summed = embs.sum(axis=0)
        norm = np.linalg.norm(summed)
        if norm > 0:
            summed = summed / norm
        vectors.append(summed.astype(np.float32))

    return np.stack(vectors, axis=0)
