from typing import List, Tuple, Dict
import numpy as np
from config import get_settings

settings = get_settings()


def search_dense(
    query_emb: np.ndarray,
    chunk_embeddings: np.ndarray,
) -> List[Tuple[int, float]]:
    """
    Computes cosine similarity for all chunk embeddings in memory.
    Returns a sorted list of tuples (chunk_index, similarity_score) descending.
    """
    if len(chunk_embeddings) == 0:
        return []

    q_norm = np.linalg.norm(query_emb)
    if q_norm == 0:
        return [(i, 0.0) for i in range(len(chunk_embeddings))]

    # Normalize query embedding
    q_normalized = query_emb / q_norm

    # Normalize chunk embeddings
    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    chunk_embeddings_normalized = chunk_embeddings / norms

    similarities = np.dot(chunk_embeddings_normalized, q_normalized)

    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    return ranked


def reciprocal_rank_fusion(
    dense_results: List[Tuple[int, float]],
    sparse_results: List[Tuple[int, float]],
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF) to combine dense and sparse search results.
    dense_results: List of (chunk_index, score) sorted by score descending
    sparse_results: List of (chunk_index, score) sorted by score descending
    """
    k = settings.rrf_k
    alpha = settings.rrf_dense_weight
    beta = settings.rrf_sparse_weight

    rrf_scores: Dict[int, float] = {}

    # Dense ranking
    for rank, (chunk_index, _) in enumerate(dense_results):
        rrf_scores[chunk_index] = rrf_scores.get(chunk_index, 0.0) + alpha / (k + rank)

    # Sparse ranking
    for rank, (chunk_index, _) in enumerate(sparse_results):
        rrf_scores[chunk_index] = rrf_scores.get(chunk_index, 0.0) + beta / (k + rank)

    # Sort candidates by fused score descending
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates
