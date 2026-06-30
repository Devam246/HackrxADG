import hashlib
import re
from typing import List, Tuple
import numpy as np
from sklearn.decomposition import PCA
import structlog
import google.generativeai as genai
from config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

genai.configure(api_key=settings.gemini_api_key)


def embed_text(
    texts: List[str],
    task_type: str = "retrieval_document",
    model: str = "gemini-embedding-001",
    batch_size: int = 100,
) -> np.ndarray:
    """Generate real Gemini embeddings (gemini-embedding-001) with batching (max 100 per API call)."""
    if not texts:
        return np.zeros((0, 3072), dtype="float32")

    if isinstance(texts, str):
        texts = [texts]

    model_name = "gemini-embedding-001"
    all_embeddings = []

    # Batch embeddings: max 100 per API call
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = genai.embed_content(model=f"models/{model_name}", content=batch, task_type=task_type)
        all_embeddings.extend(response["embedding"])

    return np.array(all_embeddings, dtype="float32")


def reduce_dimensions(vectors: np.ndarray, target_dim: int = 512) -> Tuple[np.ndarray, PCA]:
    """
    Fit PCA on `vectors`, auto-capping to <= min(n_samples, n_features),
    and making dims divisible by 8 for FAISS PQ. Returns (reduced_vectors, pca_model).
    """
    n_samples, n_features = vectors.shape
    max_dim = min(n_samples, n_features)

    td = min(target_dim, max_dim)
    td -= td % 8

    pca = PCA(n_components=td)
    reduced = pca.fit_transform(vectors)
    logger.info(
        "legacy_log",
        message=f"[PCA] d={n_features} → {td}, retained {sum(pca.explained_variance_ratio_)*100:.2f}% variance.",
    )
    return reduced.astype("float32"), pca
