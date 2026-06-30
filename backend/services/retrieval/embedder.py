import hashlib
import re
from typing import List, Tuple
import numpy as np
from sklearn.decomposition import PCA
import structlog

logger = structlog.get_logger(__name__)


def embed_voyage(texts: List[str], model: str = "local-hash", batch_size=300) -> np.ndarray:
    """Legacy-named local embedding shim kept for V1 monolith compatibility."""
    # TODO: V3 — replace with real gemini-embedding-001
    dimensions = 384
    embeddings = np.zeros((len(texts), dimensions), dtype="float32")

    for row, text in enumerate(texts):
        tokens = re.findall(r"\b\w+\b", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            embeddings[row, index] += sign

        norm = np.linalg.norm(embeddings[row])
        if norm > 0:
            embeddings[row] /= norm

    return embeddings


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
