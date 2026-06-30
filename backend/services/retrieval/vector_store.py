from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import structlog
import chromadb
from config import get_settings

from models.domain import Chunk

logger = structlog.get_logger(__name__)
settings = get_settings()

chroma_client = chromadb.PersistentClient(path=str(Path(settings.chroma_persist_dir).resolve()))


def get_collection(doc_id: str):
    return chroma_client.get_or_create_collection(
        name=f"doc_{doc_id}",
        metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 200, "hnsw:M": 32},
    )


def get_parent(parent_id: str, doc_id: str) -> Optional[str]:
    """Lookup parent text by parent_id from local JSON store"""
    parents_path = Path("cache") / f"parents_{doc_id}.json"
    if not parents_path.exists():
        return None
    try:
        with open(parents_path, "r", encoding="utf-8") as f:
            parents_dict = json.load(f)
        return parents_dict.get(parent_id)
    except Exception as e:
        logger.exception("parent_lookup_failed", error=str(e))
        return None


def store_chunks(doc_id: str, chunks: List[Chunk], embeddings: np.ndarray) -> None:
    try:
        chroma_client.delete_collection(name=f"doc_{doc_id}")
    except Exception:
        pass
    collection = get_collection(doc_id)

    # Filter to child chunks only (parents are stored in local JSON, not embedded or stored in ChromaDB)
    child_chunks = [c for c in chunks if not c.is_parent]

    ids = [c.chunk_id for c in child_chunks]
    documents = [c.text for c in child_chunks]

    metadatas = []
    for c in child_chunks:
        metadatas.append(
            {
                "doc_id": c.doc_id,
                "section": c.section,
                "page": c.page,
                "parent_id": c.parent_id if c.parent_id is not None else "",
                "chunk_id": c.chunk_id,
                "is_parent": c.is_parent,
                "token_count": c.token_count,
                "section_title": c.section_title,
                "chunk_index": c.chunk_index,
                "keywords": ",".join(c.keywords) if isinstance(c.keywords, list) else str(c.keywords),
                "raw_text": c.raw_text,
            }
        )

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
        metadatas=metadatas,
        documents=documents,
    )


def get_chunks_from_store(doc_id: str) -> Tuple[List[Chunk], np.ndarray]:
    try:
        collection = chroma_client.get_collection(name=f"doc_{doc_id}")
    except Exception:
        return [], np.zeros((0, 3072), dtype="float32")

    results = collection.get(include=["documents", "metadatas", "embeddings"])
    if not results or not results["ids"]:
        return [], np.zeros((0, 3072), dtype="float32")

    chunks = []
    embeddings_list = results["embeddings"]

    for i, cid in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        kws = meta["keywords"].split(",") if meta.get("keywords") else []

        chunks.append(
            Chunk(
                doc_id=meta["doc_id"],
                text=results["documents"][i],
                section_id=meta["section"],
                section_title=meta["section_title"],
                chunk_index=int(meta["chunk_index"]),
                keywords=kws,
                raw_text=meta["raw_text"],
                section=meta["section"],
                page=int(meta["page"]),
                parent_id=meta["parent_id"] if meta["parent_id"] != "" else None,
                chunk_id=meta["chunk_id"],
                is_parent=bool(meta["is_parent"]),
                token_count=int(meta["token_count"]),
            )
        )

    sorted_indices = np.argsort([c.chunk_index for c in chunks])
    chunks = [chunks[idx] for idx in sorted_indices]
    embeddings = np.array([embeddings_list[idx] for idx in sorted_indices], dtype="float32")

    return chunks, embeddings


def build_inverted_index(chunks: List[Chunk]) -> Dict[str, Set[int]]:
    inv = {}
    for i, c in enumerate(chunks):
        kws = c.keywords if hasattr(c, "keywords") else c.get("keywords", [])
        for kw in kws:
            inv.setdefault(kw, set()).add(i)
    return inv


def advanced_universal_retrieval(
    query_embedding: np.ndarray,
    chunks: List[Chunk],
    chunk_embeddings: np.ndarray,
    inv_index: Dict,
    query: str,
    doc_type: str,
    doc_id: str,
    initial_k: int = 20,
    final_k: int = 6,
) -> List[Chunk]:
    """
    RRF-based Hybrid Retrieval (Dense Vector + BM25 Sparse Search).
    Excludes parent chunks and retrieves parent context using parent_id lookup.
    """
    from services.retrieval.hybrid_search import search_dense, reciprocal_rank_fusion
    from services.retrieval.bm25_index import search_bm25

    logger.info("hybrid_retrieval_started", doc_id=doc_id, query=query)

    # 1. Dense search ranking
    dense_results = search_dense(query_embedding, chunk_embeddings)

    # 2. Sparse search ranking (BM25)
    # Ensure BM25 index exists (build lazily if it was deleted or missing)
    bm25_path = Path("cache") / f"bm25_{doc_id}.pkl"
    if not bm25_path.exists():
        logger.info("bm25_index_lazy_building", doc_id=doc_id)
        from services.retrieval.bm25_index import build_and_save_bm25_index

        build_and_save_bm25_index(doc_id, [c.text for c in chunks])

    sparse_results = search_bm25(doc_id, query)

    # 3. Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results)

    # 4. Take the top final_k candidates and map them to their parent chunks
    top_indices = [idx for idx, _ in fused_results[:final_k]]
    retrieved_chunks = [chunks[idx] for idx in top_indices]

    final_retrieved = []
    for chunk in retrieved_chunks:
        parent_id = chunk.parent_id
        if parent_id:
            parent_text = get_parent(parent_id, doc_id)
            if parent_text:
                chunk_copy = chunk.copy()
                chunk_copy.text = parent_text
                final_retrieved.append(chunk_copy)
                continue
        final_retrieved.append(chunk)

    logger.info(
        "hybrid_retrieval_completed",
        doc_id=doc_id,
        returned_chunks=len(final_retrieved),
        top_scores=[round(score, 5) for _, score in fused_results[:final_k]],
    )
    return final_retrieved
