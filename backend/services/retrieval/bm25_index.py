import pickle
from pathlib import Path
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import re
import structlog

logger = structlog.get_logger(__name__)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def bm25_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words of length >= 2 for BM25 retrieval."""
    return [word for word in re.findall(r"\b\w+\b", text.lower()) if len(word) > 1]


def build_and_save_bm25_index(doc_id: str, corpus: List[str]) -> None:
    """Builds and serializes a BM25 index for the given doc_id and corpus of child chunk texts."""
    tokenized_corpus = [bm25_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    index_path = CACHE_DIR / f"bm25_{doc_id}.pkl"
    try:
        with open(index_path, "wb") as f:
            pickle.dump(bm25, f)
        logger.info("bm25_index_saved", doc_id=doc_id, corpus_size=len(corpus))
    except Exception as e:
        logger.exception("bm25_index_save_failed", doc_id=doc_id, error=str(e))


def load_bm25_index(doc_id: str) -> BM25Okapi:
    """Loads a BM25 index from cache for the given doc_id."""
    index_path = CACHE_DIR / f"bm25_{doc_id}.pkl"
    if not index_path.exists():
        raise FileNotFoundError(f"BM25 index not found for doc_id: {doc_id}")
    with open(index_path, "rb") as f:
        return pickle.load(f)


def search_bm25(doc_id: str, query: str) -> List[Tuple[int, float]]:
    """
    Searches the BM25 index for a query.
    Returns a sorted list of tuples (chunk_index, score) representing child chunks.
    """
    try:
        bm25 = load_bm25_index(doc_id)
    except FileNotFoundError:
        logger.warning("bm25_index_not_found_for_search", doc_id=doc_id)
        return []

    tokenized_query = bm25_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Pair with chunk index and sort descending by BM25 score
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked
