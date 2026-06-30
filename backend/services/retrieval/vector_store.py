from collections import defaultdict
import re
from typing import Dict, List, Set, Tuple
import numpy as np
import structlog

from utils.logging import load_spacy_model

logger = structlog.get_logger(__name__)
nlp = load_spacy_model()


class NumpyL2Index:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.astype("float32")

    def search(self, queries: np.ndarray, top_k: int):
        query_vectors = np.atleast_2d(queries).astype("float32")
        distances = np.sum((query_vectors[:, None, :] - self.vectors[None, :, :]) ** 2, axis=2)
        indices = np.argsort(distances, axis=1)[:, :top_k]
        sorted_distances = np.take_along_axis(distances, indices, axis=1)
        return sorted_distances, indices


def build_faiss_index(vectors: np.ndarray, use_pq: bool = False):
    """
    Build a lightweight in-process L2 index with the old FAISS-compatible search API.
    """
    if use_pq:
        logger.warning("pq_not_supported", fallback="numpy_l2")
    return NumpyL2Index(vectors)


def extract_query_keywords(query: str, top_k=5) -> List[str]:
    doc = nlp(query)
    candidates = [tok.lemma_.lower() for tok in doc if tok.pos_ in {"NOUN", "PROPN", "ADJ"} and len(tok) > 3]
    return list(dict.fromkeys(candidates))[:top_k]


def build_inverted_index(chunks: List[Dict]) -> Dict[str, Set[int]]:
    inv = {}
    for i, c in enumerate(chunks):
        for kw in c.get("keywords", []):
            inv.setdefault(kw, set()).add(i)
    return inv


def filter_candidates(inv_index: Dict[str, Set[int]], query: str, total: int, verbose=False) -> List[int]:
    """
    Return a list of candidate chunk indices based on keyword match.
    Uses intersection if possible, else falls back to union or full set.
    """
    kws = extract_query_keywords(query)
    matched_sets = [inv_index.get(kw, set()) for kw in kws]

    if not matched_sets:
        return list(range(total))

    cand = set.intersection(*matched_sets)
    if len(cand) < 10:
        cand = set.union(*matched_sets)

    if len(cand) < 10 or len(cand) > 500:
        cand = set(range(total))

    if verbose:
        logger.info(
            "legacy_log",
            message=f"[Keyword Filter] Query: '{query}' → Keywords: {kws}",
        )
        logger.info("legacy_log", message=f"[Keyword Filter] Candidates: {len(cand)} chunks")
    return list(cand)


def search_masked_subset(qvec: np.ndarray, candidate_ids, all_vectors: np.ndarray, top_k=5):
    cids = [int(i) for i in candidate_ids]
    if not cids:
        return np.array([]), np.array([])
    vecs_subset = all_vectors[cids]
    tmp = NumpyL2Index(vecs_subset)
    D, I = tmp.search(qvec.reshape(1, -1), top_k)
    final = [cids[i] for i in I[0]]
    return D[0], final


def filter_universal_candidates(
    inv_index: Dict[str, Set[int]], query: str, chunks: List[Dict], doc_id: str
) -> List[int]:
    """
    Universal keyword filtering scoped to the current doc_id.
    """
    doc_indices = {i for i, c in enumerate(chunks) if c.get("doc_id") == doc_id}
    if not doc_indices:
        return []

    query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    numbers = set(re.findall(r"\b\d+\b", query))
    special_terms = set(re.findall(r"\b(?:section|article|chapter|part|clause)\s+\w+\b", query.lower()))
    all_terms = query_words | numbers | special_terms

    matched_sets: List[Set[int]] = []
    for term in all_terms:
        if term in inv_index:
            hits = inv_index[term] & doc_indices
            if hits:
                matched_sets.append(hits)

    if not matched_sets:
        return list(doc_indices)[:50]

    if len(matched_sets) > 1:
        candidates = set.intersection(*matched_sets)
    else:
        candidates = matched_sets[0]

    if len(candidates) < 10:
        candidates = set.union(*matched_sets)

    if len(candidates) < 5:
        candidates = doc_indices

    return list(candidates)


def get_semantic_candidates(query_emb: np.ndarray, chunk_embeddings: np.ndarray, k: int) -> List[int]:
    query_norm = query_emb / np.linalg.norm(query_emb)
    chunk_norms = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

    similarities = np.dot(chunk_norms, query_norm)
    top_indices = np.argsort(similarities)[::-1][:k]

    return top_indices.tolist()


def get_section_based_candidates(query: str, chunks: List[Dict], doc_type: str) -> List[int]:
    candidates = []
    query_lower = query.lower()

    section_refs = re.findall(r"\b(?:section|article|chapter|part|clause)\s+(\w+)\b", query_lower)

    for i, chunk in enumerate(chunks):
        section_id = chunk.get("section_id", "")
        section_title = chunk.get("section_title", "").lower()

        if any(ref in section_id.lower() for ref in section_refs):
            candidates.append(i)

        if any(word in section_title for word in query_lower.split() if len(word) > 3):
            candidates.append(i)

    return candidates[:15]


def ensemble_candidate_selection(candidate_sets: List[Set[int]], target_k: int) -> List[int]:
    candidate_scores = defaultdict(int)

    for i, candidate_set in enumerate(candidate_sets):
        weight = [0.4, 0.4, 0.2][i] if i < 3 else 0.1
        for candidate in candidate_set:
            candidate_scores[candidate] += weight

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return [cand for cand, _ in sorted_candidates[:target_k]]


def calculate_universal_scores(
    query: str,
    chunk: Dict,
    query_emb: np.ndarray,
    chunk_emb: np.ndarray,
    doc_type: str,
) -> Dict[str, float]:
    chunk_text = chunk.get("raw_text", chunk["text"]).lower()
    query_lower = query.lower()

    scores = {}

    scores["semantic"] = float(np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)))

    query_words = set(re.findall(r"\b\w{3,}\b", query_lower))
    chunk_words = set(re.findall(r"\b\w{3,}\b", chunk_text))
    overlap = len(query_words.intersection(chunk_words))
    scores["keyword"] = overlap / len(query_words) if query_words else 0

    section_title = chunk.get("section_title", "").lower()
    title_overlap = len(set(query_lower.split()).intersection(set(section_title.split())))
    scores["section_relevance"] = title_overlap / len(query_lower.split()) if query_lower.split() else 0

    scores["final"] = 0.7 * scores["semantic"] + 0.3 * scores["keyword"]

    important_terms = len(re.findall(r"\b(?:\d+|[A-Z]{2,})\b", chunk_text))
    text_length = len(chunk_text.split())
    scores["content_density"] = min(important_terms / max(text_length, 1), 1.0)

    try:
        section_num = float(re.search(r"(\d+)", chunk.get("section_id", "999")).group(1))
        scores["position"] = max(0.1, 1.0 - (section_num / 50.0))
    except Exception:
        scores["position"] = 0.5

    text_len = len(chunk_text.split())
    scores["length"] = min(1.0, text_len / 100.0) if text_len < 300 else max(0.3, 300.0 / text_len)

    scores["type_specific"] = calculate_type_specific_score(query, chunk, doc_type)

    scores["final"] = 0.7 * scores["semantic"] + 0.3 * scores["keyword"]

    return scores


def calculate_type_specific_score(query: str, chunk: Dict, doc_type: str) -> float:
    chunk_text = chunk.get("raw_text", chunk["text"]).lower()
    query_lower = query.lower()

    type_scores = {
        "legal": calculate_legal_score,
        "insurance": calculate_insurance_score,
        "technical": calculate_technical_score,
        "academic": calculate_academic_score,
    }

    if doc_type in type_scores:
        return type_scores[doc_type](query_lower, chunk_text)

    return 0.5


def calculate_legal_score(query: str, chunk_text: str) -> float:
    score = 0.0

    legal_terms = [
        "constitution",
        "article",
        "fundamental",
        "right",
        "duty",
        "amendment",
    ]
    query_legal = sum(1 for term in legal_terms if term in query)
    chunk_legal = sum(1 for term in legal_terms if term in chunk_text)

    if query_legal > 0:
        score += min(chunk_legal / query_legal, 1.0) * 0.5

    query_nums = set(re.findall(r"\b\d+[a-z]?\b", query))
    chunk_nums = set(re.findall(r"\b\d+[a-z]?\b", chunk_text))
    if query_nums and chunk_nums:
        score += len(query_nums.intersection(chunk_nums)) / len(query_nums) * 0.5

    return min(score, 1.0)


def calculate_insurance_score(query: str, chunk_text: str) -> float:
    score = 0.0

    insurance_terms = [
        "premium",
        "coverage",
        "claim",
        "waiting",
        "period",
        "policy",
    ]
    query_terms = sum(1 for term in insurance_terms if term in query)
    chunk_terms = sum(1 for term in insurance_terms if term in chunk_text)

    if query_terms > 0:
        score += min(chunk_terms / query_terms, 1.0) * 0.6

    query_nums = re.findall(r"\b\d+\b", query)
    chunk_nums = re.findall(r"\b\d+\b", chunk_text)
    if query_nums:
        num_matches = sum(1 for num in query_nums if num in chunk_nums)
        score += (num_matches / len(query_nums)) * 0.4

    return min(score, 1.0)


def calculate_technical_score(query: str, chunk_text: str) -> float:
    score = 0.0

    tech_terms = [
        "part",
        "component",
        "system",
        "procedure",
        "specification",
        "manual",
    ]
    query_terms = sum(1 for term in tech_terms if term in query)
    chunk_terms = sum(1 for term in tech_terms if term in chunk_text)

    if query_terms > 0:
        score += min(chunk_terms / query_terms, 1.0) * 0.5

    query_codes = re.findall(r"\b[A-Z]{2,}\d+\b|\b\d+[A-Z]+\b", query.upper())
    chunk_codes = re.findall(r"\b[A-Z]{2,}\d+\b|\b\d+[A-Z]+\b", chunk_text.upper())
    if query_codes:
        code_matches = sum(1 for code in query_codes if code in chunk_codes)
        score += (code_matches / len(query_codes)) * 0.5

    return min(score, 1.0)


def calculate_academic_score(query: str, chunk_text: str) -> float:
    score = 0.0

    academic_terms = [
        "theorem",
        "principle",
        "equation",
        "formula",
        "chapter",
        "research",
    ]
    query_terms = sum(1 for term in academic_terms if term in query)
    chunk_terms = sum(1 for term in academic_terms if term in chunk_text)

    if query_terms > 0:
        score += min(chunk_terms / query_terms, 1.0) * 0.6

    query_math = len(re.findall(r"[=+\-*/∫∑∆αβγδεθλμπσφψω]", query))
    chunk_math = len(re.findall(r"[=+\-*/∫∑∆αβγδεθλμπσφψω]", chunk_text))
    if query_math > 0:
        score += min(chunk_math / query_math, 1.0) * 0.4

    return min(score, 1.0)


def advanced_universal_retrieval(
    query_embedding: np.ndarray,
    chunks: List[Dict],
    chunk_embeddings: np.ndarray,
    inv_index: Dict,
    query: str,
    doc_type: str,
    doc_id: str,
    initial_k: int = 20,
    final_k: int = 6,
) -> List[Dict]:
    """Advanced retrieval system that adapts to any document type with doc isolation"""
    candidates_sets = []

    keyword_candidates = filter_universal_candidates(inv_index, query, chunks, doc_id)
    candidates_sets.append(set(keyword_candidates))

    semantic_candidates_all = get_semantic_candidates(query_embedding, chunk_embeddings, initial_k * 4)
    semantic_candidates = [i for i in semantic_candidates_all if chunks[i]["doc_id"] == doc_id]
    candidates_sets.append(set(semantic_candidates))

    section_candidates_all = get_section_based_candidates(query, chunks, doc_type)
    section_candidates = [i for i in section_candidates_all if chunks[i]["doc_id"] == doc_id]
    candidates_sets.append(set(section_candidates))

    final_candidates = ensemble_candidate_selection(candidates_sets, initial_k)
    final_candidates = [i for i in final_candidates if chunks[i]["doc_id"] == doc_id]

    candidate_chunks = [chunks[i] for i in final_candidates]
    scores = []

    for i, chunk in enumerate(candidate_chunks):
        score_components = calculate_universal_scores(
            query,
            chunk,
            query_embedding,
            chunk_embeddings[final_candidates[i]],
            doc_type,
        )

        final_score = (
            0.25 * score_components["semantic"]
            + 0.20 * score_components["keyword"]
            + 0.15 * score_components["section_relevance"]
            + 0.15 * score_components["content_density"]
            + 0.10 * score_components["position"]
            + 0.10 * score_components["length"]
            + 0.05 * score_components["type_specific"]
        )

        scores.append((final_score, i))

    scores.sort(reverse=True)
    return [candidate_chunks[i] for _, i in scores[:final_k]]


def get_adaptive_contextual_chunks(
    selected_chunks: List[Dict],
    all_chunks: List[Dict],
    doc_type: str,
    context_strategy: str = "adaptive",
) -> List[Dict]:
    enhanced_chunks = []

    for chunk in selected_chunks:
        if context_strategy == "adaptive":
            context_chunks = get_adaptive_context(chunk, all_chunks, doc_type)
        elif context_strategy == "hierarchical":
            context_chunks = get_hierarchical_context(chunk, all_chunks)
        else:
            context_chunks = get_sequential_context(chunk, all_chunks)

        enhanced_chunk = combine_chunks_intelligently(chunk, context_chunks, doc_type)
        enhanced_chunks.append(enhanced_chunk)

    return enhanced_chunks


def get_hierarchical_context(chunk: Dict, all_chunks: List[Dict]) -> List[Dict]:
    current_section = chunk.get("section_id", "")
    context = []

    hierarchy_parts = current_section.split(".")

    for i in range(len(hierarchy_parts) - 1, 0, -1):
        parent_id = ".".join(hierarchy_parts[:i])
        parent_chunk = next((c for c in all_chunks if c["section_id"] == parent_id), None)
        if parent_chunk:
            context.append(parent_chunk)
            break

    if len(hierarchy_parts) >= 2:
        sibling_prefix = ".".join(hierarchy_parts[:-1])
        for c in all_chunks:
            sid = c.get("section_id", "")
            if sid.startswith(sibling_prefix) and sid != current_section:
                context.append(c)
                if len(context) >= 3:
                    break

    return context


def get_adaptive_context(chunk: Dict, all_chunks: List[Dict], doc_type: str) -> List[Dict]:
    context_chunks = []

    if doc_type == "legal":
        context_chunks.extend(get_legal_context(chunk, all_chunks))
    elif doc_type == "academic":
        context_chunks.extend(get_academic_context(chunk, all_chunks))
    elif doc_type == "technical":
        context_chunks.extend(get_technical_context(chunk, all_chunks))
    else:
        context_chunks.extend(get_sequential_context(chunk, all_chunks))

    return context_chunks[:3]


def get_legal_context(chunk: Dict, all_chunks: List[Dict]) -> List[Dict]:
    context = []
    section_id = chunk["section_id"]

    for other_chunk in all_chunks:
        other_id = other_chunk["section_id"]

        if section_id.split(".")[0] == other_id.split(".")[0] and other_id != section_id:
            context.append(other_chunk)

        if f"article {section_id}" in other_chunk["text"].lower():
            context.append(other_chunk)

    return context


def get_academic_context(chunk: Dict, all_chunks: List[Dict]) -> List[Dict]:
    context = []
    chunk_text = chunk["text"].lower()

    for other_chunk in all_chunks:
        other_text = other_chunk["text"].lower()

        if "definition" in other_text or "defined as" in other_text:
            key_terms = extract_academic_terms(chunk_text)
            if any(term in other_text for term in key_terms):
                context.append(other_chunk)

        if "example" in other_text or "application" in other_text:
            if any(term in other_text for term in extract_academic_terms(chunk_text)):
                context.append(other_chunk)

    return context


def get_technical_context(chunk: Dict, all_chunks: List[Dict]) -> List[Dict]:
    context = []

    for other_chunk in all_chunks:
        other_text = other_chunk["text"].lower()

        if "before" in other_text or "prerequisite" in other_text or "first" in other_text:
            context.append(other_chunk)

        if "warning" in other_text or "caution" in other_text or "note" in other_text:
            context.append(other_chunk)

    return context


def get_sequential_context(chunk: Dict, all_chunks: List[Dict]) -> List[Dict]:
    context = []
    current_idx = None

    for i, c in enumerate(all_chunks):
        if c["section_id"] == chunk["section_id"] and c.get("chunk_index") == chunk.get("chunk_index"):
            current_idx = i
            break

    if current_idx is not None:
        for offset in [-1, 1]:
            neighbor_idx = current_idx + offset
            if 0 <= neighbor_idx < len(all_chunks):
                context.append(all_chunks[neighbor_idx])

    return context


def extract_academic_terms(text: str) -> List[str]:
    terms = []
    terms.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text))
    terms.extend(
        re.findall(
            r"\b(?:theorem|lemma|corollary|principle|law|equation|formula)\s+\w+\b",
            text.lower(),
        )
    )
    return list(set(terms))[:5]


def combine_chunks_intelligently(main_chunk: Dict, context_chunks: List[Dict], doc_type: str) -> Dict:
    if not context_chunks:
        return main_chunk

    combined_text = f"=== MAIN CONTENT ===\n{main_chunk['text']}"

    for ctx_chunk in context_chunks:
        if doc_type == "legal":
            combined_text += f"\n\n=== RELATED PROVISION ===\n{ctx_chunk['text']}"
        elif doc_type == "academic":
            combined_text += f"\n\n=== SUPPORTING MATERIAL ===\n{ctx_chunk['text']}"
        elif doc_type == "technical":
            combined_text += f"\n\n=== RELATED PROCEDURE ===\n{ctx_chunk['text']}"
        else:
            combined_text += f"\n\n=== ADDITIONAL CONTEXT ===\n{ctx_chunk['text']}"

    enhanced_chunk = main_chunk.copy()
    enhanced_chunk["text"] = combined_text
    enhanced_chunk["context_count"] = len(context_chunks)

    return enhanced_chunk
