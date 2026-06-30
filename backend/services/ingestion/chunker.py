from collections import Counter, defaultdict
import hashlib
import os
import pickle
from pathlib import Path
import re
from typing import Dict, List, Tuple
import unicodedata
import numpy as np
import structlog
import tiktoken

from services.ingestion.classifier import DocumentClassifier
from services.ingestion.downloader import download_file
from services.ingestion.parsers import (
    extract_clean_text_from_eml,
    extract_text_from_docx,
    extract_text_from_pdf,
)
from services.retrieval.embedder import embed_voyage

logger = structlog.get_logger(__name__)

# Initialize tokenizer for accurate token counts
tokenizer = tiktoken.get_encoding("cl100k_base")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_doc_id(url: str) -> str:
    """
    Creates a unique and safe filename hash from a document URL.
    This helps us create a unique ID for caching related to this specific document.
    """
    return hashlib.sha256(url.encode()).hexdigest()


def clean_text(text: str) -> str:
    """
    Clean PDF text while preserving structural layout for clause detection.
    """
    # Remove repeated headers/footers
    lines = text.splitlines()
    freq = {}
    for line in lines:
        freq[line] = freq.get(line, 0) + 1
    filtered = [l for l in lines if freq[l] < 3]
    text = "\n".join(filtered)

    # Fix broken hyphenated words (like "cover-\nage")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Remove page markers and numbers
    text = re.sub(r"=== Page \d+ ===", "", text)
    text = re.sub(r"Page\s*\d+(\s*of\s*\d+)?", "", text, flags=re.IGNORECASE)

    # Normalize line endings
    text = re.sub(r"\r\n", "\n", text)

    # Collapse multiple blank lines but preserve section layout
    text = re.sub(r"\n{3,}", "\n\n", text)  # Convert 3+ \n to 2
    text = re.sub(r"[ \t]+\n", "\n", text)  # Remove trailing spaces on lines

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    return text.strip()


def tag_numbered_headings(text: str) -> str:
    """
    Tag numbered headings that simulate clause structure:
    e.g., "1. Introduction", "2.3.1 Waiting Period"
    """
    pattern = r"(?m)^\s*(\d+(\.\d+)*)([\s:–-]+)([A-Z][^\n]{3,100})"
    return re.sub(pattern, r"\n\n\1\3\4", text)


def extract_clauses_flexible(text: str):
    """
    Detect clauses like '3.1.14 Maternity' or '2 AYUSH' regardless of line breaks.
    Returns list of {clause_id, clause_title, clause_body}
    """
    matches = list(
        re.finditer(
            r"\b(\d{1,2}(?:\.\d{1,2}){0,2})[\s\-:]+([A-Z][^\d]{3,100}?)\b(?=\s+\d|\s+[A-Z])",
            text,
        )
    )
    clauses = []

    for i in range(len(matches)):
        clause_id = matches[i].group(1).strip()
        clause_title = matches[i].group(2).strip()
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause_body = text[start:end].strip()

        clauses.append(
            {
                "clause_id": clause_id,
                "clause_title": clause_title,
                "clause_body": clause_body,
            }
        )

    return clauses


def extract_structured_sections(text: str, doc_type: str) -> List[Dict]:
    """
    Extract structured sections/clauses based on document type and flexible clause patterns.
    """
    patterns = {
        "legal": [
            r"(?m)^\s*(Article\s+\d+[A-Z]*[\.\s]+[^\n]{10,150})",
            r"(?m)^\s*(Section\s+\d+[A-Z]*[\.\s]+[^\n]{10,150})",
            r"(?m)^\s*(Chapter\s+\d+[A-Z]*[\.\s]+[^\n]{10,150})",
            r"(?m)^\s*(\d+[\.\)]\s+[A-Z][^\n]{10,150})",
        ],
        "insurance": [
            r"(?m)^\s*(\d+(?:\.\d+)*[\s\-:]+[A-Z][^\n]{3,150})",
            r"(?m)^\s*(SECTION\s+\d+[^\n]{10,150})",
            r"(?m)^\s*(CLAUSE\s+\d+[^\n]{10,150})",
        ],
        "technical": [
            r"(?m)^\s*(\d+(?:\.\d+)*[\s\-:]+[A-Z][^\n]{5,150})",
            r"(?m)^\s*(PART\s+[A-Z\d]+[^\n]{10,150})",
            r"(?m)^\s*(COMPONENT\s+\d+[^\n]{10,150})",
        ],
        "academic": [
            r"(?m)^\s*(Chapter\s+\d+[^\n]{10,150})",
            r"(?m)^\s*(\d+(?:\.\d+)*[\s\-:]+[A-Z][^\n]{5,150})",
            r"(?m)^\s*(THEOREM\s+\d+[^\n]{10,150})",
            r"(?m)^\s*(PRINCIPLE\s+\d+[^\n]{10,150})",
        ],
        "general": [
            r"(?m)^\s*(\d+(?:\.\d+)*[\s\-:]+[A-Z][^\n]{3,150})",
            r"(?m)^\s*([A-Z][A-Z\s]{5,50}:)",
            r"(?m)^\s*(SECTION\s+[A-Z\d]+[^\n]{5,150})",
        ],
    }

    flexible_clause_pattern = (
        r"(?m)^\s*\b(\d{1,2}(?:\.\d{1,2}){0,2})[\s\-:]+([A-Z][^\n]{3,100}?)\b(?=\s+\d|\s+[A-Z]|\s*\n\n)"
    )

    doc_patterns = patterns.get(doc_type.lower(), patterns["general"]) + [flexible_clause_pattern]

    matches = []
    for pattern in doc_patterns:
        matches.extend(list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)))

    matches.sort(key=lambda m: m.start())
    structured_sections = []

    for i, match in enumerate(matches):
        section_header = match.group(0).strip()
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_id = None
        section_title = section_header

        id_match = re.search(
            r"^\s*(\d+(\.\d+)*|Article\s+\d+|Section\s+\d+|Chapter\s+\d+|CLAUSE\s+\d+|PART\s+[A-Z\d]+)",
            section_header,
            re.IGNORECASE,
        )
        if id_match:
            section_id = id_match.group(1).strip()
            title_match = re.search(
                re.escape(id_match.group(0)) + r"[\s:–-]+([^\n]{3,150})",
                section_header,
                re.IGNORECASE,
            )
            if title_match:
                section_title = title_match.group(1).strip()
            elif len(section_header.split()) > 1:
                section_title = " ".join(section_header.split()[1:]).strip()

        if not section_id:
            section_id = f"Section_{i+1}"

        section_body = text[start_pos:end_pos].strip()

        structured_sections.append(
            {
                "section_id": section_id,
                "section_title": section_title,
                "section_body": section_body,
            }
        )

    if not structured_sections:
        logger.info(
            "legacy_log",
            message="⚠️ No structured sections found. Creating artificial sections.",
        )
        return create_artificial_sections(text)

    return structured_sections


def create_artificial_sections(text: str, section_size: int = 1000) -> List[Dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    sections = []
    current_section = ""
    section_id = 1

    for para in paragraphs:
        if len(current_section) + len(para) > section_size and current_section:
            title = extract_section_title(current_section)
            sections.append(
                {
                    "section_id": str(section_id),
                    "section_title": title,
                    "section_body": current_section,
                }
            )
            section_id += 1
            current_section = para
        else:
            current_section += "\n\n" + para if current_section else para

    if current_section:
        title = extract_section_title(current_section)
        sections.append(
            {
                "section_id": str(section_id),
                "section_title": title,
                "section_body": current_section,
            }
        )

    return sections


def extract_section_title(text: str) -> str:
    sentences = text.split("\n")[:3]

    for sentence in sentences:
        sentence = sentence.strip()
        if 10 < len(sentence) < 100:
            if sentence.isupper() or sentence.istitle():
                return sentence
            if 3 < len(sentence.split()) < 15:
                return sentence

    words = text.split()[:8]
    return " ".join(words) + "..." if len(words) == 8 else " ".join(words)


def universal_hybrid_chunking(
    sections: List[Dict],
    max_tokens: int = 600,
    overlap_tokens: int = 150,
    doc_type: str = "general",
) -> List[Dict]:
    chunks = []

    for section in sections:
        body = section.get("section_body", "")
        if not body.strip():
            continue

        if doc_type in ["legal", "academic"]:
            chunks.extend(chunk_by_paragraphs(section, max_tokens, overlap_tokens))
        elif doc_type == "technical":
            chunks.extend(chunk_by_procedures(section, max_tokens, overlap_tokens))
        else:
            chunks.extend(chunk_by_tokens(section, max_tokens, overlap_tokens))

    return chunks


def chunk_by_paragraphs(section: Dict, max_tokens: int, overlap_tokens: int) -> List[Dict]:
    paragraphs = [p.strip() for p in section["section_body"].split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para))

        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append(
                create_chunk(
                    section,
                    current_chunk,
                    len(chunks),
                    extract_universal_keywords(current_chunk),
                )
            )

            if overlap_tokens > 0:
                overlap_text = get_last_n_tokens(current_chunk, overlap_tokens)
                current_chunk = overlap_text + "\n\n" + para
                current_tokens = len(tokenizer.encode(current_chunk))
            else:
                current_chunk = para
                current_tokens = para_tokens
        else:
            current_chunk += "\n\n" + para if current_chunk else para
            current_tokens += para_tokens

    if current_chunk:
        chunks.append(
            create_chunk(
                section,
                current_chunk,
                len(chunks),
                extract_universal_keywords(current_chunk),
            )
        )

    return chunks


def chunk_by_procedures(section: Dict, max_tokens: int, overlap_tokens: int) -> List[Dict]:
    body = section["section_body"]

    step_patterns = [
        r"(?m)^\s*(?:Step\s+)?\d+[\.\)]\s+[^\n]+",
        r"(?m)^\s*[a-z][\.\)]\s+[^\n]+",
        r"(?m)^\s*[•\-\*]\s+[^\n]+",
        r"(?m)^\s*(?:First|Second|Third|Finally|Next|Then)[^\n]+",
    ]

    steps = []
    for pattern in step_patterns:
        matches = re.findall(pattern, body, re.IGNORECASE)
        if len(matches) >= 2:
            steps = matches
            break

    if not steps:
        return chunk_by_paragraphs(section, max_tokens, overlap_tokens)

    chunks = []
    current_chunk = ""

    for step in steps:
        if len(tokenizer.encode(current_chunk + step)) > max_tokens and current_chunk:
            chunks.append(
                create_chunk(
                    section,
                    current_chunk,
                    len(chunks),
                    extract_universal_keywords(current_chunk),
                )
            )
            current_chunk = step
        else:
            current_chunk += "\n" + step if current_chunk else step

    if current_chunk:
        chunks.append(
            create_chunk(
                section,
                current_chunk,
                len(chunks),
                extract_universal_keywords(current_chunk),
            )
        )

    return chunks


def chunk_by_tokens(section: Dict, max_tokens: int, overlap_tokens: int) -> List[Dict]:
    body = section["section_body"]
    tokens = tokenizer.encode(body)
    total_tokens = len(tokens)
    chunks = []
    start = 0

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append(
            create_chunk(
                section,
                chunk_text,
                len(chunks),
                extract_universal_keywords(chunk_text),
            )
        )

        start += max_tokens - overlap_tokens

    return chunks


def create_chunk(
    section: Dict,
    text: str,
    chunk_index: int,
    keywords: List[str],
    doc_id: str = "unknown",
) -> Dict:
    return {
        "doc_id": section.get("doc_id", doc_id),
        "text": f"{section['section_id']} {section['section_title']}\n{text}",
        "section_id": section["section_id"],
        "section_title": section["section_title"],
        "chunk_index": chunk_index,
        "keywords": keywords,
        "raw_text": text,
    }


def load_or_create_cache(url: str, raw_text: str, doc_type: str) -> Tuple[List[Dict], np.ndarray, str]:
    doc_id = get_doc_id(url)
    chunks_path = CACHE_DIR / f"{doc_id}_chunks.pkl"
    embeddings_path = CACHE_DIR / f"{doc_id}_embeddings.npy"

    if chunks_path.exists() and embeddings_path.exists():
        logger.info(
            "legacy_log",
            message=f"✅ Using cached chunks and embeddings for {url}",
        )
        chunks = pickle.load(open(chunks_path, "rb"))
        embeddings = np.load(embeddings_path)
    else:
        logger.info("legacy_log", message=f"⚡ Creating new chunks and embeddings for {url}")
        sections = extract_structured_sections(raw_text, doc_type)
        for section in sections:
            section["doc_id"] = doc_id

        chunks = universal_hybrid_chunking(sections, doc_type=doc_type)

        for c in chunks:
            c["doc_id"] = doc_id

        embeddings = embed_voyage([c["text"] for c in chunks])

        pickle.dump(chunks, open(chunks_path, "wb"))
        np.save(embeddings_path, embeddings)

    return chunks, embeddings, doc_id


def get_last_n_tokens(text: str, n: int) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) <= n:
        return text
    return tokenizer.decode(tokens[-n:])


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    stopwords = set(
        [
            "the",
            "this",
            "and",
            "that",
            "with",
            "from",
            "shall",
            "will",
            "for",
            "are",
            "have",
            "has",
            "any",
            "you",
            "not",
            "such",
            "may",
            "each",
            "more",
            "been",
            "can",
            "who",
            "whose",
            "than",
            "per",
            "being",
            "must",
            "under",
            "also",
            "all",
            "these",
            "shall",
            "is",
            "was",
            "were",
        ]
    )

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    filtered = [word for word in words if word not in stopwords]

    common = Counter(filtered).most_common(top_k)
    return [word for word, _ in common]


def extract_universal_keywords(text: str, top_k: int = 12) -> List[str]:
    stopwords = {
        "the",
        "this",
        "that",
        "with",
        "from",
        "shall",
        "will",
        "for",
        "are",
        "have",
        "has",
        "any",
        "you",
        "not",
        "such",
        "may",
        "each",
        "more",
        "been",
        "can",
        "who",
        "whose",
        "than",
        "per",
        "being",
        "must",
        "under",
        "also",
        "all",
        "these",
        "those",
        "is",
        "was",
        "were",
        "be",
        "by",
        "at",
        "an",
        "as",
        "if",
        "or",
        "but",
        "in",
        "on",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
    }

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    numbers = re.findall(r"\b\d+\b", text)
    special_terms = re.findall(r"\b(?:section|article|chapter|clause|part|step)\s+\w+\b", text.lower())

    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    word_freq = Counter(filtered_words)

    important_patterns = [
        r"\b(?:constitution|law|act|rule|regulation)\b",
        r"\b(?:section|article|chapter|clause)\s+\d+\b",
        r"\b(?:part|component|system|procedure)\b",
        r"\b(?:theorem|principle|equation|formula)\b",
        r"\b(?:policy|coverage|premium|claim)\b",
    ]

    for pattern in important_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            word_freq[match] = word_freq.get(match, 0) + 3

    top_words = [word for word, _ in word_freq.most_common(top_k)]
    significant_numbers = [num for num in set(numbers) if numbers.count(num) >= 2]
    top_words.extend(significant_numbers[:3])

    return top_words[:top_k]


def load_or_create_chunks(
    file_url: str,
) -> Tuple[List[Dict], np.ndarray, str, str]:
    logger.info("legacy_log", message="🌍 [Universal RAG] Starting document processing")
    start_total = os.time() if hasattr(os, "time") else __import__("time").time()

    doc_id = get_doc_id(file_url)
    local_path, file_ext = download_file(file_url, doc_id)

    if file_ext == "pdf":
        raw_text = extract_text_from_pdf(local_path)
    elif file_ext == "docx":
        raw_text = extract_text_from_docx(local_path)
    elif file_ext == "eml":
        raw_text = extract_clean_text_from_eml(local_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    logger.info("legacy_log", message=f"`✅ Text extracted: {len(raw_text)} characters`")
    cleaned_text = clean_text(raw_text)

    doc_type = DocumentClassifier.classify_document(cleaned_text[:5000])
    logger.info("legacy_log", message=f"🎯 Document type detected: {doc_type.upper()}")

    chunks_path = CACHE_DIR / f"{doc_id}_chunks.pkl"
    embeddings_path = CACHE_DIR / f"{doc_id}_embeddings.npy"

    if chunks_path.exists() and embeddings_path.exists():
        logger.info(
            "legacy_log",
            message=f"✅ Using cached chunks and embeddings for {file_url}",
        )
        chunks = pickle.load(open(chunks_path, "rb"))
        embeddings = np.load(embeddings_path)
    else:
        logger.info(
            "legacy_log",
            message=f"⚡ Creating new chunks and embeddings for {file_url}",
        )
        sections = extract_structured_sections(cleaned_text, doc_type)
        for section in sections:
            section["doc_id"] = doc_id
        logger.info("legacy_log", message=f"📄 Sections extracted: {len(sections)}")
        chunks = universal_hybrid_chunking(sections, max_tokens=600, overlap_tokens=150, doc_type=doc_type)

        for c in chunks:
            c["doc_id"] = doc_id
            if "keywords" not in c:
                c["keywords"] = extract_universal_keywords(c.get("raw_text", c["text"]))

        logger.info("legacy_log", message=f"🧩 Chunks created: {len(chunks)}")
        embeddings = embed_voyage([c["text"] for c in chunks])
        pickle.dump(chunks, open(chunks_path, "wb"))
        np.save(embeddings_path, embeddings)

    duration = (os.time() if hasattr(os, "time") else __import__("time").time()) - start_total
    logger.info(
        "legacy_log",
        message=f"⏱️ Total processing time: {duration:.2f} seconds",
    )
    return chunks, embeddings, doc_id, doc_type
