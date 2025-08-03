from dotenv import load_dotenv
import os
load_dotenv()
import requests
import pdfplumber
import re
import unicodedata
import spacy
import tiktoken
import requests
import fitz 
# _____________________________________________________________________________
#                        CATCHING CODE
# Step 1: Setup and Caching Utilities

# Add these imports at the top of rag_pipeline.py
import pickle # For saving/loading python objects (like our list of chunks)
import hashlib # For creating a unique ID from the document URL
from pathlib import Path # For modern file path handling
import numpy as np # Already imported, but ensure it's there for saving embeddings
from typing import List
# --- Caching Setup ---
"""
CMT ADDED BY ME
Basically we are making an directory with name cache via using Path("")
In second line we are saying, make directory with this name cache if already made fine , dont show any errors
"""
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_doc_id(url: str) -> str:
    """
    Creates a unique and safe filename hash from a document URL.
    This helps us create a unique ID for caching related to this specific document.
    Example: "http://example.com/doc.pdf" -> "f3a2b1c..."
    """
    # Create a SHA256 hash of the URL and return its hex digest.
    return hashlib.sha256(url.encode()).hexdigest()
# _____________________________________________________________________________



from dotenv import load_dotenv
import os

load_dotenv()  # ✅ Loads .env variables into os.environ

"""# Step 1 - Data PreProcessing and Chunking"""


# Initialize spaCy for sentence splitting (optional)
nlp = spacy.load("en_core_web_sm")

# Initialize tokenizer for accurate token counts
tokenizer = tiktoken.get_encoding("cl100k_base")

import os
import requests
from urllib.parse import urlparse

def download_file(url: str, default_filename: str = "document") -> str:
    """
    Download a file (PDF, DOCX, or EML) from a URL using streaming and save it locally with the correct extension.

    Args:
        url (str): Publicly accessible URL to the file.
        default_filename (str): Default name for the saved file (without extension).

    Returns:
        str: Local file path where the file is saved.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Try to get filename from content-disposition header
    content_disposition = response.headers.get("content-disposition", "")
    parsed = urlparse(url)
    file_ext = os.path.splitext(parsed.path)[1] or ".bin"


    # Final local file name with extension
    local_path = f"{default_filename}{file_ext}"

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"✅ Downloaded file to {local_path}")
    return local_path, file_ext.lstrip(".").lower()


# 4. Extract raw text from PDF using PyMuPDF
def extract_text_from_pdf(path: str) -> str:
    """
    Extracts text from each page using PyMuPDF.
    Tags each page number for reference.
    """
    text_pages = []
    doc = fitz.open(path)
    for i, page in enumerate(doc, start=1):
        page_text = page.get_text("text") or ""
        text_pages.append(f"\n\n=== Page {i} ===\n\n{page_text}")
    doc.close()
    return "\n".join(text_pages)


from docx import Document
import extract_msg
from bs4 import BeautifulSoup

import quopri
import mailparser

import email

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_clean_text_from_eml(file_path):
    with open(file_path, "rb") as f:
        msg = email.message_from_binary_file(f)

    text = ""

    for part in msg.walk():
        content_type = part.get_content_type()
        content_disposition = str(part.get("Content-Disposition"))

        # Ignore attachments
        if "attachment" in content_disposition:
            continue

        # Extract plain text if available
        if content_type == "text/plain":
            charset = part.get_content_charset() or "utf-8"
            try:
                text = part.get_payload(decode=True).decode(charset)
                if text.strip():
                    break  # Prefer plain text if available
            except Exception:
                continue

        # Fallback to extracting from HTML
        elif content_type == "text/html":
            charset = part.get_content_charset() or "utf-8"
            try:
                html = part.get_payload(decode=True).decode(charset)
                soup = BeautifulSoup(html, "html.parser")
                clean_text = soup.get_text(separator="\n")
                if clean_text.strip():
                    text = clean_text
            except Exception:
                continue

    return text.strip()

#NEW Version of Text Cleaning
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

#This would help to get numbered Headings to act like clause Retrival

def tag_numbered_headings(text: str) -> str:
    """
    Tag numbered headings that simulate clause structure:
    e.g., "1. Introduction", "2.3.1 Waiting Period"
    """
    # Match at line starts, optional whitespace, followed by numbered pattern
    pattern = r"(?m)^\s*(\d+(\.\d+)*)([\s:–-]+)([A-Z][^\n]{3,100})"
    return re.sub(pattern, r"\n\n\1\3\4", text)

def extract_clauses_flexible(text: str):
    """
    Detect clauses like '3.1.14 Maternity' or '2 AYUSH' regardless of line breaks.
    Returns list of {clause_id, clause_title, clause_body}
    """
    import re
    matches = list(re.finditer(r"\b(\d{1,2}(?:\.\d{1,2}){0,2})[\s\-:]+([A-Z][^\d]{3,100}?)\b(?=\s+\d|\s+[A-Z])", text))
    clauses = []

    for i in range(len(matches)):
        clause_id = matches[i].group(1).strip()
        clause_title = matches[i].group(2).strip()
        start = matches[i].start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        clause_body = text[start:end].strip()

        clauses.append({
            "clause_id": clause_id,
            "clause_title": clause_title,
            "clause_body": clause_body
        })

    return clauses




#ADVANCED DOCUMENT CLASSIFIER
class DocumentClassifier:
    """Automatically detect document type and adjust processing accordingly"""

    DOCUMENT_TYPES = {
        'legal': ['constitution', 'law', 'legal', 'article', 'section', 'chapter', 'act', 'rule', 'regulation'],
        'insurance': ['policy', 'premium', 'coverage', 'claim', 'benefit', 'maternity', 'waiting', 'grace'],
        'technical': ['specification', 'manual', 'part', 'component', 'system', 'procedure', 'operation'],
        'academic': ['chapter', 'theorem', 'principle', 'equation', 'formula', 'hypothesis', 'research'],
        'medical': ['diagnosis', 'treatment', 'patient', 'disease', 'symptom', 'medication', 'therapy'],
        'financial': ['investment', 'portfolio', 'return', 'risk', 'asset', 'liability', 'balance'],
        'general': []  # fallback
    }

    @classmethod
    def classify_document(cls, text_sample: str) -> str:
        """Classify document type based on content analysis"""
        text_lower = text_sample.lower()
        word_counts = Counter(re.findall(r'\b\w+\b', text_lower))

        type_scores = {}
        for doc_type, keywords in cls.DOCUMENT_TYPES.items():
            if doc_type == 'general':
                continue
            score = sum(word_counts.get(keyword, 0) for keyword in keywords)
            type_scores[doc_type] = score

        if not type_scores or max(type_scores.values()) < 3:
            return 'general'

        return max(type_scores, key=type_scores.get)

#ADVANCED CLAUSE EXTRACTION
import re
from typing import List, Dict
from collections import defaultdict

def extract_structured_sections(text: str, doc_type: str) -> List[Dict]:
    """
    Extract structured sections/clauses based on document type and flexible clause patterns.
    Returns list of {section_id, section_title, section_body}
    """

    patterns = {
        'legal': [
            r"(?m)^\s*(Article\s+\d+[A-Z]*[\.\s]+[^\n]{10,150})",
            r"(?m)^\s*(Section\s+\d+[A-Z]*[\.\s]+[^\n]{10,150})",
            r"(?m)^\s*(Chapter\s+\d+[A-Z]*[\.\s]+[^\n]{10,150})",
            r"(?m)^\s*(\d+[\.\)]\s+[A-Z][^\n]{10,150})"
        ],
        'insurance': [
            r"(?m)^\s*(\d+(?:\.\d+)*[\s\-:]+[A-Z][^\n]{3,150})",
            r"(?m)^\s*(SECTION\s+\d+[^\n]{10,150})",
            r"(?m)^\s*(CLAUSE\s+\d+[^\n]{10,150})"
        ],
        'technical': [
            r"(?m)^\s*(\d+(?:\.\d+)*[\s\-:]+[A-Z][^\n]{5,150})",
            r"(?m)^\s*(PART\s+[A-Z\d]+[^\n]{10,150})",
            r"(?m)^\s*(COMPONENT\s+\d+[^\n]{10,150})"
        ],
        'academic': [
            r"(?m)^\s*(Chapter\s+\d+[^\n]{10,150})",
            r"(?m)^\s*(\d+(?:\.\d+)*[\s\-:]+[A-Z][^\n]{5,150})",
            r"(?m)^\s*(THEOREM\s+\d+[^\n]{10,150})",
            r"(?m)^\s*(PRINCIPLE\s+\d+[^\n]{10,150})"
        ],
        'general': [
            r"(?m)^\s*(\d+(?:\.\d+)*[\s\-:]+[A-Z][^\n]{3,150})",
            r"(?m)^\s*([A-Z][A-Z\s]{5,50}:)",
            r"(?m)^\s*(SECTION\s+[A-Z\d]+[^\n]{5,150})"
        ]
    }

    # Also include flexible clause regex always
    flexible_clause_pattern = r"(?m)^\s*\b(\d{1,2}(?:\.\d{1,2}){0,2})[\s\-:]+([A-Z][^\n]{3,100}?)\b(?=\s+\d|\s+[A-Z]|\s*\n\n)"


    doc_patterns = patterns.get(doc_type.lower(), patterns['general']) + [flexible_clause_pattern]

    # Track all matched headers
    matches = []
    for pattern in doc_patterns:
        matches.extend(list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)))

    # Sort matches by their position in text
    matches.sort(key=lambda m: m.start())

    structured_sections = []

    for i, match in enumerate(matches):
        section_header = match.group(0).strip()
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        # Try to extract section ID and title from the header
        # Prioritize numerical/standard patterns for ID
        section_id = None
        section_title = section_header

        # Attempt to find ID in standard patterns first
        id_match = re.search(r"^\s*(\d+(\.\d+)*|Article\s+\d+|Section\s+\d+|Chapter\s+\d+|CLAUSE\s+\d+|PART\s+[A-Z\d]+)", section_header, re.IGNORECASE)
        if id_match:
            section_id = id_match.group(1).strip()
            # Extract title after the ID
            title_match = re.search(re.escape(id_match.group(0)) + r"[\s:–-]+([^\n]{3,150})", section_header, re.IGNORECASE)
            if title_match:
                 section_title = title_match.group(1).strip()
            elif len(section_header.split()) > 1:
                 section_title = ' '.join(section_header.split()[1:]).strip()

        # Fallback if no standard ID found
        if not section_id:
             section_id = f"Section_{i+1}" # Generate a unique ID

        section_body = text[start_pos:end_pos].strip()

        structured_sections.append({
            "section_id": section_id,
            "section_title": section_title,
            "section_body": section_body
        })

    # If no structured sections found, create artificial ones
    if not structured_sections:
        print("⚠️ No structured sections found. Creating artificial sections.")
        return create_artificial_sections(text)


    return structured_sections

def create_artificial_sections(text: str, section_size: int = 1000) -> List[Dict]:
    """Create artificial sections when no clear structure is found"""

    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    sections = []
    current_section = ""
    section_id = 1

    for para in paragraphs:
        if len(current_section) + len(para) > section_size and current_section:
            # Create section from accumulated paragraphs
            title = extract_section_title(current_section)
            sections.append({
                "section_id": str(section_id),
                "section_title": title,
                "section_body": current_section
            })
            section_id += 1
            current_section = para
        else:
            current_section += "\n\n" + para if current_section else para

    # Add final section
    if current_section:
        title = extract_section_title(current_section)
        sections.append({
            "section_id": str(section_id),
            "section_title": title,
            "section_body": current_section
        })

    return sections

def extract_section_title(text: str) -> str:
    """Extract a meaningful title from section text"""

    # Try to find a heading-like sentence
    sentences = text.split('\n')[:3]  # Look at first 3 lines

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and len(sentence) < 100:
            # Check if it looks like a title
            if sentence.isupper() or sentence.istitle():
                return sentence
            # Check if it's the first substantial sentence
            if len(sentence.split()) > 3 and len(sentence.split()) < 15:
                return sentence

    # Fallback: use first few words
    words = text.split()[:8]
    return ' '.join(words) + "..." if len(words) == 8 else ' '.join(words)

def universal_hybrid_chunking(
    sections: List[Dict],
    max_tokens: int = 600,  # Increased for better context
    overlap_tokens: int = 150,
    doc_type: str = 'general'
) -> List[Dict]:
    """Universal chunking that adapts to document type"""

    chunks = []

    for section in sections:
        # Ensure section has necessary keys, provide defaults if missing
        section_id = section.get("section_id", f"UnknownSection_{len(chunks)}")
        section_title = section.get("section_title", "Unknown Title")
        body = section.get("section_body", "")

        if not body.strip():
            continue # Skip empty sections

        # Adaptive chunking based on document type
        if doc_type in ['legal', 'academic']:
            # Preserve paragraph integrity for legal/academic texts
            chunks.extend(chunk_by_paragraphs(section, max_tokens, overlap_tokens))
        elif doc_type == 'technical':
            # Preserve procedural steps
            chunks.extend(chunk_by_procedures(section, max_tokens, overlap_tokens))
        else:
            # Standard token-based chunking
            chunks.extend(chunk_by_tokens(section, max_tokens, overlap_tokens))

    return chunks

def chunk_by_paragraphs(section: Dict, max_tokens: int, overlap_tokens: int) -> List[Dict]:
    """Chunk by preserving paragraph boundaries"""

    paragraphs = [p.strip() for p in section["section_body"].split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para))

        if current_tokens + para_tokens > max_tokens and current_chunk:
            # Create chunk
            chunks.append(create_chunk(
                section, current_chunk, len(chunks),
                extract_universal_keywords(current_chunk)
            ))

            # Start new chunk with overlap
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

    # Add final chunk
    if current_chunk:
        chunks.append(create_chunk(
            section, current_chunk, len(chunks),
            extract_universal_keywords(current_chunk)
        ))

    return chunks

def chunk_by_procedures(section: Dict, max_tokens: int, overlap_tokens: int) -> List[Dict]:
    """Chunk by preserving procedural steps"""

    body = section["section_body"]

    # Look for step patterns
    step_patterns = [
        r'(?m)^\s*(?:Step\s+)?\d+[\.\)]\s+[^\n]+',
        r'(?m)^\s*[a-z][\.\)]\s+[^\n]+',
        r'(?m)^\s*[•\-\*]\s+[^\n]+',
        r'(?m)^\s*(?:First|Second|Third|Finally|Next|Then)[^\n]+'
    ]

    steps = []
    for pattern in step_patterns:
        matches = re.findall(pattern, body, re.IGNORECASE)
        if len(matches) >= 2:  # If we find procedural structure
            steps = matches
            break

    if not steps:
        # Fallback to paragraph chunking
        return chunk_by_paragraphs(section, max_tokens, overlap_tokens)

    # Group steps into chunks
    chunks = []
    current_chunk = ""

    for step in steps:
        if len(tokenizer.encode(current_chunk + step)) > max_tokens and current_chunk:
            chunks.append(create_chunk(
                section, current_chunk, len(chunks),
                extract_universal_keywords(current_chunk)
            ))
            current_chunk = step
        else:
            current_chunk += "\n" + step if current_chunk else step

    if current_chunk:
        chunks.append(create_chunk(
            section, current_chunk, len(chunks),
            extract_universal_keywords(current_chunk)
        ))

    return chunks

def chunk_by_tokens(section: Dict, max_tokens: int, overlap_tokens: int) -> List[Dict]:
    """Standard token-based chunking"""

    body = section["section_body"]
    tokens = tokenizer.encode(body)
    total_tokens = len(tokens)
    chunks = []
    start = 0

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append(create_chunk(
            section, chunk_text, len(chunks),
            extract_universal_keywords(chunk_text)
        ))

        start += max_tokens - overlap_tokens

    return chunks

def create_chunk(section: Dict, text: str, chunk_index: int, keywords: List[str], doc_id: str) -> Dict:
    """Create standardized chunk object"""
    return {
        "doc_id": section.get("doc_id"),
        "text": f"{section['section_id']} {section['section_title']}\n{text}",
        "section_id": section["section_id"],
        "section_title": section["section_title"],
        "chunk_index": chunk_index,
        "keywords": keywords,
        "raw_text": text
    }

def load_or_create_cache(url: str, sections: List[Dict], doc_type: str) -> (List[Dict], np.ndarray):
    doc_id = get_doc_id(url)
    chunks_path = CACHE_DIR / f"{doc_id}_chunks.pkl"
    embeddings_path = CACHE_DIR / f"{doc_id}_embeddings.npy"

    if chunks_path.exists() and embeddings_path.exists():
        print(f"✅ Using cached chunks and embeddings for {url}")
        chunks = pickle.load(open(chunks_path, "rb"))
        embeddings = np.load(embeddings_path)
    else:
        print(f"⚡ Creating new chunks and embeddings for {url}")
        chunks = universal_hybrid_chunking(sections, doc_type=doc_type)
        # Add doc_id to all chunks
        for c in chunks:
            c["doc_id"] = doc_id

        embeddings = embed_voyage([c["text"] for c in chunks])

        # Save cache
        pickle.dump(chunks, open(chunks_path, "wb"))
        np.save(embeddings_path, embeddings)

    return chunks, embeddings, doc_id



def get_last_n_tokens(text: str, n: int) -> str:
    """Get last n tokens from text"""
    tokens = tokenizer.encode(text)
    if len(tokens) <= n:
        return text
    return tokenizer.decode(tokens[-n:])

import re
from collections import Counter
from typing import List

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Naive keyword extractor: returns top_k most frequent meaningful words in text.
    Stopwords are filtered manually for now.
    """
    stopwords = set([
        'the', 'this', 'and', 'that', 'with', 'from', 'shall', 'will',
        'for', 'are', 'have', 'has', 'any', 'you', 'not', 'such', 'may',
        'each', 'more', 'been', 'can', 'who', 'whose', 'than', 'per',
        'being', 'must', 'under', 'also', 'all', 'these', 'shall', 'is', 'was'
    ])

    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())  # words with ≥4 letters
    filtered = [word for word in words if word not in stopwords]

    common = Counter(filtered).most_common(top_k)
    return [word for word, _ in common]

#ADVANCED KEYWORD EXTRACTION
def extract_universal_keywords(text: str, top_k: int = 12) -> List[str]:
    """Extract keywords that work for any document type"""

    # Universal stopwords (expanded)
    stopwords = {
        'the', 'this', 'that', 'with', 'from', 'shall', 'will', 'for', 'are', 'have',
        'has', 'any', 'you', 'not', 'such', 'may', 'each', 'more', 'been', 'can',
        'who', 'whose', 'than', 'per', 'being', 'must', 'under', 'also', 'all',
        'these', 'those', 'is', 'was', 'were', 'be', 'by', 'at', 'an', 'as', 'if',
        'or', 'but', 'in', 'on', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how'
    }

    # Extract meaningful terms
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Extract numbers and special terms
    numbers = re.findall(r'\b\d+\b', text)
    special_terms = re.findall(r'\b(?:section|article|chapter|clause|part|step)\s+\w+\b', text.lower())

    # Filter and count
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]

    # Combine different types of keywords
    word_freq = Counter(filtered_words)

    # Add special importance to certain patterns
    important_patterns = [
        r'\b(?:constitution|law|act|rule|regulation)\b',
        r'\b(?:section|article|chapter|clause)\s+\d+\b',
        r'\b(?:part|component|system|procedure)\b',
        r'\b(?:theorem|principle|equation|formula)\b',
        r'\b(?:policy|coverage|premium|claim)\b'
    ]

    for pattern in important_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            word_freq[match] = word_freq.get(match, 0) + 3  # Boost important terms

    # Get top keywords
    top_words = [word for word, _ in word_freq.most_common(top_k)]

    # Add numbers as keywords if they appear significant
    significant_numbers = [num for num in set(numbers) if numbers.count(num) >= 2]
    top_words.extend(significant_numbers[:3])

    return top_words[:top_k]



"""
# Step 2 - Embedding System
"""

#All Imports
import os
import threading
from functools import lru_cache
from typing import List, Dict, Set

import numpy as np
import faiss

# For OpenAI embeddings (Now Cohere HiHi)
# import openai
import voyageai


# For local sentence-transformer embeddings
# from sentence_transformers import SentenceTransformer

# For dimensionality reduction
from sklearn.decomposition import PCA

# For query parsing (spaCy)
import spacy

# Initialize spaCy for keyword extraction
nlp = spacy.load("en_core_web_sm")

@lru_cache()
def get_voyage_client():
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
    return voyageai.Client(api_key=VOYAGE_API_KEY)

# ============================================
# 1. Embedding Functions
# ============================================
import numpy as np
from typing import List

def embed_voyage(texts: List[str], model: str = "voyage-3.5", batch_size=300) -> np.ndarray:
    """
    Embed texts using Voyage AI embeddings API, processing them in batches
    to avoid exceeding API limits.
    Returns: (N, D) array of embeddings.
    """
    vo = get_voyage_client()
    all_embeddings = []

    # Loop through the texts in chunks of 'batch_size'
    for i in range(0, len(texts), batch_size):
        # Create a batch of texts to process
        batch = texts[i:i + batch_size]
        
        # Get embeddings for the current batch
        response = vo.embed(texts=batch, model=model)
        
        # Add the new embeddings to our list
        all_embeddings.extend(response.embeddings)
    
    # Convert the list of embeddings to a single NumPy array
    return np.array(all_embeddings, dtype='float32')

# ============================================
# 2. Dimensionality Reduction & Quantization
# ============================================
from typing import Tuple
def reduce_dimensions(
    vectors: np.ndarray,
    target_dim: int = 512
) -> Tuple[np.ndarray, PCA]:
    """
    Fit PCA on `vectors`, auto-capping to <= min(n_samples, n_features),
    and making dims divisible by 8 for FAISS PQ. Returns (reduced_vectors, pca_model).
    """
    n_samples, n_features = vectors.shape
    max_dim = min(n_samples, n_features)

    # Make target_dim <= max_dim and divisible by 8
    td = min(target_dim, max_dim)
    td -= (td % 8)  # e.g. 514 → 512

    pca = PCA(n_components=td)
    reduced = pca.fit_transform(vectors)
    print(f"[PCA] d={n_features} → {td}, retained {sum(pca.explained_variance_ratio_)*100:.2f}% variance.")
    return reduced.astype("float32"), pca
# ------------------------------------------------------------------------------
# 3. FAISS Index Builder
# ------------------------------------------------------------------------------

import faiss

def build_faiss_index(vectors: np.ndarray, use_pq: bool = False):
    """
    Build FAISS index:
      - FlatL2 if use_pq=False
      - IVF+PQ (quantized) if use_pq=True (requires d % m == 0)

    Args:
        vectors: (N, D) numpy array
        use_pq: True to use IVF+PQ (faster, quantized)

    Returns:
        FAISS index object
    """
    d = vectors.shape[1]

    if use_pq:
        # IVF+PQ: choose m (subquantizers), d must be divisible by m
        nlist, m, nbits = int(np.sqrt(len(vectors))), 8, 8
        assert d % m == 0, f"Embedding dimension {d} must be divisible by m={m} for PQ"
        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        idx.train(vectors)
        idx.add(vectors)
        idx.nprobe = 1  # controls search breadth
    else:
        idx = faiss.IndexFlatL2(d)
        idx.add(vectors)

    return idx

# ------------------------------------------------------------------------------
# 4. Keyword Thing
# ------------------------------------------------------------------------------

def extract_query_keywords(query: str, top_k=5) -> List[str]:
    doc = nlp(query)
    candidates = [tok.lemma_.lower() for tok in doc
                  if tok.pos_ in {"NOUN","PROPN","ADJ"} and len(tok)>3]
    return list(dict.fromkeys(candidates))[:top_k]

def build_inverted_index(chunks: List[Dict]) -> Dict[str, Set[int]]:
    inv = {}
    for i, c in enumerate(chunks):
        for kw in c.get("keywords",[]):
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
        return list(range(total))  # fallback to all

    cand = set.intersection(*matched_sets)
    if len(cand) < 10:
        cand = set.union(*matched_sets)  # broaden scope

    if len(cand) < 10 or len(cand) > 500:
        cand = set(range(total))  # fallback again

    if verbose:
        print(f"[Keyword Filter] Query: '{query}' → Keywords: {kws}")
        print(f"[Keyword Filter] Candidates: {len(cand)} chunks")

    return list(cand)

# ============================================
# 5. UNIVERSAL QUERY EXPANSION
# ============================================

def universal_query_expansion(query: str, doc_type: str) -> str:
    """Expand queries based on document type and universal patterns"""

    # Universal expansions that work across domains
    universal_expansions = {
        'what is': 'definition explanation meaning description',
        'how': 'procedure method process steps way',
        'when': 'time period duration date timeline',
        'where': 'location place position section part',
        'why': 'reason purpose cause explanation',
        'which': 'specification type kind category',
        'who': 'person responsible authority entity',
    }

    # Domain-specific expansions
    domain_expansions = {
        'legal': {
            'article': 'article section clause provision rule',
            'constitution': 'constitution fundamental law basic principle',
            'right': 'right freedom liberty privilege entitlement',
            'duty': 'duty obligation responsibility requirement',
            'amendment': 'amendment modification change revision',
            'court': 'court tribunal judiciary legal authority',
            'law': 'law statute act regulation rule provision'
        },
        'insurance': {
            'premium': 'premium payment cost fee amount',
            'coverage': 'coverage benefit protection indemnity',
            'claim': 'claim settlement reimbursement payout',
            'waiting': 'waiting period time duration months',
            'exclusion': 'exclusion limitation restriction exception'
        },
        'technical': {
            'part': 'part component element piece section',
            'system': 'system mechanism apparatus device',
            'procedure': 'procedure process method operation steps',
            'specification': 'specification requirement standard parameter',
            'manual': 'manual guide instruction handbook'
        },
        'academic': {
            'theorem': 'theorem principle law rule formula',
            'principle': 'principle concept theory fundamental law',
            'equation': 'equation formula expression relation',
            'chapter': 'chapter section part topic subject',
            'research': 'research study investigation analysis'
        }
    }

    expanded = query.lower()

    # Apply universal expansions
    for term, expansion in universal_expansions.items():
        if term in expanded:
            expanded += ' ' + expansion

    # Apply domain-specific expansions
    if doc_type in domain_expansions:
        for term, expansion in domain_expansions[doc_type].items():
            if term in expanded:
                expanded += ' ' + expansion

    return expanded

# ============================================
# 6. ADVANCED MULTI-MODAL RETRIEVAL
# ============================================

def advanced_universal_retrieval(
    query_embedding: np.ndarray,
    chunks: List[Dict],
    chunk_embeddings: np.ndarray,
    inv_index: Dict,
    query: str,
    doc_type: str,
    doc_id: str,   # ✅ Added parameter
    initial_k: int = 20,
    final_k: int = 6
) -> List[Dict]:
    """Advanced retrieval system that adapts to any document type with doc isolation"""

    # Stage 1: Multi-method candidate generation
    candidates_sets = []

    # Method 1: Keyword-based candidates (filtered by doc_id)
    keyword_candidates = filter_universal_candidates(inv_index, query, chunks, doc_id)
    candidates_sets.append(set(keyword_candidates))

    # Method 2: Semantic similarity candidates (filtered by doc_id)
    semantic_candidates_all = get_semantic_candidates(query_embedding, chunk_embeddings, initial_k * 4)
    semantic_candidates = [i for i in semantic_candidates_all if chunks[i]['doc_id'] == doc_id]
    candidates_sets.append(set(semantic_candidates))

    # Method 3: Section-based candidates (filtered by doc_id)
    section_candidates_all = get_section_based_candidates(query, chunks, doc_type)
    section_candidates = [i for i in section_candidates_all if chunks[i]['doc_id'] == doc_id]
    candidates_sets.append(set(section_candidates))

    # Combine candidates using ensemble approach
    final_candidates = ensemble_candidate_selection(candidates_sets, initial_k)

    # Filter final candidates by doc_id (extra safety)
    final_candidates = [i for i in final_candidates if chunks[i]['doc_id'] == doc_id]

    # Stage 2: Advanced reranking with multiple signals
    candidate_chunks = [chunks[i] for i in final_candidates]
    scores = []

    for i, chunk in enumerate(candidate_chunks):
        score_components = calculate_universal_scores(
            query, chunk, query_embedding,
            chunk_embeddings[final_candidates[i]], doc_type
        )

        # Weighted combination optimized for universal performance
        final_score = (
            0.25 * score_components['semantic'] +
            0.20 * score_components['keyword'] +
            0.15 * score_components['section_relevance'] +
            0.15 * score_components['content_density'] +
            0.10 * score_components['position'] +
            0.10 * score_components['length'] +
            0.05 * score_components['type_specific']
        )

        scores.append((final_score, i))

    # Sort and return top-k chunks
    scores.sort(reverse=True)
    return [candidate_chunks[i] for _, i in scores[:final_k]]

from typing import List, Set, Dict
import re

def filter_universal_candidates(
    inv_index: Dict[str, Set[int]],
    query: str,
    chunks: List[Dict],
    doc_id: str
) -> List[int]:
    """
    Universal keyword filtering scoped to the current doc_id.
    1) Pre-filter to this doc’s chunk indices.
    2) Find keyword matches only among those indices.
    3) Fallback always stays within the same doc.
    """
    # 1) All indices for this document
    doc_indices = {i for i, c in enumerate(chunks) if c.get('doc_id') == doc_id}
    if not doc_indices:
        return []

    # 2) Extract search terms
    query_words   = set(re.findall(r'\b\w{3,}\b', query.lower()))
    numbers       = set(re.findall(r'\b\d+\b', query))
    special_terms = set(
        re.findall(r'\b(?:section|article|chapter|part|clause)\s+\w+\b', query.lower())
    )
    all_terms = query_words | numbers | special_terms

    # 3) Build matched sets per term, restricted to this doc
    matched_sets: List[Set[int]] = []
    for term in all_terms:
        if term in inv_index:
            hits = inv_index[term] & doc_indices
            if hits:
                matched_sets.append(hits)

    # 4) If no term matched, return first 50 chunks of this doc
    if not matched_sets:
        return list(doc_indices)[:50]

    # 5) Intersection first, else union
    if len(matched_sets) > 1:
        candidates = set.intersection(*matched_sets)
    else:
        candidates = matched_sets[0]

    if len(candidates) < 10:
        candidates = set.union(*matched_sets)

    # 6) If still too few, fall back to all doc chunks
    if len(candidates) < 5:
        candidates = doc_indices

    return list(candidates)




def get_semantic_candidates(query_emb: np.ndarray, chunk_embeddings: np.ndarray, k: int) -> List[int]:
    """Get candidates based on semantic similarity"""

    # Calculate cosine similarities
    query_norm = query_emb / np.linalg.norm(query_emb)
    chunk_norms = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

    similarities = np.dot(chunk_norms, query_norm)
    top_indices = np.argsort(similarities)[::-1][:k]

    return top_indices.tolist()

def get_section_based_candidates(query: str, chunks: List[Dict], doc_type: str) -> List[int]:
    """Get candidates from same or related sections"""

    candidates = []
    query_lower = query.lower()

    # Look for section references in query
    section_refs = re.findall(r'\b(?:section|article|chapter|part|clause)\s+(\w+)\b', query_lower)

    for i, chunk in enumerate(chunks):
        section_id = chunk.get('section_id', '')
        section_title = chunk.get('section_title', '').lower()

        # Direct section reference match
        if any(ref in section_id.lower() for ref in section_refs):
            candidates.append(i)

        # Title relevance
        if any(word in section_title for word in query_lower.split() if len(word) > 3):
            candidates.append(i)

    return candidates[:15]  # Limit section-based candidates

def ensemble_candidate_selection(candidate_sets: List[Set[int]], target_k: int) -> List[int]:
    """Ensemble method to combine different candidate selection approaches"""

    # Score each candidate by how many methods selected it
    candidate_scores = defaultdict(int)

    for i, candidate_set in enumerate(candidate_sets):
        weight = [0.4, 0.4, 0.2][i] if i < 3 else 0.1  # Weight different methods
        for candidate in candidate_set:
            candidate_scores[candidate] += weight

    # Get top candidates
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return [cand for cand, _ in sorted_candidates[:target_k]]

def calculate_universal_scores(
    query: str,
    chunk: Dict,
    query_emb: np.ndarray,
    chunk_emb: np.ndarray,
    doc_type: str
) -> Dict[str, float]:
    """Calculate comprehensive scoring for any document type"""

    chunk_text = chunk.get('raw_text', chunk['text']).lower()
    query_lower = query.lower()

    scores = {}

    # 1. Semantic similarity (cosine similarity)
    scores['semantic'] = float(np.dot(query_emb, chunk_emb) /
                              (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)))

    # 2. Keyword overlap
    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
    chunk_words = set(re.findall(r'\b\w{3,}\b', chunk_text))
    overlap = len(query_words.intersection(chunk_words))
    scores['keyword'] = overlap / len(query_words) if query_words else 0

    # 3. Section relevance
    section_title = chunk.get('section_title', '').lower()
    title_overlap = len(set(query_lower.split()).intersection(set(section_title.split())))
    scores['section_relevance'] = title_overlap / len(query_lower.split()) if query_lower.split() else 0

    # 4. Content density (important terms per unit text)
    important_terms = len(re.findall(r'\b(?:\d+|[A-Z]{2,})\b', chunk_text))
    text_length = len(chunk_text.split())
    scores['content_density'] = min(important_terms / max(text_length, 1), 1.0)

    # 5. Position score (earlier sections often more important)
    try:
        section_num = float(re.search(r'(\d+)', chunk.get('section_id', '999')).group(1))
        scores['position'] = max(0.1, 1.0 - (section_num / 50.0))
    except:
        scores['position'] = 0.5

    # 6. Length appropriateness
    text_len = len(chunk_text.split())
    scores['length'] = min(1.0, text_len / 100.0) if text_len < 300 else max(0.3, 300.0 / text_len)

    # 7. Document type specific scoring
    scores['type_specific'] = calculate_type_specific_score(query, chunk, doc_type)

    return scores

def calculate_type_specific_score(query: str, chunk: Dict, doc_type: str) -> float:
    """Calculate document type specific relevance score"""

    chunk_text = chunk.get('raw_text', chunk['text']).lower()
    query_lower = query.lower()

    type_scores = {
        'legal': calculate_legal_score,
        'insurance': calculate_insurance_score,
        'technical': calculate_technical_score,
        'academic': calculate_academic_score
    }

    if doc_type in type_scores:
        return type_scores[doc_type](query_lower, chunk_text)

    return 0.5  # Neutral score for unknown types

def calculate_legal_score(query: str, chunk_text: str) -> float:
    """Legal document specific scoring"""
    score = 0.0

    # Look for legal terms alignment
    legal_terms = ['constitution', 'article', 'fundamental', 'right', 'duty', 'amendment']
    query_legal = sum(1 for term in legal_terms if term in query)
    chunk_legal = sum(1 for term in legal_terms if term in chunk_text)

    if query_legal > 0:
        score += min(chunk_legal / query_legal, 1.0) * 0.5

    # Article/section number matching
    query_nums = set(re.findall(r'\b\d+[a-z]?\b', query))
    chunk_nums = set(re.findall(r'\b\d+[a-z]?\b', chunk_text))
    if query_nums and chunk_nums:
        score += len(query_nums.intersection(chunk_nums)) / len(query_nums) * 0.5

    return min(score, 1.0)

def calculate_insurance_score(query: str, chunk_text: str) -> float:
    """Insurance document specific scoring"""
    score = 0.0

    # Insurance specific terms
    insurance_terms = ['premium', 'coverage', 'claim', 'waiting', 'period', 'policy']
    query_terms = sum(1 for term in insurance_terms if term in query)
    chunk_terms = sum(1 for term in insurance_terms if term in chunk_text)

    if query_terms > 0:
        score += min(chunk_terms / query_terms, 1.0) * 0.6

    # Numerical matching (important for insurance)
    query_nums = re.findall(r'\b\d+\b', query)
    chunk_nums = re.findall(r'\b\d+\b', chunk_text)
    if query_nums:
        num_matches = sum(1 for num in query_nums if num in chunk_nums)
        score += (num_matches / len(query_nums)) * 0.4

    return min(score, 1.0)

def calculate_technical_score(query: str, chunk_text: str) -> float:
    """Technical document specific scoring"""
    score = 0.0

    # Technical terms
    tech_terms = ['part', 'component', 'system', 'procedure', 'specification', 'manual']
    query_terms = sum(1 for term in tech_terms if term in query)
    chunk_terms = sum(1 for term in tech_terms if term in chunk_text)

    if query_terms > 0:
        score += min(chunk_terms / query_terms, 1.0) * 0.5

    # Part numbers and codes
    query_codes = re.findall(r'\b[A-Z]{2,}\d+\b|\b\d+[A-Z]+\b', query.upper())
    chunk_codes = re.findall(r'\b[A-Z]{2,}\d+\b|\b\d+[A-Z]+\b', chunk_text.upper())
    if query_codes:
        code_matches = sum(1 for code in query_codes if code in chunk_codes)
        score += (code_matches / len(query_codes)) * 0.5

    return min(score, 1.0)

def calculate_academic_score(query: str, chunk_text: str) -> float:
    """Academic document specific scoring"""
    score = 0.0

    # Academic terms
    academic_terms = ['theorem', 'principle', 'equation', 'formula', 'chapter', 'research']
    query_terms = sum(1 for term in academic_terms if term in query)
    chunk_terms = sum(1 for term in academic_terms if term in chunk_text)

    if query_terms > 0:
        score += min(chunk_terms / query_terms, 1.0) * 0.6

    # Mathematical expressions
    query_math = len(re.findall(r'[=+\-*/∫∑∆αβγδεθλμπσφψω]', query))
    chunk_math = len(re.findall(r'[=+\-*/∫∑∆αβγδεθλμπσφψω]', chunk_text))
    if query_math > 0:
        score += min(chunk_math / query_math, 1.0) * 0.4

    return min(score, 1.0)

# ============================================
# 7. ADAPTIVE CONTEXTUAL ENHANCEMENT
# ============================================

def get_adaptive_contextual_chunks(
    selected_chunks: List[Dict],
    all_chunks: List[Dict],
    doc_type: str,
    context_strategy: str = 'adaptive'
) -> List[Dict]:
    """Get contextual chunks based on document type and content"""

    enhanced_chunks = []

    for chunk in selected_chunks:
        if context_strategy == 'adaptive':
            context_chunks = get_adaptive_context(chunk, all_chunks, doc_type)
        elif context_strategy == 'hierarchical':
            context_chunks = get_hierarchical_context(chunk, all_chunks)
        else:
            context_chunks = get_sequential_context(chunk, all_chunks)

        enhanced_chunk = combine_chunks_intelligently(chunk, context_chunks, doc_type)
        enhanced_chunks.append(enhanced_chunk)

    return enhanced_chunks

def get_hierarchical_context(chunk: Dict, all_chunks: List[Dict]) -> List[Dict]:
    """Get hierarchical context: parent and sibling sections"""

    current_section = chunk.get('section_id', '')
    context = []

    # Break into section hierarchy parts (e.g., "3.2.1" => ["3", "2", "1"])
    hierarchy_parts = current_section.split('.')
    
    # Try parent section (e.g., "3.2.1" → "3.2", then "3")
    for i in range(len(hierarchy_parts) - 1, 0, -1):
        parent_id = '.'.join(hierarchy_parts[:i])
        parent_chunk = next((c for c in all_chunks if c['section_id'] == parent_id), None)
        if parent_chunk:
            context.append(parent_chunk)
            break  # Add only closest parent

    # Try sibling sections (same level, same parent)
    if len(hierarchy_parts) >= 2:
        sibling_prefix = '.'.join(hierarchy_parts[:-1])
        for c in all_chunks:
            sid = c.get('section_id', '')
            if sid.startswith(sibling_prefix) and sid != current_section:
                context.append(c)
                if len(context) >= 3:  # Avoid too many siblings
                    break

    return context

def get_adaptive_context(chunk: Dict, all_chunks: List[Dict], doc_type: str) -> List[Dict]:
    """Get context based on document type and content analysis"""

    context_chunks = []
    section_id = chunk['section_id']

    # Document type specific context strategies
    if doc_type == 'legal':
        # For legal docs, get related articles/sections
        context_chunks.extend(get_legal_context(chunk, all_chunks))
    elif doc_type == 'academic':
        # For academic docs, get prerequisite and follow-up content
        context_chunks.extend(get_academic_context(chunk, all_chunks))
    elif doc_type == 'technical':
        # For technical docs, get procedural context
        context_chunks.extend(get_technical_context(chunk, all_chunks))
    else:
        # Generic sequential context
        context_chunks.extend(get_sequential_context(chunk, all_chunks))

    return context_chunks[:3]  # Limit context to avoid token explosion

def get_legal_context(chunk: Dict, all_chunks: List[Dict]) -> List[Dict]:
    """Get legal document specific context"""
    context = []
    section_id = chunk['section_id']

    # Look for related articles, sub-articles
    for other_chunk in all_chunks:
        other_id = other_chunk['section_id']

        # Same article family (e.g., 19, 19A, 19.1)
        if (section_id.split('.')[0] == other_id.split('.')[0] and
            other_id != section_id):
            context.append(other_chunk)

        # Referenced articles
        if f"article {section_id}" in other_chunk['text'].lower():
            context.append(other_chunk)

    return context

def get_academic_context(chunk: Dict, all_chunks: List[Dict]) -> List[Dict]:
    """Get academic document specific context"""
    context = []
    chunk_text = chunk['text'].lower()

    # Look for definitional context
    for other_chunk in all_chunks:
        other_text = other_chunk['text'].lower()

        # Definitions and prerequisites
        if ('definition' in other_text or 'defined as' in other_text):
            # Check if it defines terms used in current chunk
            key_terms = extract_academic_terms(chunk_text)
            if any(term in other_text for term in key_terms):
                context.append(other_chunk)

        # Examples and applications
        if ('example' in other_text or 'application' in other_text):
            if any(term in other_text for term in extract_academic_terms(chunk_text)):
                context.append(other_chunk)

    return context

def get_technical_context(chunk: Dict, all_chunks: List[Dict]) -> List[Dict]:
    """Get technical document specific context"""
    context = []

    # Look for prerequisite procedures
    chunk_text = chunk['text'].lower()

    for other_chunk in all_chunks:
        other_text = other_chunk['text'].lower()

        # Prerequisites
        if ('before' in other_text or 'prerequisite' in other_text or
            'first' in other_text):
            context.append(other_chunk)

        # Safety warnings and notes
        if ('warning' in other_text or 'caution' in other_text or
            'note' in other_text):
            context.append(other_chunk)

    return context

def get_sequential_context(chunk: Dict, all_chunks: List[Dict]) -> List[Dict]:
    """Get sequential context (neighboring chunks)"""
    context = []
    current_idx = None

    # Find current chunk index
    for i, c in enumerate(all_chunks):
        if c['section_id'] == chunk['section_id'] and c.get('chunk_index') == chunk.get('chunk_index'):
            current_idx = i
            break

    if current_idx is not None:
        # Get neighboring chunks
        for offset in [-1, 1]:
            neighbor_idx = current_idx + offset
            if 0 <= neighbor_idx < len(all_chunks):
                context.append(all_chunks[neighbor_idx])

    return context

def extract_academic_terms(text: str) -> List[str]:
    """Extract academic key terms for context matching"""

    # Look for capitalized terms, mathematical symbols, etc.
    terms = []

    # Capitalized terms (likely important concepts)
    terms.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))

    # Mathematical terms
    terms.extend(re.findall(r'\b(?:theorem|lemma|corollary|principle|law|equation|formula)\s+\w+\b', text.lower()))

    return list(set(terms))[:5]

def combine_chunks_intelligently(
    main_chunk: Dict,
    context_chunks: List[Dict],
    doc_type: str
) -> Dict:
    """Intelligently combine main chunk with context"""

    if not context_chunks:
        return main_chunk

    # Start with main chunk
    combined_text = f"=== MAIN CONTENT ===\n{main_chunk['text']}"

    # Add context based on relevance and type
    for i, ctx_chunk in enumerate(context_chunks):
        if doc_type == 'legal':
            combined_text += f"\n\n=== RELATED PROVISION ===\n{ctx_chunk['text']}"
        elif doc_type == 'academic':
            combined_text += f"\n\n=== SUPPORTING MATERIAL ===\n{ctx_chunk['text']}"
        elif doc_type == 'technical':
            combined_text += f"\n\n=== RELATED PROCEDURE ===\n{ctx_chunk['text']}"
        else:
            combined_text += f"\n\n=== ADDITIONAL CONTEXT ===\n{ctx_chunk['text']}"

    enhanced_chunk = main_chunk.copy()
    enhanced_chunk['text'] = combined_text
    enhanced_chunk['context_count'] = len(context_chunks)

    return enhanced_chunk

# ============================================
# 8. UNIVERSAL CONFIDENCE SCORING
# ============================================

def calculate_universal_confidence(
    query: str,
    retrieved_chunks: List[Dict],
    doc_type: str
) -> float:
    """Calculate normalized confidence score for any document type"""

    confidence_signals = []

    # 1. Keyword Overlap
    query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
    all_chunk_words = set()
    for chunk in retrieved_chunks:
        chunk_words = set(re.findall(r'\b\w{3,}\b', chunk['text'].lower()))
        all_chunk_words.update(chunk_words)

    if query_words:
        overlap_score = len(query_words.intersection(all_chunk_words)) / len(query_words)
        confidence_signals.append(min(overlap_score, 1.0))
    else:
        confidence_signals.append(0.0)

    # 2. Content Specificity
    specificity_score = 0.0
    for chunk in retrieved_chunks:
        text = chunk['text']
        specifics = len(re.findall(r'\b\d+\b|\b[A-Z]{2,}\d+\b|\b\d{4}-\d{2}-\d{2}\b', text))
        specificity_score += min(specifics / 100.0, 0.3)

    confidence_signals.append(min(specificity_score, 1.0))

    # 3. Section Title Relevance
    title_relevance = 0.0
    query_words_title = set(query.lower().split())
    for chunk in retrieved_chunks:
        title_words = set(chunk.get('section_title', '').lower().split())
        if query_words_title:
            overlap = len(title_words.intersection(query_words_title)) / len(query_words_title)
            title_relevance = max(title_relevance, overlap)

    confidence_signals.append(min(title_relevance, 1.0))

    # 4. Document Type Specific Confidence
    type_confidence = calculate_type_specific_confidence(query, retrieved_chunks, doc_type)
    confidence_signals.append(min(type_confidence, 1.0))

    # 5. Chunk Consistency
    consistency_score = calculate_chunk_consistency(retrieved_chunks)
    confidence_signals.append(min(consistency_score, 1.0))

    # ✅ Clamp the final average to [0, 1]
    avg_confidence = sum(confidence_signals) / len(confidence_signals)
    return min(max(avg_confidence, 0.0), 1.0)


def calculate_type_specific_confidence(query: str, chunks: List[Dict], doc_type: str) -> float:
    """Calculate document type specific confidence"""

    type_indicators = {
        'legal': ['article', 'section', 'constitution', 'law', 'right', 'duty'],
        'insurance': ['policy', 'premium', 'coverage', 'claim', 'benefit'],
        'technical': ['part', 'component', 'procedure', 'specification'],
        'academic': ['theorem', 'principle', 'equation', 'research', 'study']
    }

    if doc_type not in type_indicators:
        return 0.5

    indicators = type_indicators[doc_type]
    query_indicators = sum(1 for ind in indicators if ind in query.lower())

    chunk_indicators = 0
    for chunk in chunks:
        chunk_indicators += sum(1 for ind in indicators if ind in chunk['text'].lower())

    if query_indicators > 0:
        return min(chunk_indicators / (query_indicators * len(chunks)), 1.0)

    return 0.5

def calculate_chunk_consistency(chunks: List[Dict]) -> float:
    """Calculate how consistent/related the retrieved chunks are"""

    if len(chunks) < 2:
        return 1.0

    # Check section similarity
    sections = [chunk.get('section_id', '') for chunk in chunks]
    section_prefixes = [s.split('.')[0] if '.' in s else s for s in sections]

    # If chunks come from similar sections, higher consistency
    unique_prefixes = len(set(section_prefixes))
    consistency = 1.0 - (unique_prefixes - 1) / len(chunks)

    return max(consistency, 0.2)

# ------------------------------------------------------------------------------
# 5. Masked Search Subset
# ------------------------------------------------------------------------------
def search_masked_subset(qvec: np.ndarray, candidate_ids, all_vectors: np.ndarray, top_k=5):
    # ensure integer list
    cids = [int(i) for i in candidate_ids]
    if not cids:
        return np.array([]), np.array([])
    vecs_subset = all_vectors[cids]
    dim = vecs_subset.shape[1]
    tmp = faiss.IndexFlatL2(dim)
    tmp.add(vecs_subset)
    D, I = tmp.search(qvec.reshape(1,-1), top_k)
    final = [cids[i] for i in I[0]]
    return D[0], final

"""# Step 3 - LLMs Setting Up"""

from concurrent.futures import ThreadPoolExecutor

# ──────────────────────────────────────────────────────────────────────────────
# 0) Imports & LLM Client Setup (run once)
# ──────────────────────────────────────────────────────────────────────────────
import os, time, json
import google.generativeai as genai
# configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

def build_batch_prompt(
    queries: List[str],
    top_chunks: List[List[Dict]],
    snippet_len: int = 200
) -> Tuple[str,str]:
    """
    Returns (system_message, user_prompt) where:
      - system_message: instructions + JSON schema
      - user_prompt: numbered list of queries + their top chunk snippets
    """
    system = (
    "You are a helpful insurance assistant. You will be asked multiple questions "
    "related to an insurance policy. Each question will be accompanied by relevant clauses "
    "from the policy document. For each question:\n\n"
    "- Read the question carefully.\n"
    "- Use the supporting clauses provided to generate a clear, correct, and concise answer.\n"
    "- Do not make up facts not present in the clauses.\n"
    "- Your final output must ONLY be a JSON object with an 'answers' key, "
    "which contains a list of plain English answers (one for each question) in order.\n\n"
    "Example output:\n"
    "{\n"
    '  "answers": [\n'
    '    "Yes, the policy covers cataract surgery after a waiting period of 2 years.",\n'
    '    "A grace period of 30 days is provided for premium payment.",\n'
    '    "AYUSH treatment is covered up to the sum insured when taken in AYUSH hospitals."\n'
    "  ]\n"
    "}\n"
    "Do not wrap your output in markdown, triple backticks, or any code block. Return only raw JSON.\n"
)

    lines = []
    for i, q in enumerate(queries, start=1):
        lines.append(f"{i}) Question: {q}\nClauses:")
        for c in top_chunks[i-1]:
            snippet = c["text"].replace("\n"," ")[:snippet_len]
            lines.append(f"- [Clause {c['clause_id']}] {snippet}")
        lines.append("---")
    user = "\n".join(lines) + "\nRespond with JSON:"

    return system, user

def build_universal_prompt(
    queries: List[str],
    top_chunks: List[List[Dict]],
    confidence_scores: List[float],
    doc_type: str,
    snippet_len: int = 600
) -> Tuple[str, str]:
    """Build prompts that work optimally for any document type"""

    # Document type specific instructions
    type_instructions = {
        'legal': (
            "You are analyzing legal documents. Provide precise answers with exact article/section references. "
            "Include specific legal provisions, rights, duties, and procedures. "
            "When citing provisions, use exact numbering (e.g., 'Article 19(1)(a)')."
        ),
        'insurance': (
            "You are analyzing insurance policy documents. Provide detailed answers with specific terms, "
            "conditions, waiting periods, coverage limits, and exclusions. "
            "Include exact timeframes (days/months/years) and monetary amounts where applicable."
        ),
        'technical': (
            "You are analyzing technical documentation. Provide precise answers with specific part numbers, "
            "procedures, specifications, and safety requirements. "
            "Include step-by-step processes and exact technical parameters."
        ),
        'academic': (
            "You are analyzing academic/scientific content. Provide comprehensive answers with definitions, "
            "principles, formulas, and supporting evidence. "
            "Include mathematical expressions and cite specific theorems or principles."
        ),
        'general': (
            "You are analyzing document content. Provide accurate, detailed answers based on the provided text. "
            "Include specific facts, numbers, and relevant details from the source material."
        )
    }

    type_instruction = type_instructions.get(doc_type, type_instructions['general'])

    system = f"""You are an expert document analyst specializing in {doc_type} documents.

{type_instruction}

CRITICAL INSTRUCTIONS:
- You MUST respond with ONLY a valid JSON object
- NO additional text, explanations, or markdown formatting
- NO backticks, code blocks, or other formatting
- The JSON must have exactly this structure: {{"answers": ["answer1", "answer2", ...]}}
- Provide COMPLETE, DETAILED answers with SPECIFIC information
- Include exact numbers, dates, references, and conditions
- Use information ONLY from the provided document sections
- Along with Specific Information, if clause contain some necessary sounding details along.. add them as well.
- Also State Yes and No wherever recquired.
- For Questions with low confidence :- Provide Best Possible Answer

Example of correct response format:
{{"answers": ["The grace period is 30 days from the due date.", "Pre-existing diseases have a waiting period of 36 months."]}}"""

    lines = []
    for i, (query, chunks, conf) in enumerate(zip(queries, top_chunks, confidence_scores), 1):
        confidence_level = "HIGH" if conf > 0.7 else "MEDIUM" if conf > 0.5 else "LOW"

        lines.append(f"\n{i}) QUERY: {query}")
        lines.append(f"CONFIDENCE: {confidence_level}")
        lines.append("RELEVANT SECTIONS:")

        for j, chunk in enumerate(chunks, 1):
            section_info = f"Section {chunk.get('section_id', 'N/A')}: {chunk.get('section_title', 'N/A')}"
            snippet = chunk["text"][:snippet_len].replace('\n', ' ')

            lines.append(f"\n[{j}] {section_info}")
            lines.append(f"Content: {snippet}")

        lines.append("\n" + "="*50)

    user = "\n".join(lines) + "\n\nRespond with valid JSON only:"

    return system, user

def batch_llm_answer(system: str, user: str, max_output_tokens: int = 2048) -> List[str]:
    """
    Enhanced Gemini API call with robust JSON parsing and fallback mechanisms
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Combine system and user messages properly
            full_prompt = f"{system}\n\n{user}"

            response = gemini_model.generate_content(
                contents=[full_prompt],
                generation_config={
                    "temperature": 0.1,  # Low temperature for consistent JSON
                    "max_output_tokens": max_output_tokens,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )

            # Extract response content
            if not response.candidates:
                raise Exception("No response candidates from Gemini")

            content = response.candidates[0].content.parts[0].text.strip()

            # Clean and parse JSON
            parsed_answers = parse_json_response(content)

            if parsed_answers:
                return parsed_answers
            else:
                print(f"⚠️ Attempt {attempt + 1}: Failed to parse JSON, retrying...")

        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print("🔴 All attempts failed, using fallback extraction")
                return extract_answers_fallback(content if 'content' in locals() else "")

    return ["Error: Could not generate response"] * 10  # Fallback

def parse_json_response(content: str) -> List[str]:
    """
    Robust JSON parsing with multiple fallback strategies
    """
    try:
        # Strategy 1: Direct JSON parsing
        return json.loads(content)["answers"]
    except:
        pass

    try:
        # Strategy 2: Remove markdown formatting
        cleaned = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        return json.loads(cleaned)["answers"]
    except:
        pass

    try:
        # Strategy 3: Extract JSON object from text
        json_match = re.search(r'\{.*?"answers"\s*:\s*\[.*?\].*?\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)["answers"]
    except:
        pass

    try:
        # Strategy 4: Find answers array specifically
        answers_match = re.search(r'"answers"\s*:\s*\[(.*?)\]', content, re.DOTALL)
        if answers_match:
            answers_str = f'{{"answers": [{answers_match.group(1)}]}}'
            return json.loads(answers_str)["answers"]
    except:
        pass

    return None

def extract_answers_fallback(content: str) -> List[str]:
    """
    Manual extraction when JSON parsing completely fails
    """
    answers = []

    # Strategy 1: Look for quoted strings that look like answers
    quoted_text = re.findall(r'"([^"]{20,500})"', content)
    for text in quoted_text:
        if any(keyword in text.lower() for keyword in
               ['period', 'coverage', 'policy', 'days', 'months', 'years', 'benefit', 'claim']):
            answers.append(text)

    # Strategy 2: Look for numbered answers
    if not answers:
        lines = content.split('\n')
        for line in lines:
            if re.match(r'^\d+[.)]\s*', line.strip()):
                answer = re.sub(r'^\d+[.)]\s*', '', line.strip())
                if len(answer) > 10:
                    answers.append(answer)

    # Strategy 3: Look for sentences with key terms
    if not answers:
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            if (len(sentence.strip()) > 20 and
                any(keyword in sentence.lower() for keyword in
                    ['period', 'coverage', 'policy', 'waiting', 'claim', 'benefit'])):
                answers.append(sentence.strip())

    # Limit to reasonable number and clean up
    answers = answers[:10]  # Limit to expected number of queries
    answers = [ans[:500] for ans in answers]  # Limit length

    return answers if answers else ["Unable to extract answer from response"] * 5

def test_json_parsing():
    """Test function to verify JSON parsing works correctly"""
    test_responses = [
        '{"answers": ["Test answer 1", "Test answer 2"]}',
        '```json\n{"answers": ["Test answer 1", "Test answer 2"]}\n```',
        'Here is the response: {"answers": ["Test answer 1", "Test answer 2"]} Hope this helps!',
        '"answers": ["Test answer 1", "Test answer 2"]'
    ]

    for i, test in enumerate(test_responses):
        result = parse_json_response(test)
        print(f"Test {i+1}: {result}")

# Uncomment to test: test_json_parsing()

# ──────────────────────────────────────────────────────────────────────────────
# 2) Call Gemini & Parse JSON
# ──────────────────────────────────────────────────────────────────────────────
# def batch_llm_answer(system: str, user: str, max_output_tokens: int = 2048):
#     """
#     Use Gemini developer API to send prompt and get JSON response.
#     """
#     response = gemini_model.generate_content(
#         contents=[
#             {"role": "user", "parts": [system]},
#             {"role": "user", "parts": [user]},
#         ],
#         generation_config={
#             "temperature": 0.0,
#             "max_output_tokens": max_output_tokens,
#         }
#     )

#     # ✅ Get the generated text safely
#     try:
#         content = response.candidates[0].content.parts[0].text
#         return json.loads(content)["answers"]
#     except Exception as e:
#         print("⚠️ Failed to parse Gemini response:")
#         print("Raw content:", content if 'content' in locals() else "No content available")
#         raise e

# ============================================
# 1. ENHANCED QUERY PREPROCESSING
# ============================================

def enhanced_query_preprocessing(query: str, doc_type: str) -> Dict[str, str]:
    """
    Advanced query preprocessing that extracts intent and key entities
    without adding computational overhead
    """
    original_query = query.strip()

    # Extract numerical entities (crucial for insurance/legal docs)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:days?|months?|years?|%|percent|rs|rupees?)\b', query.lower())

    # Extract specific entities by type
    entities = {
        'numbers': numbers,
        'timeframes': re.findall(r'\b(?:waiting|grace)\s+period\b', query.lower()),
        'coverage': re.findall(r'\b(?:cover|coverage|benefit|claim|premium)\w*\b', query.lower()),
        'medical': re.findall(r'\b(?:maternity|surgery|treatment|hospital|medical)\w*\b', query.lower()),
        'sections': re.findall(r'\b(?:section|article|clause|part)\s+\w+\b', query.lower())
    }

    # Generate focused expansion based on entities
    expansions = []
    if entities['numbers']:
        expansions.extend(['period', 'duration', 'time', 'limit'])
    if entities['coverage']:
        expansions.extend(['eligible', 'applicable', 'conditions', 'requirements'])
    if entities['medical']:
        expansions.extend(['hospital', 'treatment', 'medical', 'healthcare'])

    # Create entity-aware query
    expanded = f"{original_query} {' '.join(expansions[:5])}"  # Limit to 5 terms

    return {
        'original': original_query,
        'expanded': expanded,
        'entities': entities,
        'intent': classify_query_intent(original_query)
    }

def classify_query_intent(query: str) -> str:
    """Fast rule-based intent classification"""
    query_lower = query.lower()

    if any(word in query_lower for word in ['what is', 'define', 'definition']):
        return 'definition'
    elif any(word in query_lower for word in ['how much', 'amount', 'cost', 'premium']):
        return 'amount'
    elif any(word in query_lower for word in ['waiting period', 'how long', 'duration']):
        return 'timeframe'
    elif any(word in query_lower for word in ['cover', 'include', 'eligible']):
        return 'coverage'
    elif any(word in query_lower for word in ['procedure', 'process', 'how to']):
        return 'procedure'
    else:
        return 'general'

# ============================================
# 2. SMART CANDIDATE FILTERING
# ============================================

def smart_candidate_filtering(
    query_processed: Dict,
    chunks: List[Dict],
    inv_index: Dict,
    max_candidates: int = 30
) -> List[int]:
    """
    Multi-stage candidate filtering that's fast but more precise
    """
    query = query_processed['original']
    entities = query_processed['entities']
    intent = query_processed['intent']

    candidate_sets = []

    # Stage 1: Entity-based filtering
    if entities['numbers']:
        number_candidates = set()
        for num_phrase in entities['numbers']:
            for chunk_idx, chunk in enumerate(chunks):
                if any(num in chunk['text'].lower() for num in num_phrase.split()):
                    number_candidates.add(chunk_idx)
        if number_candidates:
            candidate_sets.append(number_candidates)

    # Stage 2: Intent-based keyword filtering
    intent_keywords = get_intent_keywords(intent, query)
    intent_candidates = set()
    for keyword in intent_keywords:
        if keyword in inv_index:
            intent_candidates.update(inv_index[keyword])
    if intent_candidates:
        candidate_sets.append(intent_candidates)

    # Stage 3: Original keyword filtering
    original_candidates = set()
    query_words = re.findall(r'\b\w{3,}\b', query.lower())
    for word in query_words:
        if word in inv_index:
            original_candidates.update(inv_index[word])
    if original_candidates:
        candidate_sets.append(original_candidates)

    # Fusion strategy: Prioritize intersection, fall back to union
    if len(candidate_sets) > 1:
        # Try intersection first
        intersected = set.intersection(*candidate_sets)
        if len(intersected) >= 5:
            final_candidates = intersected
        else:
            # Use weighted union
            weighted_candidates = defaultdict(int)
            for i, cand_set in enumerate(candidate_sets):
                weight = [3, 2, 1][i] if i < 3 else 1  # Higher weight for entity matches
                for cand in cand_set:
                    weighted_candidates[cand] += weight

            # Sort by weight and take top candidates
            sorted_candidates = sorted(weighted_candidates.items(), key=lambda x: x[1], reverse=True)
            final_candidates = set([cand for cand, _ in sorted_candidates[:max_candidates]])
    else:
        final_candidates = candidate_sets[0] if candidate_sets else set(range(min(len(chunks), max_candidates)))

    return list(final_candidates)

def get_intent_keywords(intent: str, query: str) -> List[str]:
    """Generate intent-specific keywords for better filtering"""
    base_words = re.findall(r'\b\w{4,}\b', query.lower())

    intent_expansions = {
        'definition': ['meaning', 'defined', 'refers', 'means'],
        'amount': ['amount', 'cost', 'price', 'fee', 'charges'],
        'timeframe': ['period', 'duration', 'time', 'months', 'days', 'years'],
        'coverage': ['covered', 'includes', 'eligible', 'applicable'],
        'procedure': ['process', 'steps', 'procedure', 'method']
    }

    expanded = base_words + intent_expansions.get(intent, [])
    return expanded[:8]  # Limit to prevent over-expansion

# ============================================
# 3. CONTEXTUAL CHUNK HEADERS
# ============================================

def add_contextual_headers(chunks: List[Dict], doc_type: str) -> List[Dict]:
    """
    Add rich contextual headers to chunks for better semantic understanding
    Zero embedding overhead - just text preprocessing
    """
    enhanced_chunks = []

    for chunk in chunks:
        # Create rich header based on document type
        header = create_smart_header(chunk, doc_type)

        # Combine header with content
        enhanced_text = f"{header}\n\n{chunk['text']}"

        enhanced_chunk = chunk.copy()
        enhanced_chunk['text'] = enhanced_text
        enhanced_chunk['header'] = header
        enhanced_chunks.append(enhanced_chunk)

    return enhanced_chunks

def create_smart_header(chunk: Dict, doc_type: str) -> str:
    """Create intelligent headers that improve semantic matching"""
    section_id = chunk.get('section_id', 'Unknown')
    section_title = chunk.get('section_title', 'Untitled')

    # Extract key themes from chunk text
    text = chunk.get('raw_text', chunk['text'])
    themes = extract_chunk_themes(text, doc_type)

    # Build contextual header
    if doc_type == 'insurance':
        header = f"INSURANCE POLICY SECTION {section_id}: {section_title}"
        if themes:
            header += f" | Key Topics: {', '.join(themes)}"
    elif doc_type == 'legal':
        header = f"LEGAL DOCUMENT ARTICLE {section_id}: {section_title}"
        if themes:
            header += f" | Legal Aspects: {', '.join(themes)}"
    else:
        header = f"DOCUMENT SECTION {section_id}: {section_title}"
        if themes:
            header += f" | Main Topics: {', '.join(themes)}"

    return header

def extract_chunk_themes(text: str, doc_type: str) -> List[str]:
    """Fast theme extraction for chunk headers"""
    text_lower = text.lower()

    theme_patterns = {
        'insurance': {
            'waiting_period': r'\b(?:waiting|wait)\s+period\b',
            'coverage': r'\b(?:cover|coverage|benefit)\w*\b',
            'exclusions': r'\b(?:exclud|not cover|except)\w*\b',
            'premium': r'\b(?:premium|payment|cost)\w*\b',
            'claim': r'\b(?:claim|settlement|reimburs)\w*\b',
            'maternity': r'\bmaternity\b',
            'surgery': r'\bsurgery\b',
            'medical': r'\b(?:medical|hospital|treatment)\w*\b'
        },
        'legal': {
            'rights': r'\b(?:right|freedom|liberty)\w*\b',
            'duties': r'\b(?:duty|obligation|responsibility)\w*\b',
            'procedures': r'\b(?:procedure|process|method)\w*\b',
            'penalties': r'\b(?:penalty|punishment|fine)\w*\b',
            'amendments': r'\b(?:amend|modify|change)\w*\b'
        }
    }

    themes = []
    patterns = theme_patterns.get(doc_type, {})

    for theme, pattern in patterns.items():
        if re.search(pattern, text_lower):
            themes.append(theme.replace('_', ' '))

    return themes[:3]  # Limit to top 3 themes

# ============================================
# 4. PRECISION RERANKING (Lightweight)
# ============================================

def lightweight_reranking(
    query_processed: Dict,
    retrieved_chunks: List[Dict],
    top_k: int = 5
) -> List[Dict]:
    """
    Fast reranking using multiple lightweight signals
    No additional embeddings needed
    """
    scores = []

    for chunk in retrieved_chunks:
        score_components = calculate_lightweight_scores(query_processed, chunk)

        # Weighted combination optimized for accuracy
        final_score = (
            0.30 * score_components['exact_match'] +
            0.25 * score_components['entity_overlap'] +
            0.20 * score_components['semantic_proximity'] +
            0.15 * score_components['intent_alignment'] +
            0.10 * score_components['section_relevance']
        )

        scores.append((final_score, chunk))

    # Sort by score and return top_k
    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scores[:top_k]]

def calculate_lightweight_scores(query_processed: Dict, chunk: Dict) -> Dict[str, float]:
    """Calculate multiple scoring signals efficiently with proper bounds checking"""
    query = query_processed['original'].lower()
    entities = query_processed['entities']
    intent = query_processed['intent']
    chunk_text = chunk.get('raw_text', chunk['text']).lower()

    scores = {}

    # 1. Exact phrase matching (highest signal)
    query_phrases = extract_phrases(query)
    if query_phrases:
        exact_matches = sum(1 for phrase in query_phrases if phrase in chunk_text)
        scores['exact_match'] = min(1.0, exact_matches / len(query_phrases))
    else:
        scores['exact_match'] = 0.0

    # 2. Entity overlap
    entity_score = 0
    total_entities = 0
    for entity_type, entity_list in entities.items():
        total_entities += len(entity_list)
        for entity in entity_list:
            if entity.lower() in chunk_text:
                entity_score += 1

    if total_entities > 0:
        scores['entity_overlap'] = min(1.0, entity_score / total_entities)
    else:
        scores['entity_overlap'] = 0.0

    # 3. Semantic proximity (word distance)
    scores['semantic_proximity'] = max(0.0, min(1.0, calculate_word_proximity(query, chunk_text)))

    # 4. Intent alignment
    scores['intent_alignment'] = max(0.0, min(1.0, calculate_intent_alignment(intent, chunk_text)))

    # 5. Section relevance
    section_title = chunk.get('section_title', '').lower()
    query_words = set(query.split())
    title_words = set(section_title.split())

    if query_words:
        intersection_count = len(query_words.intersection(title_words))
        scores['section_relevance'] = min(1.0, intersection_count / len(query_words))
    else:
        scores['section_relevance'] = 0.0

    # Ensure all scores are within [0, 1] bounds
    return {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

def calculate_word_proximity(query: str, text: str) -> float:
    """Calculate how close query words appear in text with proper bounds"""
    query_words = query.split()
    text_words = text.split()

    if len(query_words) < 2:
        return 1.0 if query_words and query_words[0] in text_words else 0.0

    positions = defaultdict(list)
    for i, word in enumerate(text_words):
        if word in query_words:
            positions[word].append(i)

    # Calculate minimum distance between consecutive query words
    min_distance = float('inf')
    for i in range(len(query_words) - 1):
        word1, word2 = query_words[i], query_words[i + 1]
        if word1 in positions and word2 in positions:
            for pos1 in positions[word1]:
                for pos2 in positions[word2]:
                    distance = abs(pos2 - pos1)
                    if distance > 0:  # Avoid division by zero
                        min_distance = min(min_distance, distance)

    # Convert distance to proximity score with proper bounds
    if min_distance == float('inf'):
        return 0.0

    # Normalize by expected max distance (cap at 50 words)
    max_expected_distance = 50
    proximity = max(0.0, 1.0 - (min_distance / max_expected_distance))
    return min(1.0, proximity)

def calculate_intent_alignment(intent: str, text: str) -> float:
    """Check alignment between query intent and chunk content with bounds"""
    intent_indicators = {
        'definition': ['means', 'defined', 'refers to', 'is', 'shall mean'],
        'amount': ['amount', 'cost', 'price', 'rupees', 'rs', '%', 'percent'],
        'timeframe': ['days', 'months', 'years', 'period', 'duration', 'time'],
        'coverage': ['covered', 'includes', 'eligible', 'benefit', 'applicable'],
        'procedure': ['process', 'steps', 'procedure', 'shall', 'must', 'following']
    }

    indicators = intent_indicators.get(intent, [])
    if not indicators:
        return 0.5  # Neutral score for unknown intents

    matches = sum(1 for indicator in indicators if indicator in text.lower())

    # Calculate alignment score with proper bounds
    alignment_score = matches / len(indicators)
    return min(1.0, alignment_score)

def calculate_universal_scores(
    query: str,
    chunk: Dict,
    query_emb: np.ndarray,
    chunk_emb: np.ndarray,
    doc_type: str
) -> Dict[str, float]:
    """Calculate comprehensive scoring for any document type with bounds checking"""

    chunk_text = chunk.get('raw_text', chunk['text']).lower()
    query_lower = query.lower()

    scores = {}

    # 1. Semantic similarity (cosine similarity)
    query_norm = np.linalg.norm(query_emb)
    chunk_norm = np.linalg.norm(chunk_emb)

    if query_norm > 0 and chunk_norm > 0:
        cosine_sim = np.dot(query_emb, chunk_emb) / (query_norm * chunk_norm)
        scores['semantic'] = max(-1.0, min(1.0, float(cosine_sim)))  # Cosine can be negative
        scores['semantic'] = (scores['semantic'] + 1.0) / 2.0  # Normalize to [0, 1]
    else:
        scores['semantic'] = 0.0

    # 2. Keyword overlap
    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
    chunk_words = set(re.findall(r'\b\w{3,}\b', chunk_text))

    if query_words:
        overlap = len(query_words.intersection(chunk_words))
        scores['keyword'] = min(1.0, overlap / len(query_words))
    else:
        scores['keyword'] = 0.0

    # 3. Section relevance
    section_title = chunk.get('section_title', '').lower()
    title_words = set(section_title.split())
    query_words_title = set(query_lower.split())

    if query_words_title:
        title_overlap = len(query_words_title.intersection(title_words))
        scores['section_relevance'] = min(1.0, title_overlap / len(query_words_title))
    else:
        scores['section_relevance'] = 0.0

    # 4. Content density (important terms per unit text)
    important_terms = len(re.findall(r'\b(?:\d+|[A-Z]{2,})\b', chunk_text))
    text_length = len(chunk_text.split())

    if text_length > 0:
        density = important_terms / text_length
        scores['content_density'] = min(1.0, density * 10)  # Scale appropriately
    else:
        scores['content_density'] = 0.0

    # 5. Position score (earlier sections often more important)
    try:
        section_match = re.search(r'(\d+)', chunk.get('section_id', '999'))
        if section_match:
            section_num = float(section_match.group(1))
            scores['position'] = max(0.1, min(1.0, 1.0 - (section_num / 100.0)))
        else:
            scores['position'] = 0.5
    except:
        scores['position'] = 0.5

    # 6. Length appropriateness
    text_len = len(chunk_text.split())
    if text_len < 300:
        scores['length'] = min(1.0, text_len / 100.0)
    else:
        scores['length'] = max(0.3, min(1.0, 300.0 / text_len))

    # 7. Document type specific scoring
    scores['type_specific'] = calculate_type_specific_score(query, chunk, doc_type)

    # Ensure all scores are within valid bounds
    return {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

def calculate_type_specific_score(query: str, chunk: Dict, doc_type: str) -> float:
    """Calculate document type specific relevance score with bounds"""

    chunk_text = chunk.get('raw_text', chunk['text']).lower()
    query_lower = query.lower()

    type_scores = {
        'legal': calculate_legal_score,
        'insurance': calculate_insurance_score,
        'technical': calculate_technical_score,
        'academic': calculate_academic_score
    }

    if doc_type in type_scores:
        score = type_scores[doc_type](query_lower, chunk_text)
        return max(0.0, min(1.0, score))

    return 0.5  # Neutral score for unknown types

def calculate_legal_score(query: str, chunk_text: str) -> float:
    """Legal document specific scoring with bounds"""
    score = 0.0

    # Look for legal terms alignment
    legal_terms = ['constitution', 'article', 'fundamental', 'right', 'duty', 'amendment']
    query_legal = sum(1 for term in legal_terms if term in query)
    chunk_legal = sum(1 for term in legal_terms if term in chunk_text)

    if query_legal > 0:
        term_score = min(1.0, chunk_legal / query_legal)
        score += term_score * 0.5

    # Article/section number matching
    query_nums = set(re.findall(r'\b\d+[a-z]?\b', query))
    chunk_nums = set(re.findall(r'\b\d+[a-z]?\b', chunk_text))

    if query_nums:
        num_overlap = len(query_nums.intersection(chunk_nums))
        num_score = min(1.0, num_overlap / len(query_nums))
        score += num_score * 0.5

    return min(1.0, score)

def calculate_insurance_score(query: str, chunk_text: str) -> float:
    """Insurance document specific scoring with bounds"""
    score = 0.0

    # Insurance specific terms
    insurance_terms = ['premium', 'coverage', 'claim', 'waiting', 'period', 'policy']
    query_terms = sum(1 for term in insurance_terms if term in query)
    chunk_terms = sum(1 for term in insurance_terms if term in chunk_text)

    if query_terms > 0:
        term_score = min(1.0, chunk_terms / query_terms)
        score += term_score * 0.6

    # Numerical matching (important for insurance)
    query_nums = re.findall(r'\b\d+\b', query)
    chunk_nums = re.findall(r'\b\d+\b', chunk_text)

    if query_nums:
        num_matches = sum(1 for num in query_nums if num in chunk_nums)
        num_score = min(1.0, num_matches / len(query_nums))
        score += num_score * 0.4

    return min(1.0, score)

def calculate_technical_score(query: str, chunk_text: str) -> float:
    """Technical document specific scoring with bounds"""
    score = 0.0

    # Technical terms
    tech_terms = ['part', 'component', 'system', 'procedure', 'specification', 'manual']
    query_terms = sum(1 for term in tech_terms if term in query)
    chunk_terms = sum(1 for term in tech_terms if term in chunk_text)

    if query_terms > 0:
        term_score = min(1.0, chunk_terms / query_terms)
        score += term_score * 0.5

    # Part numbers and codes
    query_codes = re.findall(r'\b[A-Z]{2,}\d+\b|\b\d+[A-Z]+\b', query.upper())
    chunk_codes = re.findall(r'\b[A-Z]{2,}\d+\b|\b\d+[A-Z]+\b', chunk_text.upper())

    if query_codes:
        code_matches = sum(1 for code in query_codes if code in chunk_codes)
        code_score = min(1.0, code_matches / len(query_codes))
        score += code_score * 0.5

    return min(1.0, score)

def calculate_academic_score(query: str, chunk_text: str) -> float:
    """Academic document specific scoring with bounds"""
    score = 0.0

    # Academic terms
    academic_terms = ['theorem', 'principle', 'equation', 'formula', 'chapter', 'research']
    query_terms = sum(1 for term in academic_terms if term in query)
    chunk_terms = sum(1 for term in academic_terms if term in chunk_text)

    if query_terms > 0:
        term_score = min(1.0, chunk_terms / query_terms)
        score += term_score * 0.6

    # Mathematical expressions
    query_math = len(re.findall(r'[=+\-*/∫∑∆αβγδεθλμπσφψω]', query))
    chunk_math = len(re.findall(r'[=+\-*/∫∑∆αβγδεθλμπσφψω]', chunk_text))

    if query_math > 0:
        math_score = min(1.0, chunk_math / query_math)
        score += math_score * 0.4

    return min(1.0, score)

def extract_phrases(text: str) -> List[str]:
    """Extract meaningful phrases from query"""
    # Extract bigrams and trigrams
    words = text.split()
    phrases = []

    # Bigrams
    for i in range(len(words) - 1):
        phrases.append(' '.join(words[i:i+2]))

    # Trigrams
    for i in range(len(words) - 2):
        phrases.append(' '.join(words[i:i+3]))

    # Filter meaningful phrases (avoid stop word combinations)
    meaningful = []
    for phrase in phrases:
        if len(phrase) > 8 and not all(word in ['the', 'is', 'of', 'and', 'to', 'a', 'in'] for word in phrase.split()):
            meaningful.append(phrase)

    return meaningful

"""# Step 4 Execution Code"""

# Global variable to store document type
CURRENT_DOC_TYPE = 'general' 

def load_or_create_chunks(file_url: str) -> Tuple[List[Dict], np.ndarray, str, str]:
    """
    Loads cached chunks & embeddings for a document or creates them if not cached.
    Returns: chunks, embeddings, doc_id, doc_type
    """
    print("🌍 [Universal RAG] Starting document processing")
    start_total = time.time()

    # Step 1: Download and extract text
    local_path, file_ext = download_file(file_url)
    if file_ext == "pdf":
        raw_text = extract_text_from_pdf(local_path)
    elif file_ext == "docx":
        raw_text = extract_text_from_docx(local_path)
    elif file_ext == "eml":
        raw_text = extract_clean_text_from_eml(local_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    print(f"✅ Text extracted: {len(raw_text)} characters")

    # Clean text
    cleaned_text = clean_text(raw_text)

    # Detect document type
    doc_type = DocumentClassifier.classify_document(cleaned_text[:5000])
    print(f"🎯 Document type detected: {doc_type.upper()}")

    # Generate doc_id and cache paths
    doc_id = get_doc_id(file_url)
    chunks_path = CACHE_DIR / f"{doc_id}_chunks.pkl"
    embeddings_path = CACHE_DIR / f"{doc_id}_embeddings.npy"

    if chunks_path.exists() and embeddings_path.exists():
        print(f"✅ Using cached chunks and embeddings for {file_url}")
        chunks = pickle.load(open(chunks_path, "rb"))
        embeddings = np.load(embeddings_path)
    else:
        print(f"⚡ Creating new chunks and embeddings for {file_url}")
        # Extract structured sections and chunk
        sections = extract_structured_sections(cleaned_text, doc_type)
        print(f"📄 Sections extracted: {len(sections)}")

        chunks = universal_hybrid_chunking(sections, max_tokens=600, overlap_tokens=150, doc_type=doc_type)

        # Add doc_id to chunks
        for c in chunks:
            c["doc_id"] = doc_id
            if 'keywords' not in c:
                c['keywords'] = extract_universal_keywords(c.get('raw_text', c['text']))

        print(f"🧩 Chunks created: {len(chunks)}")

        embeddings = embed_voyage([c["text"] for c in chunks])
        pickle.dump(chunks, open(chunks_path, "wb"))
        np.save(embeddings_path, embeddings)

    print(f"⏱️ Total processing time: {time.time() - start_total:.2f} seconds")
    return chunks, embeddings, doc_id, doc_type


def handle_queries(queries: List[str], chunks: List[Dict], chunk_embeddings: np.ndarray, inv_index: Dict, doc_type: str, doc_id: str, top_k: int = 6) -> List[str]:
    """
    Handles multiple queries using advanced retrieval and reranking.
    Ensures doc_id isolation for accuracy.
    """
    results = []
    for i, query in enumerate(queries):
        print(f"\n🔍 Query {i+1}/{len(queries)}: {query}")

        # Embed query
        query_embedding = embed_voyage([query])[0]

        # Retrieve chunks
        retrieved_chunks = advanced_universal_retrieval(
            query_embedding, chunks, chunk_embeddings, inv_index, query, doc_type, doc_id, initial_k=20, final_k=top_k
        )

        # Debug: Show retrieved sections
        for rank, chunk in enumerate(retrieved_chunks, 1):
            print(f"   Top {rank}: {chunk['section_id']} → {chunk['text'][:80]}...")

        # Build prompt and get LLM answer
        system, user_prompt = build_universal_prompt([query], [retrieved_chunks], [1.0], doc_type)
        answer = batch_llm_answer(system, user_prompt, max_output_tokens=2000)[0]
        results.append(answer)

    return results

def advanced_universal_retrieval(
    query_embedding: np.ndarray,
    chunks: List[Dict],
    chunk_embeddings: np.ndarray,
    inv_index: Dict,
    query: str,
    doc_type: str,
    doc_id: str,
    initial_k: int = 20,
    final_k: int = 6
) -> List[Dict]:
    """Advanced retrieval system with doc isolation"""

    candidates_sets = []

    # Method 1: Keyword-based
    keyword_candidates = filter_universal_candidates(inv_index, query, chunks, doc_id)
    candidates_sets.append(set(keyword_candidates))

    # Method 2: Semantic similarity
    semantic_candidates_all = get_semantic_candidates(query_embedding, chunk_embeddings, initial_k * 4)
    semantic_candidates = [i for i in semantic_candidates_all if chunks[i]['doc_id'] == doc_id]
    candidates_sets.append(set(semantic_candidates))

    # Method 3: Section-based
    section_candidates_all = get_section_based_candidates(query, chunks, doc_type)
    section_candidates = [i for i in section_candidates_all if chunks[i]['doc_id'] == doc_id]
    candidates_sets.append(set(section_candidates))

    final_candidates = ensemble_candidate_selection(candidates_sets, initial_k)
    final_candidates = [i for i in final_candidates if chunks[i]['doc_id'] == doc_id]

    candidate_chunks = [chunks[i] for i in final_candidates]
    scores = []

    for i, chunk in enumerate(candidate_chunks):
        score_components = calculate_universal_scores(
            query, chunk, query_embedding, chunk_embeddings[final_candidates[i]], doc_type
        )

        final_score = (
            0.25 * score_components['semantic'] +
            0.20 * score_components['keyword'] +
            0.15 * score_components['section_relevance'] +
            0.15 * score_components['content_density'] +
            0.10 * score_components['position'] +
            0.10 * score_components['length'] +
            0.05 * score_components['type_specific']
        )

        scores.append((final_score, i))

    scores.sort(reverse=True)
    return [candidate_chunks[i] for _, i in scores[:final_k]]
