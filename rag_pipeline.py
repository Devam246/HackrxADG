#trying commit first time
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
# Initialize spaCy for sentence splitting (optional)
nlp = spacy.load("en_core_web_sm")

# Initialize tokenizer for accurate token counts
tokenizer = tiktoken.get_encoding("cl100k_base")

#downloaded file and tells the type of file
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

# 4. Extract raw text from docx and email
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

#Acc to new Clause Extractor
def hybrid_chunking(clauses: list, max_tokens: int = 500, overlap_tokens: int = 120) -> list:
    """
    Given a list of clauses (clause_id, clause_title, clause_body),
    splits each clause into overlapping token chunks.

    Returns a list of chunks with metadata:
    [
        {
            "text": "...",
            "clause_id": "3.1.14",
            "clause_title": "Maternity",
            "start_token": 0,
            "end_token": 500
        },
        ...
    ]
    """
    chunks = []
    for clause in clauses:
        clause_id = clause["clause_id"]
        clause_title = clause["clause_title"]
        body = clause["clause_body"]

        tokens = tokenizer.encode(body)
        total_tokens = len(tokens)
        start = 0

        while start < total_tokens:
            end = min(start + max_tokens, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunk_keywords = extract_keywords(chunk_text)

            chunks.append({
                "text": f"{clause_id} {clause_title}\n{chunk_text}",
                "clause_id": clause_id,
                "clause_title": clause_title,
                "start_token": start,
                "end_token": end,
                "keywords": chunk_keywords   # 👈 new field added

            })

            start += max_tokens - overlap_tokens

    return chunks

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

# 8. (Optional) Sentence segmentation for fine-grained chunks

def sentence_chunking(text: str) -> list:
    """
    Use spaCy to split any text into individual sentences.
    Useful if you need sentence-level retrieval.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

"""All above are Functions.. No function is called in final yet

# Step 2 - Embedding System
"""

#All Imports
import os
import threading
from functools import lru_cache
from typing import List, Dict, Set

import numpy as np
import faiss

# For OpenAI embeddings (Now Cohere HiHi) (Now Voyage HAhaHA)
# import openai
import voyageai

# For local sentence-transformer embeddings

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

# def embed_cohere(texts: List[str], model="embed-english-v3.0", batch_size=300) -> np.ndarray:
#     """
#     Batch-embed texts using OpenAI embeddings API.
#     Returns: (N, D) array of embeddings.
#     """
#     embeddings = []
#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i + batch_size]
#         resp = co.embed(
#             texts=batch,
#             model=model,
#             input_type="search_document"  # use "search_query" if embedding a user query
#         )
#         embeddings.extend(resp.embeddings)

#     return np.array(embeddings, dtype='float32')
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

from functools import lru_cache

@lru_cache()
def get_gemini_model():
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash-lite")


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
    "- if you find information not mentioned then try to most relevent fact from clauses and also use your general knowlege to factify it the very best\n"
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

# ──────────────────────────────────────────────────────────────────────────────
# 2) Call Gemini & Parse JSON
# ──────────────────────────────────────────────────────────────────────────────
def batch_llm_answer(system: str, user: str, max_output_tokens: int = 2048):
    """
    Use Gemini developer API to send prompt and get JSON response.
    """
    gemini_model = get_gemini_model()
    response = gemini_model.generate_content(
        contents=[
            {"role": "user", "parts": [system]},
            {"role": "user", "parts": [user]},
        ],
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": max_output_tokens,
        }
    )

    # ✅ Get the generated text safely
    try:
        content = response.candidates[0].content.parts[0].text
        return json.loads(content)["answers"]
    except Exception as e:
        print("⚠️ Failed to parse Gemini response:")
        print("Raw content:", content if 'content' in locals() else "No content available")
        raise e


import time
import os
import mimetypes

def extract_chunks_from_any_file(file_url: str):
    """
    End-to-end clause-aware chunking pipeline for PDF, DOCX, and EML files.

    Args:
        file_url (str): Publicly accessible URL to PDF, DOCX, or EML file.

    Returns:
        List[Dict]: List of clause-aware hybrid chunks.
    """
    start_total = time.time()
    #__________________________________________________________________
    # 1. Generate a unique ID for the document to use as a cache key.
    doc_id = get_doc_id(file_url)
    chunk_cache_path = CACHE_DIR / f"chunks_{doc_id}.pkl"
    # 2. Check if the chunks are already cached.
    if chunk_cache_path.exists():
        print(f"✅ CACHE HIT for chunks. Loading from {chunk_cache_path}")
        with open(chunk_cache_path, "rb") as f:
            chunks = pickle.load(f)
        print(f"🕒 Total Time (from cache): {time.time() - start_total:.2f} seconds")
        return chunks, doc_id
    #___________________________________________________________________

    # Step 1: Download the file
    start = time.time()
    local_path, file_ext = download_file(file_url)
    print(f"✅ Step 1 - File Downloaded: {local_path} in {time.time() - start:.2f} seconds")

    # Step 2: Extract raw text based on type
    start = time.time()
    if file_ext == "pdf":
        raw_text = extract_text_from_pdf(local_path)
    elif file_ext == "docx":
        raw_text = extract_text_from_docx(local_path)
    elif file_ext == "eml":
        raw_text = extract_clean_text_from_eml(local_path)

    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    print(f"✅ Step 2 - Text Extracted in {time.time() - start:.2f} seconds")

    # Step 3: Clean and preprocess text
    start = time.time()
    cleaned_text = clean_text(raw_text)
    print(f"✅ Step 3 - Text Cleaned in {time.time() - start:.2f} seconds")


    # Step 4: Clause extraction
    start = time.time()
    clauses = extract_clauses_flexible(cleaned_text)
    print(f"✅ Step 4 - Clause Extraction in {time.time() - start:.2f} seconds")
    print(f"📄 Total clauses detected: {len(clauses)}")

    # Step 5: Hybrid Chunking
    start = time.time()
    chunks = hybrid_chunking(clauses)
    print(f"✅ Step 5 - Hybrid Chunking in {time.time() - start:.2f} seconds")
    print(f"🧩 Total chunks generated: {len(chunks)}")

    print(f"🕒 Total Time: {time.time() - start_total:.2f} seconds")
#_______________________________________________________________________________
 # 3. Save the newly generated chunks to the cache before returning.
    print(f"💾 Saving chunks to cache: {chunk_cache_path}")
    with open(chunk_cache_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"🕒 Total Time (processed and cached): {time.time() - start_total:.2f} seconds")
#________________________________________________________________________________
    return chunks, doc_id 
#(returning both chunks and doc_id ABOVE)
    #return chunks

def handle_queries(
    queries: List[str],
    chunks,
    doc_id: str, # Accept the document ID
    top_k: int = 5
) -> List[Dict]:
    """
    Full multi‑query pipeline:
      - batch embed & PCA
      - parallel keyword+FAISS retrieval
      - build batch prompt
      - single Gemini call
    Returns: list of plain answer strings
    """
    #________________________________________________________________________
    embedding_cache_path = CACHE_DIR / f"embeddings_{doc_id}.npy"
       # Check if embeddings for this document are already cached.
    if embedding_cache_path.exists():
        print(f"✅ CACHE HIT for embeddings. Loading from {embedding_cache_path}")
        embs_full = np.load(embedding_cache_path)
    else:
        print("⚠️ CACHE MISS for embeddings. Generating and saving.")
        texts = [chunk['text'] for chunk in chunks]
        # Embed the chunks using Cohere
        #embs_full = embed_cohere(texts, model="embed-english-v3.0", batch_size=300)
        embs_full = embed_voyage(texts, model="voyage-3.5", batch_size=300)

        # Save the numpy array to the cache file.
        np.save(embedding_cache_path, embs_full)
    #__________________________________________________________________________
    print("\n🟢 [handle_queries] Starting query handling")
    print(f"🔍 Total queries received: {len(queries)}")
    for i, q in enumerate(queries, start=1):
        print(f"  {i}. {q}")
    print(f"🧩 Total document chunks available: {len(chunks)}")

    timings = {}

    # 1. Embed chunks using Voyage
    #texts = [chunk['text'] for chunk in chunks]
    #t0 = time.time()
    #embs_full = embed_voyage(texts, model="embed-english-v3.0", batch_size=300)
    #timings['chunk_embedding'] = time.time() - t0
    #print(f"🧠 Embedded {len(texts)} chunks in {timings['chunk_embedding']:.3f} sec")

    # 2. Build inverted keyword index
    inv_index = build_inverted_index(chunks)

    # 3. Embed queries
    t1 = time.time()
    q_raw = embed_voyage(queries)
    q_red = q_raw  # no PCA currently
    timings['query_embedding'] = time.time() - t1
    print(f"📨 Embedded {len(queries)} queries in {timings['query_embedding']:.3f} sec")

    # 4. Parallel retrieval of top_k chunks per query
    def retrieve_one(i):
        qv = q_red[i].astype("float32")
        cands = filter_candidates(inv_index, queries[i], len(chunks))
        _, idxs = search_masked_subset(qv, cands, embs_full, top_k)
        return [chunks[j] for j in idxs]

    t2 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as exe:
        top_k_chunks = list(exe.map(retrieve_one, range(len(queries))))
    timings['retrieval'] = time.time() - t2
    print(f"🔎 Retrieved top {top_k} chunks for each query in {timings['retrieval']:.3f} sec")

    # 5. Build prompt
    system, user_prompt = build_batch_prompt(queries, top_k_chunks)
    timings['prompt_build'] = 0.0  # negligible
    print("📜 Built batch prompt for Gemini")

    # 6. LLM call (Gemini)
    t3 = time.time()
    answers = batch_llm_answer(system, user_prompt, max_output_tokens=len(queries)*200)
    timings['LLM'] = time.time() - t3
    print(f"🤖 Gemini answered all queries in {timings['LLM']:.3f} sec")

    # 7. Summary log
    print("\n📊 Final Timing Breakdown:")
    for k, v in timings.items():
        print(f"  {k:17s}: {v:.3f} sec")
    total = sum(timings.values())
    print(f"  {'TOTAL':17s}: {total:.3f} sec\n")

    return answers


"""# TESTING CODE"""

# ──────────────────────────────────────────────────────────────────────────────
# 4) Example Usage (10 queries)
# ──────────────────────────────────────────────────────────────────────────────
# file_url = " "

# queries = [
# 
# ]

# extract_chunks_from_pdf(pdf_url)
# chunks = extract_chunks_from_any_file(file_url)
# answers = handle_queries(queries,chunks, top_k=3)
# print(json.dumps(answers, indent=2))
