# PROJECT ANALYSIS: RAG-based Insurance Decisioning System (HackRx 6)

This report provides a Senior Software Architect's complete, brutally honest, and file-by-file analysis of the codebase.

---

# 1. Executive Summary

### What This Project Does
This project is a Retrieval-Augmented Generation (RAG) API service designed to ingest unstructured files (PDF, DOCX, EML) from public URLs, process and segment the documents, perform semantic search, and synthesize answers using LLMs for insurance policy questions.

### Overall Architecture
The application runs as a Python-based FastAPI web service ([main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py)) which interfaces with a single-file pipeline engine ([rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py)). The pipeline performs the following steps:
1. **Document Ingestion**: Downloads the document URL locally and extracts text using file-type-specific engines (`PyMuPDF` for PDF, `python-docx` for DOCX, `mailparser` for EML).
2. **Text Classification**: Classifies the document text into domains (e.g., insurance, legal, technical) to select specialized processing rules.
3. **Adaptive Chunking**: Sections are extracted and segmentized using hybrid logic.
4. **Vector Embeddings**: Voyage AI embeddings are generated and stored locally in a disk-based cache (`cache/` folder as `.npy` files) alongside raw chunks (`.pkl` files).
5. **Inverted Index Filter**: Filters out irrelevant chunks using keyword sets before performing vector similarity matches.
6. **LLM Generation**: Feeds the retrieved context to Google Gemini (specifically `gemini-2.5-flash-lite`) to produce a structured JSON response containing answers.

### Tech Stack
* **Web Framework**: FastAPI (Uvicorn server)
* **Text Extraction**: `PyMuPDF` (`fitz`), `python-docx`, `mailparser`, `BeautifulSoup` (bs4)
* **Text & Language Processing**: `spaCy` (using `en_core_web_sm` model), `tiktoken`
* **Embeddings**: Voyage AI API (`voyage-3.5`)
* **Vector Math & Matrix Ops**: NumPy, scikit-learn (PCA is imported but unused)
* **LLM Engine**: Google Generative AI API (`gemini-2.5-flash-lite`)
* **Caching**: Disk caching (NumPy and pickle files) and a node-local custom in-memory LRU cache ([main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py)).

### Current Maturity
**Pre-alpha / Experimental Prototype.**
The system contains multiple **critical blockers** that prevent the application from launching or operating on new files. The codebase displays signs of hurried development, copy-paste duplicate functions, dead code, and inconsistent implementation compared to the README.

### Biggest Strengths
1. **Multi-Format Extraction**: Built-in parsers for PDF, DOCX, and EML.
2. **Hybrid Retrieval**: Combines keyword-based inverted indices, semantic vector search, and custom document-type heuristics.
3. **Multi-Layer Caching**: Combines disk-based caches (`cache/` directory) and an in-memory LRU cache to minimize latency and API usage fees.
4. **Concurrency**: Uses `ThreadPoolExecutor` to execute multiple queries against the LLM in parallel.

### Biggest Weaknesses
1. **Startup Blocker (ImportError)**: [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py) tries to import `build_enhanced_inverted_index` from [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py) which is not defined, causing the server to crash immediately on boot.
2. **Chunking Blocker (TypeError)**: The [create_chunk](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L556-L566) function requires a fifth argument `doc_id`, but is called with only four arguments by all chunking sub-routines. Any cache-miss on a new document URL crashes the system.
3. **Retrieval Blocker (TypeError/KeyError)**: The bottom definition of [advanced_universal_retrieval](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L2614-L2673) contains a nested loop calling [calculate_universal_scores](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L2246-L2325) using `...` (Ellipsis) which crashes. Additionally, it tries to read the `final` key from a dict returned by the bottom definition of [calculate_universal_scores](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L2246-L2325) which does not compute or return the `final` key.
4. **Concurrency Race Condition**: All file downloads are saved to a static name (`document.pdf`, `document.docx`, `document.eml`) in the workspace root. Concurrent requests will overwrite files, leading to mixed data or lock issues.
5. **Drastic README Discrepancy**: The [README.md](file:///c:/Summer%2026/RAG/HackrxADG/README.md) lists Pinecone, OpenAI GPT-4, S3, and PostgreSQL, none of which exist in the actual code.

---

### Architecture & Design Scores

| Category | Score (1-10) | Explanation |
| :--- | :---: | :--- |
| **Architecture** | **3/10** | Monolithic structure with zero abstraction layers. No separated services, configuration, or utility models. Severe code duplication at the module level. |
| **Code Quality** | **2/10** | Heavy function shadowing (duplicate functions overriding each other), dangling libraries, syntax-legal but runtime-broken code (`...` argument), and major type signature mismatches. |
| **Maintainability** | **2/10** | A single 100 KB file contains all retrieval logic. Absence of automated tests, configuration files, and proper logging. |
| **Scalability** | **2/10** | Hardcoded download names block multi-user concurrency. In-memory cache is node-bound. Vector search uses in-memory NumPy loops, which will degrade with large document counts. |
| **Performance** | **5/10** | Parallel query threads are good. Custom caching saves API latency. However, loading spaCy's large pipeline at import blocks FastAPI container start times. |
| **Security** | **4/10** | Simple bearer token authentication exists, but there is no URL sanitization. Vulnerable to Server-Side Request Forgery (SSRF) and malicious file injection. |
| **Documentation** | **2/10** | The README is a placeholder representing a completely different technology stack (OpenAI, S3, Pinecone, PostgreSQL) and code structure. |
| **Folder Organization** | **3/10** | Flat codebase structure with zero separation of folders except for the local `cache/` directory. |

---

# 2. Folder Structure

```
HackrxADG/
│
├── cache/                  # Local directory for disk-cached chunks and embeddings
├── .git/                   # Git VCS metadata
├── .gitignore              # Ignored files (e.g. cache, env vars, venv)
├── Procfile                # Heroku/Render deployment process configuration
├── README.md               # Readme documentation (outdated / incorrect info)
├── main.py                 # FastAPI server, route endpoints, validation, and LRU cache
├── rag_pipeline.py         # 100 KB monolith containing parser, retrieval, and LLM orchestration
├── requirements.txt        # Third-party dependency definitions
└── start.sh                # Uvicorn startup script
```

### Folder Review

#### 1. Root Directory (`/`)
* **Purpose**: Houses configuration and code entrypoints.
* **Responsibility**: Orchestration and configuration.
* **Problems**: Code files are flat. Business logic is mixed with text parsing and third-party API clients in a single script.
* **Separation of Concerns**: Violates separation of concerns. [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py) and [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py) should be separated into subdirectories (e.g., `app/api/`, `app/core/`, `app/services/`).

#### 2. `cache/`
* **Purpose**: Local disk persistence for parsed chunk objects (`_chunks.pkl`) and vectorized NumPy files (`_embeddings.npy`).
* **Responsibility**: Reduces repeat API calls.
* **Problems**: Storage has no auto-eviction, age-based expiration, or metadata indexes. It grows infinitely. Files are named using raw hex strings (`hashlib.sha256`) which are difficult to audit.
* **Separation of Concerns**: Good utility, but file operations are hardcoded to a local path rather than configured via environment variables.

---

# 3. File-by-File Analysis

### 1. [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py)
* **Purpose**: FastAPI web server API and application controller.
* **Responsibilities**:
  * Initializes the web application, sets up CORS middleware, and GZip compression.
  * Validates request body using Pydantic models.
  * Enforces token-based authentication via headers.
  * Implements an LRU in-memory document cache.
  * Exposes routes: `/`, `/health`, `/ping`, `/cache-stats`, and `/hackrx/run`.
* **Dependencies**: `fastapi`, `pydantic`, `dotenv`, [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py).
* **Who imports this file**: None (execution entry point).
* **Complexity**: Medium. Contains clean custom LRU dictionary implementations and FastAPI dependency injections.
* **Estimated Maintainability**: 6/10. Written with clear Pydantic schemas and standard FastAPI middleware.
* **Code Smells**:
  * Inline imports (e.g., `from rag_pipeline import get_doc_id` inside route definitions).
  * Direct execution of background threads via `asyncio.get_event_loop().run_in_executor(None, ...)`.
* **Potential Bugs**:
  * **Critical Startup Crash**: Line 22 attempts to import `build_enhanced_inverted_index` from [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py). This function does not exist in [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py), preventing the application from starting.
  * **Invalid Authorization Check**: Split operation on line 128 will fail with a raw `ValueError` if the header is a single word (e.g. `"secret"`), which is caught correctly but indicates fragile string manipulation.
* **Refactoring Opportunities**:
  * Extract token verification to a separate security module.
  * Fix the invalid import of `build_enhanced_inverted_index`.
* **Missing Items**:
  * No validation on incoming URL values beyond the Pydantic `HttpUrl` type.
  * Missing structured logs; uses standard string interpolation.
* **Score**: 5/10 (Clean structure, but blocked from execution by import error).

---

### 2. [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py)
* **Purpose**: Complete RAG extraction, indexing, retrieval, reranking, and generation pipeline.
* **Responsibilities**:
  * Downloads documents and parses text based on file extensions.
  * Classifies document domain types to run tailored models.
  * Implements hybrid chunking (token and sentence based).
  * Calculates query relevance scores using keyword and semantic metrics.
  * Controls LLM content generation via Gemini.
* **Dependencies**: `requests`, `pdfplumber`, `fitz` (PyMuPDF), `docx` (python-docx), `bs4`, `mailparser`, `spacy`, `tiktoken`, `voyageai`, `google.generativeai`, `numpy`, `faiss` (unused), `sklearn` (unused).
* **Who imports this file**: [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py).
* **Complexity**: High. A 100 KB monolith containing deep regex structures, manual mathematical operations, and duplicate logic layers.
* **Estimated Maintainability**: 2/10. Extremely difficult to modify safely due to code duplication and global function overrides.
* **Code Smells**:
  * **Massive Function Shadowing**: Almost all scoring and retrieval methods are defined twice, once in the middle of the file (lines 933-1264) and once at the bottom of the file (lines 2246-2439, 2614-2673). The bottom definitions override the top ones.
  * Unused imports (`faiss`, `pdfplumber`, `PCA`).
  * Empty `except: pass` blocks.
* **Potential Bugs**:
  * **Critical Bug #1 (Nested Loop & Ellipsis)**: In the overridden [advanced_universal_retrieval](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L2614-L2673) at the bottom of the file (lines 2666-2668), a nested loop calls [calculate_universal_scores](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L2246-L2325) using `...` (Ellipsis). This will throw a `TypeError` at runtime.
  * **Critical Bug #2 (Missing Dict Key)**: The overridden `advanced_universal_retrieval` accesses `sc['final']` on line 2668. However, the bottom definition of [calculate_universal_scores](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L2246-L2325) does not compute or return the `'final'` key, causing a `KeyError` at runtime.
  * **Critical Bug #3 (Chunking Mismatch)**: In [create_chunk](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L556-L566), a fifth positional argument `doc_id` is required. However, the callers `chunk_by_paragraphs` (lines 461, 480), `chunk_by_procedures` (lines 517, 526), and `chunk_by_tokens` (line 547) only pass 4 arguments, which will raise a `TypeError` if there is a cache miss.
  * **Critical Bug #4 (Race Condition)**: [download_file](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L63-L91) saves downloaded files to a static name like `document.pdf` in the workspace root. Concurrent requests will clash, leading to file access errors and mixed document data.
  * **Missing Document Data**: The `extract_text_from_docx` function ignores all tables within DOCX files, leaving out key policy structures.
* **Refactoring Opportunities**:
  * Split this file into separate modular services: `document_downloader.py`, `text_extractor.py`, `chunker.py`, `retriever.py`, and `llm_generator.py`.
  * Clean up and delete duplicate functions.
  * Pass unique names (e.g. `doc_id` or using python `tempfile`) for downloaded documents.
* **Score**: 1/10 (Contains multiple severe runtime bugs and duplicate methods).

---

### 3. [requirements.txt](file:///c:/Summer%2026/RAG/HackrxADG/requirements.txt)
* **Purpose**: Manages python dependencies.
* **Responsibilities**: Declares the exact package versions required for deployment.
* **Dependencies**: 159 packages listed.
* **Complexity**: Low.
* **Estimated Maintainability**: 4/10. List is highly bloated with packages not explicitly referenced in the code (e.g. `supabase`, `cohere`, `langchain-core`).
* **Unused Packages**: `supabase`, `gotrue`, `postgrest`, `realtime`, `cohere`, `langchain-core`, `langchain-text-splitters`.
* **Score**: 4/10 (Bloated, containing unused dependencies).

---

### 4. [Procfile](file:///c:/Summer%2026/RAG/HackrxADG/Procfile) & [start.sh](file:///c:/Summer%2026/RAG/HackrxADG/start.sh)
* **Purpose**: Deployment boot commands.
* **Responsibilities**: Tells Uvicorn to run `main:app` on port 10000.
* **Estimated Maintainability**: 9/10. Short and standard deployment setups.
* **Score**: 8/10.

---

# 4. Architecture Analysis

### Current Architecture Model: Hybrid Layered Monolith
The architecture is a single-module monolith. It uses a basic layered design where [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py) is the API gateway controller and [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py) serves as the business layer.

```
+----------------------------------------+
|               Client                   |
+-------------------+--------------------+
                    | (POST /hackrx/run)
                    v
+-------------------+--------------------+
|               main.py                  |
|  - Bearer Token Auth                   |
|  - In-Memory LRU Cache                 |
|  - Route Mapping / Pydantic schemas    |
+-------------------+--------------------+
                    | (asyncio executor)
                    v
+-------------------+--------------------+
|          rag_pipeline.py               |
|  - download_file() (Local save)        |
|  - extract_text() & clean_text()       |
|  - Classifier / Chunker                |
|  - Index / Embeddings / Cache          |
|  - Reranker / Scoring                  |
|  - LLM client calls (Gemini)           |
+----------------------------------------+
```

### Architectural Violations
1. **Module Bloat (Single Responsibility Principle Violation)**: [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py) handles input/output operations, HTML cleaning, PDF parsing, domain classification, mathematical vector alignment, disk caching, local directory creation, and third-party API connectivity.
2. **Global Namespace Shadowing**: Functions are redefined multiple times within the same module scope, causing Python to dynamically override earlier definitions. This creates confusion and leads to bugs.
3. **Execution Safety Violations**: Syntactically valid but logically broken statements (e.g., using `...` as a placeholder inside an active loop) are left in production files.
4. **State and File Operations coupling**: Disk cache reading and writing is embedded directly in the data loaders rather than abstracting it behind a data layer.

### Coupling & Cohesion
* **Coupling**: High coupling. The API controller ([main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py)) directly runs execution loops on functions inside [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py) that rely on global mutable state variables (`CACHE_DIR`, `tokenizer`, `nlp`, `CURRENT_DOC_TYPE`).
* **Cohesion**: Low cohesion. The functions in [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py) range from low-level parsing to high-level prompt engineering, failing to group related operations into separate modules.

---

# 5. Feature Inventory

### 1. Document Ingest & Parsing
* **Purpose**: Accepts URLs and extracts text.
* **Files involved**: [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py).
* **Dependencies**: `requests`, `fitz` (PyMuPDF), `docx` (python-docx), `mailparser`, `BeautifulSoup`.
* **Current Quality**: Poor. Relies on a hardcoded local filename which causes write race conditions under concurrent loads. Ignores tables in DOCX.

### 2. Document Classification
* **Purpose**: Identifies document domains (e.g. Legal, Technical, Insurance) based on keyword frequency.
* **Files involved**: [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py).
* **Dependencies**: `collections.Counter`, `re`.
* **Current Quality**: Average. Heuristics are basic and can easily be misclassified if keywords overlap.

### 3. Adaptive Chunking
* **Purpose**: Chunks text depending on the classified document type.
* **Files involved**: [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py).
* **Dependencies**: `tiktoken`.
* **Current Quality**: Broken. The chunking subroutines call [create_chunk](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L556-L566) with 4 arguments instead of the required 5, raising a `TypeError` if a new URL needs to be chunked.

### 4. Vector Retrieval & Reranking
* **Purpose**: Fetches relevant chunks using keyword indices, Voyage embeddings, and section metadata.
* **Files involved**: [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py).
* **Dependencies**: `numpy`, `voyageai`.
* **Current Quality**: Broken. The bottom definition of [advanced_universal_retrieval](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L2614-L2673) contains a nested loop that calls [calculate_universal_scores](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L2246-L2325) using `...`, causing a `TypeError`. It also accesses a non-existent `final` dict key.

### 5. LLM Synthesis
* **Purpose**: Calls Gemini to generate answers in a clean JSON format.
* **Files involved**: [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py).
* **Dependencies**: `google-generativeai`.
* **Current Quality**: Good. Features robust JSON cleanup regex parsers and fallback extraction strategies if the LLM output is not formatted correctly.

### 6. Memory and Disk Caching
* **Purpose**: Caches vector arrays and text chunks locally.
* **Files involved**: [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py), [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py).
* **Current Quality**: Good. Disk-based caching reduces LLM cost and retrieval latency for repeated files.

---

# 6. Routing Analysis

FastAPI routes are defined in [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py):

| Route Path | Method | Auth Required | Description | Status |
| :--- | :---: | :---: | :--- | :--- |
| `/` | `GET` | No | Return generic server run metadata. | Functional |
| `/health` | `GET` | No | Return system health info. | Functional |
| `/ping` | `GET` | No | Simple heartbeat utility endpoint. | Functional |
| `/cache-stats` | `GET` | Yes (Bearer) | Returns count/keys of active memory items. | Functional |
| `/hackrx/run` | `POST` | Yes (Bearer) | Primary orchestrator for document Q&A. | **Broken** (due to missing imports and pipeline runtime bugs) |

---

# 7. API Analysis

### Endpoint: `POST /hackrx/run`
* **Method**: `POST`
* **Request Schema**:
  ```json
  {
    "documents": "https://example.com/policy.pdf",
    "questions": [
      "What is the waiting period for cataract surgery?",
      "Is dental cover included?"
    ]
  }
  ```
  *(Note: Request body uses "documents" instead of "document_url" or "document_path" as mentioned in the README).*
* **Response Schema (Expected)**:
  ```json
  {
    "answers": [
      "The waiting period for cataract surgery is 24 months.",
      "Dental cover is excluded unless caused by an accident."
    ],
    "processing_time": 4.123,
    "document_id": "8a3cf..."
  }
  ```
  *(Note: The response only outputs plain text answers, whereas the README lists complex objects with fields like `justification`, `sources`, and `confidence`)*
* **Validation**:
  * Enforces `documents` is a valid URL and `questions` contains between 1 and 50 strings.
* **Authentication**: Header verification via `verify_token` matching the `HACKATHON_BEARER_TOKEN` environment variable.
* **Performance Concerns**: If the file is not cached, embedding generation and Gemini API calls are synchronous and can easily block server processing threads if multiple users submit documents simultaneously.

---

# 8. Database Analysis

There is **no database** configured in this project.
* **Issue**: The [README.md](file:///c:/Summer%2026/RAG/HackrxADG/README.md) details metadata, audit trails, and provenance logging using a PostgreSQL database. However, this functionality is entirely absent from the actual codebase.
* **Impact**: There are no persistent audit logs, tracking capabilities, or history storage. The only persistence mechanism is the local cache files (`cache/`) stored on Render's local ephemeral filesystem, which will be lost whenever the container restarts.

---

# 9. State Management Analysis

State is managed at three levels:
1. **Memory Cache**: `document_cache` (custom LRU cache with capacity 10 in [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py)). Avoids repeating text parsing, indexing, and embedding computation for active documents.
2. **Disk Cache**: Chunks and embeddings are stored inside the `cache/` directory.
3. **Application State**: A global variable `CURRENT_DOC_TYPE` (defined in [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py)) tracks the current document class, but it is not thread-safe. Under concurrent load, multiple threads will overwrite this global variable, leading to processing errors.

---

# 10. Component Analysis

Key algorithmic structures inside [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py):

* **DocumentClassifier**:
  * *Purpose*: Identifies document domain (e.g. Legal, Insurance, Technical) using keyword lists.
  * *State*: Static method.
  * *Reusability*: Good.
* **DocumentCache** (in [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py)):
  * *Purpose*: Extends python `OrderedDict` to implement an LRU cache.
  * *Props*: `capacity=10`.
  * *Reusability*: High.
* **ThreadPoolExecutor**:
  * *Purpose*: Spawns background worker threads to parallelize user queries.
  * *Coupling*: Highly coupled with `_process_single_query`.

---

# 11. Performance Analysis

### Performance Concerns
1. **spaCy Startup Latency**:
   Running `spacy.load("en_core_web_sm")` at import level blocks the server startup.
2. **CPU-Bound In-Memory Similarity Matching**:
   Cosine similarity computation and sorting are performed manually in Python loops. While fine for short documents, it will degrade when scaling to thousands of chunks.
3. **API Rate Limiting**:
   Parallel threads sending multiple requests concurrently to the Voyage AI and Gemini APIs may trigger API rate limit errors.
4. **Local Disk Ephemeral Storage**:
   Downloading and caching files locally will fail if the server is deployed to multi-node environments, as nodes do not share local disk state.

---

# 12. Code Quality Analysis

### Major Code Smells & Bugs

#### 1. Redundant Overrides
[rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py) contains duplicates of core functions that override each other. For example:
- `calculate_universal_scores` is defined on line 1107 and then redefined on line 2246.
- `advanced_universal_retrieval` is defined on line 933 and then redefined on line 2614.
This adds noise, bloats the file size, and causes unexpected runtime behavior.

#### 2. Logic Errors and Broken Nested Loops
The bottom definition of [advanced_universal_retrieval](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L2614-L2673) contains the following code:
```python
2649:     for i, chunk in enumerate(candidate_chunks):
2650:         score_components = calculate_universal_scores(
2651:             query, chunk, query_embedding, chunk_embeddings[final_candidates[i]], doc_type
2652:         )
2653: 
2654:         # final_score = (
2655:         #     ...
2662:         # )
2663: 
2664:         # scores.append((final_score, i))
2665: 
2666:         for i, chunk in enumerate(candidate_chunks):
2667:             sc = calculate_universal_scores(...)
2668:             scores.append((sc['final'], i))
```
- Line 2667 calls `calculate_universal_scores` with the ellipsis literal `...`. This will raise a `TypeError` at runtime.
- Line 2668 accesses `sc['final']`. However, the active definition of `calculate_universal_scores` does not compute or return the `'final'` key, which will cause a `KeyError`.

#### 3. Argument Mismatches
[create_chunk](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L556-L566) requires 5 parameters:
`def create_chunk(section: Dict, text: str, chunk_index: int, keywords: List[str], doc_id: str)`
However, callers in `chunk_by_paragraphs`, `chunk_by_procedures`, and `chunk_by_tokens` only pass 4 arguments, omitting `doc_id`. This causes a `TypeError` if a new URL is chunked.

#### 4. Magic Numbers
Heuristic scores use magic weight numbers:
`final_score = 0.25 * semantic + 0.20 * keyword + 0.15 * section_relevance + ...`
These weights are hardcoded and have not been validated using an evaluation set.

---

# 13. Security Review

### Vulnerabilities
1. **Server-Side Request Forgery (SSRF)**:
   The `POST /hackrx/run` endpoint accepts any arbitrary URL under the `documents` parameter. The server downloads files directly from this URL without sanitizing the target IP or checking for internal network address ranges (e.g. `127.0.0.1` or `169.254.169.254`), making it vulnerable to SSRF attacks.
2. **Malicious Ingestion and Remote Execution**:
   The application uses third-party parsing tools (`PyMuPDF`, `python-docx`) to process user-supplied files. Maliciously crafted PDFs or Word documents exploiting parser vulnerabilities could lead to buffer overflows or remote code execution.
3. **Hardcoded Secrets**:
   The app uses `os.getenv("HACKATHON_BEARER_TOKEN")`, which is safe, but there are no fallback policies or signature checks, making token handling basic.

---

# 14. Error Handling

### Error Handling Quality: Poor
* **Import Failures**: The invalid import of `build_enhanced_inverted_index` causes a complete system crash on boot.
* **Try-Catch Blocks**: The system has minimal catch structures. For example, if PDF text extraction fails inside `load_or_create_chunks`, the error propagates all the way up to the router, returning a generic 500 error without cleaning up downloaded files.
* **Lack of User Feedback**: Errors are printed to stdout, but the client only receives a generic `"Internal server error"` response, making troubleshooting difficult.

---

# 15. Dependency Analysis

### Key Dependencies

| Dependency | Purpose | Status | Alternative |
| :--- | :--- | :--- | :--- |
| `fastapi` | Web API router and endpoint validator. | Required | None |
| `fitz` (PyMuPDF) | PDF text parser. | Required | `pdfplumber` |
| `python-docx` | Word text parser. | Required | `pypandoc` |
| `spacy` | NLP pipeline for token extraction. | Heavy | `nltk` or custom regex |
| `voyageai` | Generates vector embeddings. | Required | OpenAI or local SentenceTransformers |
| `google-generativeai`| Gemini API integration. | Required | OpenAI API / LangChain |
| `numpy` | Vector calculations. | Required | None |
| `faiss-cpu` | Vector search engine. | **Unused** | Can be removed |
| `cohere` | Embeddings provider. | **Unused** | Can be removed |
| `supabase` | Supabase SDK wrapper. | **Unused** | Can be removed |
| `langchain-core` | LLM utilities. | **Unused** | Can be removed |

---

# 16. Configuration Analysis

* **Environment Configuration**: Relies on a local `.env` file containing `HACKATHON_BEARER_TOKEN`, `VOYAGE_API_KEY`, and `GEMINI_API_KEY`.
* **Deployment Config**: [Procfile](file:///c:/Summer%2026/RAG/HackrxADG/Procfile) and [start.sh](file:///c:/Summer%2026/RAG/HackrxADG/start.sh) are set up correctly for Render or Heroku deployments.
* **Formatting & Linting**: There are no configurations for linters (Pylint, Flake8) or formatters (Black, Ruff), resulting in inconsistent code styling.
* **TypeScript & Docker**: No typescript configurations or Dockerfiles are present in this repository.

---

# 17. Testing Analysis

There are **no tests** in the repository.
* **Issue**: The [README.md](file:///c:/Summer%2026/RAG/HackrxADG/README.md) mentions that `pytest` is included and mocks the LLM and database. However, no test files exist in the directory.
* **Impact**: Code changes cannot be verified automatically, increasing the risk of introducing regressions or compile-time errors.

---

# 18. Documentation Analysis

The [README.md](file:///c:/Summer%2026/RAG/HackrxADG/README.md) is **highly inaccurate**:
* It references an architecture based on **Pinecone**, **GPT-4**, and **PostgreSQL**.
* The codebase actually implements a solution using **Voyage AI**, **Gemini**, and **local filesystem caching** with no database.
* The API response formats described in the README do not match the actual response structure returned by [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py).

---

# 19. Maintainability Assessment

### Metrics
* **Readability**: Low. Function shadowing and duplicate logic blocks in [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py) make it difficult to follow the execution flow.
* **Modularity**: Low. Business logic, downloading, parsing, and scoring are tightly coupled in a single file.
* **Extensibility**: Very Low. Modifying the prompt structure or adding new document formats requires changing the core file, increasing the risk of introducing bugs.
* **Technical Debt**: Extremely High. Multiple runtime crashes and duplicate code blocks must be resolved before the system can be deployed safely.

---

# 20. Refactoring Opportunities

### Phase 1: Quick Wins (Under 30 minutes)
1. **Fix Startup Import Error** (Difficulty: Easy):
   Modify the import in [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py) to import the existing `build_inverted_index` function instead of the missing `build_enhanced_inverted_index`.
2. **Resolve chunk_by_* Argument Mismatch** (Difficulty: Easy):
   Update the `create_chunk` signature to make `doc_id` optional or pass it correctly from the chunking helpers.
3. **Fix advanced_universal_retrieval runtime bug** (Difficulty: Easy):
   Remove the duplicate bottom definition of `advanced_universal_retrieval` in [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py) to allow the correct, working definition at the top of the file to execute.

### Phase 2: Small Refactors (1-3 hours)
1. **Prevent Concurrency Race Conditions**:
   Modify the downloader to save files using unique identifiers (e.g. `doc_id` or UUIDs) instead of static names like `document.pdf`.
2. **Clean Up Unused Dependencies**:
   Remove unused libraries (`supabase`, `cohere`, `faiss-cpu`, `langchain`) from `requirements.txt`.
3. **Implement Unit Tests**:
   Add test coverage using mock responses for Gemini and Voyage AI APIs.

### Phase 3: Medium / Large Refactors
1. **Modularize the Pipeline**:
   Split [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py) into separate service modules (e.g., parsing, chunking, retrieval, generation).
2. **Add Database Integration**:
   Implement a lightweight database (e.g. SQLite or PostgreSQL) to store audit logs and metadata as described in the README.

---

# 21. Suggested Folder Structure

A cleaner, modular folder structure that separates concerns without changing core functionality:

```
HackrxADG/
│
├── config/
│   └── settings.py          # Centralized configuration and env verification
│
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI controller definitions
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── downloader.py    # Safe concurrent file downloader
│   │   ├── parsers.py       # PDF, DOCX, EML text extraction
│   │   ├── classifier.py    # Document domain classifier
│   │   ├── chunker.py       # Text chunking logic
│   │   ├── retriever.py     # Hybrid search and scoring algorithms
│   │   └── generator.py     # Gemini API integration
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py       # Caching and helper utilities
│
├── tests/
│   ├── test_api.py          # Integration tests for API routes
│   └── test_pipeline.py     # Unit tests for parsing and retrieval
│
├── cache/                   # Local file cache
├── requirements.txt
├── Procfile
├── README.md
└── start.sh
```

---

# 22. Prioritized Improvement Roadmap

### Phase 1: Critical Stability Fixes (Effort: 2 hours)
* Resolve the missing import issue in [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py).
* Fix the parameter mismatch in [create_chunk](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L556-L566).
* Remove duplicate functions from [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py).
* Implement unique filenames for downloaded documents to prevent concurrent write errors.

### Phase 2: Refactoring & Cleanup (Effort: 4 hours)
* Separate [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py) into standalone modules under `src/services/`.
* Remove unused dependencies from `requirements.txt`.
* Standardize logging and error responses.

### Phase 3: Testing & Security (Effort: 4 hours)
* Add mock-based unit tests for the pipeline.
* Implement URL sanitization to mitigate SSRF risks.
* Set up a linter and code formatter.

### Phase 4: Documentation (Effort: 1 hour)
* Update [README.md](file:///c:/Summer%2026/RAG/HackrxADG/README.md) to accurately document the active tech stack (Voyage AI, Gemini, Local Cache).

---

# 23. Technical Debt Report

| Technical Debt Item | Severity | Impact | Recommendation | Est. Effort |
| :--- | :--- | :--- | :--- | :---: |
| **Missing Import Crash** | **Critical** | Prevents the application from starting. | Fix import target in [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py). | 5 mins |
| **create_chunk Argument Mismatch** | **Critical** | Crashes the pipeline when processing any new document. | Update the callers to pass all required arguments. | 10 mins |
| **Duplicate shadow code** | **High** | Causes developer confusion and potential runtime bugs. | Delete duplicate functions. | 15 mins |
| **Static Download Filenames** | **High** | Causes race conditions during concurrent API requests. | Generate unique names for downloads. | 20 mins |
| **Outdated README** | **Medium** | Misleads developers about the project architecture. | Update documentation to match the actual code. | 30 mins |
| **Lack of Automated Tests** | **Medium** | Increases the risk of regressions. | Add unit tests for core services. | 4 hours |
| **Unused Dependencies** | **Low** | Bloats deployment build sizes. | Clean up `requirements.txt`. | 15 mins |

---

# 24. Final Assessment

### Overall Score
* **Architecture**: 3/10
* **Code Quality**: 2/10

### Key Risks
* **Stability**: Multiple runtime bugs prevent the pipeline from executing successfully on new files.
* **Concurrency**: Concurrent requests will corrupt downloaded files due to static naming.
* **Security**: Exposed to SSRF and parser exploit vulnerabilities.

### Top 20 High-ROI Refactoring Recommendations
1. Update [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py) to import the correct `build_inverted_index` function instead of the missing `build_enhanced_inverted_index`.
2. Delete the duplicate `advanced_universal_retrieval` implementation at the end of [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py).
3. Delete the duplicate `calculate_universal_scores` implementation in [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py).
4. Delete other duplicate functions (`calculate_type_specific_score`, `calculate_legal_score`, etc.) in [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py).
5. Fix the missing `doc_id` argument in all [create_chunk](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py#L556-L566) calls inside [rag_pipeline.py](file:///c:/Summer%2026/RAG/HackrxADG/rag_pipeline.py).
6. Use a unique filename (e.g. `f"doc_{doc_id}{file_ext}"`) in `download_file` to support concurrency.
7. Change `CURRENT_DOC_TYPE` from a global variable to a function-scoped variable to ensure thread-safety.
8. Remove unused packages like `supabase`, `cohere`, and `faiss-cpu` from `requirements.txt`.
9. Wrap file extraction steps in try-except blocks to catch and log parser errors.
10. Implement basic URL validation in [main.py](file:///c:/Summer%2026/RAG/HackrxADG/main.py) to check for internal IP addresses.
11. Add a `tempfile` clean-up routine to delete downloaded files after processing.
12. Update `extract_text_from_docx` to extract text from tables.
13. Replace manual cosine similarity calculations with NumPy vectorized operations.
14. Expose the confidence score and sources in the API response as described in the README.
15. Add a simple health check validation for `VOYAGE_API_KEY` and `GEMINI_API_KEY` on startup.
16. Move configuration constants (`EXPECTED_BEARER_TOKEN`, etc.) to a centralized config module.
17. Write unit tests for document parsers.
18. Set up code formatting rules using Black or Ruff.
19. Correct the [README.md](file:///c:/Summer%2026/RAG/HackrxADG/README.md) to reflect the active technology stack.
20. Implement a structured logging helper instead of using basic print statements.
