# RAG v3 - Gemini Embeddings + ChromaDB

**Date:** 2026-06-30
**Scope:** Replacing the mock local hash embedding shim with real `gemini-embedding-001` embeddings and the local memory index with a persistent ChromaDB vector store.

---

## Goal

- Connect the retrieval system to real 3072-dimensional `gemini-embedding-001` embeddings from the Google Generative AI API.
- Replace the monolithic pickle/NumPy cache with a persistent, queryable `ChromaDB` vector database.
- Store chunks in ChromaDB with one collection per document, using a cosine distance space.
- Populate chunk metadata with required fields: `doc_id`, `section`, `page`, `parent_id`, `chunk_id`, `is_parent`, and `token_count`.

---

## Changed Files

| File | Changes |
|---|---|
| `backend/requirements.txt` | Added `chromadb==1.5.9`. Confirmed `voyageai` is completely absent. |
| `backend/services/retrieval/embedder.py` | Replaced local hash shim with calls to `google.generativeai` using `gemini-embedding-001`. Added batching logic (max 100 per request). |
| `backend/services/retrieval/vector_store.py` | Added ChromaDB PersistentClient. Implemented collection creation per doc (`doc_{doc_id}`), HNSW configurations (`hnsw:space="cosine"`, `hnsw:construction_ef=200`, `hnsw:M=32`), chunk inserts/queries, and mapping to indexes. |
| `backend/services/ingestion/chunker.py` | Updated `create_chunk` to compute `token_count` via `tiktoken` and populate metadata. Updated `load_or_create_chunks` to store/retrieve chunks directly in ChromaDB. |
| `docs/structure.md` | Updated to reflect V3 status and ChromaDB path. |
| `docs/RAG_v3.md` | Added this V3 completion log. |

---

## Verification Findings

### 1. Manual End-to-End Test (`verify_v3_e2e.py`)
Successfully ingested `dummy.pdf` from W3C and verified the entire pipeline:
```text
Ingesting PDF: https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf
[info] file_downloaded path=.../rag_c712e96563e8...pdf extension=.pdf
[info] legacy_log message=Text extracted: 33 characters
[info] legacy_log message=Document type detected: GENERAL
[info] legacy_log message=Creating new chunks and embeddings for dummy.pdf
[info] legacy_log message=Sections extracted: 1
[info] legacy_log message=Chunks created: 1
[info] legacy_log message=Total processing time: 1.02 seconds
Ingested successfully! doc_id: c712e96563e8f8b0868ca5150548c4fde964b2a7d94061dd66473c0bc3b0d29a, doc_type: general
Number of chunks: 1
Embeddings shape: (1, 3072)
ChromaDB collection count: 1
Running query: What is this document about?
Answer received: This document appears to be a dummy PDF file, as indicated by the content '1 Dummy PDF file Dummy PDF file'.
```
This confirms that real Gemini embeddings generation, ChromaDB vector collection storage, collection-scoped retrieval, and Gemini LLM question answering are fully functional.

> [!NOTE]
> This end-to-end verification was a **plumbing check** to ensure all API connections, database persistency, and pipeline wiring are correct. It does not validate retrieval quality on realistic multi-section insurance documents; that evaluation and optimization work is scheduled for V5+.

### 2. Automated Tests
All smoke tests continue to pass successfully in unit mocking mode:
```text
tests\test_smoke.py .                                                    [100%]
============================= 1 passed in 10.74s ==============================
```

---

## Design Decisions & Known Limitations

1. **Page Tracking Limitation:**
   - **Problem:** `clean_text()` strips page markers (`=== Page N ===`) before chunking happens. Consequently, page numbers are not available during the chunking phase.
   - **Resolution:** As per user instructions, `page` is set to `-1` to serve as a clear placeholder indicating that page numbers are not parsed. We do not fabricate or guess page numbers.
   
2. **Flat Parent Placeholders:**
   - **Problem:** Parent-child chunking hierarchy is out of scope for V3 and belongs to V4.
   - **Resolution:** To avoid early complexity while matching schema target, `parent_id` is set to `None` and `is_parent` is set to `False` for all chunks. No hierarchical logic is implemented in V3.

3. **ChromaDB Collection Isolation:**
   - Uses `doc_{doc_id}` naming convention (where `doc_id` is the SHA-256 hash of the document URL).
   - Ingesting a document deletes any pre-existing collection with the same `doc_id` to prevent duplicate indexing.
