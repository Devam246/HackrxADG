# RAG Insurance Decisioning — README

**Project:** RAG-based Insurance Decisioning System

**Short description**
A Retrieval-Augmented Generation (RAG) system that processes unstructured documents (PDFs, Word files, emails) and answers natural-language insurance-related questions with accurate, explainable, and fast responses. Built as an API service (FastAPI) that uses vector search (Pinecone) for retrieval, GPT-4 for answer synthesis, and PostgreSQL for metadata and audit logs. The system is designed to return both an answer and a justification for each question.

---

## Table of Contents

* [Key features](#key-features)
* [Architecture](#architecture)
* [API specification](#api-specification)
* [Getting started](#getting-started)

  * [Requirements](#requirements)
  * [Environment variables](#environment-variables)
  * [Installation](#installation)
  * [Run locally](#run-locally)
* [Docker / Deployment](#docker--deployment)
* [Testing the API](#testing-the-api)
* [Evaluation & performance targets](#evaluation--performance-targets)
* [Prompt engineering & cost-control tips](#prompt-engineering--cost-control-tips)
* [Security & privacy](#security--privacy)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)
* [License](#license)

---

## Key features

* Accepts document URLs (PDF/Word/HTML) and ingests them into a searchable vector index.
* Semantic retrieval using Pinecone (or pluggable vector DB).
* LLM synthesis using GPT-4 with explicit justifications citing retrieved chunks.
* Fast responses targeted to be under **30 seconds** for the whole pipeline (retrieval + LLM inference).
* API designed for programmatic decisioning workflows (insurance claim triage, rule-checking, data extraction).
* Audit logs and provenance stored in PostgreSQL for traceability.

---

## Architecture

Simple flow:

```
[Client] -> POST /hackrx/run -> [FastAPI App]
                              |- Download & chunk document (if not already ingested)
                              |- Embed chunks (OpenAI / local embedder)
                              |- Upsert into Pinecone index
                              |- Query Pinecone for top-K relevant chunks
                              |- Call GPT-4 with a retrieval-augmented prompt
                              |- Return answers + justification and save audit in Postgres
```

Components:

* **FastAPI**: HTTP API layer, request validation, orchestration.
* **Embeddings**: OpenAI embeddings or alternative.
* **Vector DB**: Pinecone for vector storage and nearest-neighbor search.
* **LLM**: GPT-4 (via OpenAI API) for final answer + explanation.
* **DB**: PostgreSQL for metadata, logs, and audit trail.
* **Storage**: Documents can be kept in object store (S3) or referenced by URL.

---

## API specification

**Endpoint**: `POST /hackrx/run`

**Description**: Main endpoint that accepts a document URL and an array of questions. Returns an array of `answer` objects; each contains the `question`, `answer`, and `justification` (text that cites the retrieved chunks and explains reasoning).

### Request

```json
{
  "document_url": "https://example.com/docs/claim123.pdf",
  "questions": [
    "Is this claim eligible for reimbursement?",
    "List the diagnoses mentioned and the treatment dates."
  ],
  "options": {
    "top_k": 5,
    "model": "gpt-4",
    "temperature": 0.0
  }
}
```

* `document_url` (string) — public or pre-signed URL to the document. If the document has been previously ingested, the service will reuse the existing index entries.
* `questions` (array[string]) — list of natural-language questions to answer.
* `options` (object, optional) — runtime options: `top_k` (number of retrieved chunks), `model`, `temperature`, etc.

### Response

```json
{
  "answers": [
    {
      "question": "Is this claim eligible for reimbursement?",
      "answer": "Yes — based on Section 4.2 and the documented dates the claim falls within coverage.",
      "justification": "Supporting excerpts: [Chunk #12] 'treatment performed on 2024-11-12', [Chunk #5] 'policy covers inpatient procedures for X'...",
      "sources": [
        {"chunk_id":"123","page":5,"text_snippet":"treatment performed on 2024-11-12"}
      ],
      "confidence": 0.87
    },
    {
      "question": "List the diagnoses mentioned and the treatment dates.",
      "answer": "Diagnosis: Acute appendicitis (2024-11-12).",
      "justification": "Found explicit diagnosis on page 2: 'Acute appendicitis' and discharge summary with date 2024-11-12.",
      "sources": [...],
      "confidence": 0.92
    }
  ]
}
```

> The `justification` field is required in the system design: it should contain direct citations to the retrieved chunks (IDs and short snippets) and a concise explanation of how the answer was arrived at.

---

## Getting started

### Requirements

* Python 3.10+
* PostgreSQL 12+
* Pinecone account (or a compatible vector DB)
* OpenAI account with GPT-4 access (or a compatible LLM endpoint)

### Environment variables

Create a `.env` file or configure environment variables:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENV=us-west1-gcp
PINECONE_INDEX=rag-insurance-index
DATABASE_URL=postgresql://user:pass@localhost:5432/rag_db
MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOP_K=10
```

### Installation

```bash
git clone <repo-url>
cd rag-insurance-decisioning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` to view Swagger UI and test the `/hackrx/run` endpoint.

---

## Docker / Deployment

Include a `Dockerfile` and `docker-compose.yml` for easier deployment. Example `docker-compose.yml` should bring up the API container and a PostgreSQL service. For production, use managed Postgres and Pinecone; secure secrets in environment management.

---

## Testing the API

Use `curl` or Postman to call the endpoint.

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
 -H "Content-Type: application/json" \
 -d '{"document_url":"https://.../claim.pdf","questions":["Is this claim valid?"]}'
```

Unit tests: `pytest` included in repo. Integration tests mock the LLM and vector DB for deterministic outputs.

---

## Evaluation & performance targets

* **Accuracy / correctness**: evaluate answers against a labeled test set — measure exact-match, F1, and human-annotated correctness.
* **Latency**: target end-to-end response time < 30s (set by the project requirement). Benchmark retrieval, embedding, and LLM steps separately.
* **Explainability**: human evaluation of justifications for transparency.

---

## Prompt engineering & cost-control tips

* Use concise, deterministic prompts with low temperature (0.0–0.2) for decisioning.
* Include retrieved chunks as few-shot context rather than the entire doc.
* Limit `top_k` to the necessary number (e.g., 3–7) to avoid large token costs.
* Cache embeddings / index lookups for repeated documents.

---

## Security & privacy

* Do not log sensitive PII unmasked. Store only metadata and minimal snippets required for justification.
* If handling PHI/medical data, follow applicable regulations (e.g., HIPAA-equivalent local rules). Use encrypted storage and secure network channels (HTTPS, TLS).
* Use pre-signed URLs for private documents and short-lived credentials for storage access.

---

## Troubleshooting

* **Slow responses**: measure which stage is slow (download, embedding, vector search, LLM). Increase chunking parallelism or use larger `top_k` caching.
* **Low-quality answers**: increase top_k, check embedding model, add clarifying prompt instructions, or increase context quality by better chunking.
* **Index not found**: confirm Pinecone index name and environment; ensure API keys have permission.

---


## Future work / Roadmap

* Support multi-document queries and cross-document reasoning.
* Add a lightweight local embedding option for offline usage.
* Add a UI for human-in-the-loop verification and correction.
* Add differential privacy / redaction for PII-sensitive fields.

---

## Contact

Maintainer: Your name — `dev.amkharbanda246@gmail.com`

---

## License

MIT License — see `LICENSE` file.
