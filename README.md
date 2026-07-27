# AI Research & Knowledge Assistant

A production-oriented, Retrieval-Augmented Generation (RAG) backend for uploading, semantically searching, summarizing, comparing, and classifying research/technical PDF documents — built with FastAPI, ChromaDB, and TensorFlow.

---

## 1. Project Overview

Organizations accumulate large collections of PDFs (research papers, specs, internal docs) that keyword search handles poorly and generic LLMs can only answer about by hallucinating. This project implements a REST API that:

- Ingests PDFs (text extraction → intelligent chunking → embedding → vector indexing)
- Answers questions **grounded only in retrieved document context**, with explicit citations (document + page number) and a confidence score
- Maintains **multi-turn conversation memory**, so a follow-up like *"What are its limitations?"* correctly resolves "its" to the document discussed previously
- Supports **semantic, keyword (BM25), and hybrid search**
- **Compares** and **summarizes** documents using an LLM constrained to retrieved content
- **Classifies** uploaded documents into technical categories using a custom-trained **TensorFlow** model
- Exposes **system analytics** (documents indexed, chunks processed, most-queried documents, etc.)

---

## 2. Architecture Diagram

```
┌────────────────┐
│   PDF Upload    │
└───────┬────────┘
        │
        ▼
┌───────────────────────┐      ┌─────────────────────────┐
│ PDF Parser & Metadata  │ ───► │ TensorFlow Domain        │
│  (PyMuPDF, page-level) │      │ Classifier (.h5 model)   │
└───────┬───────────────┘      └─────────────────────────┘
        │
        ▼
┌───────────────────────┐
│ Recursive Chunking     │  (paragraph → sentence → word → hard split,
│ (chunker.py)           │   with overlap)
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐      ┌─────────────────────────┐
│ Embedding Engine       │ ───► │ ChromaDB Vector Index    │
│ (sentence-transformers)│      │ (persisted to disk)      │
└───────────────────────┘      └────────────┬────────────┘
                                             │
                    ┌────────────────────────┴───────────────────────┐
                    ▼                                                ▼
        ┌───────────────────────┐                       ┌─────────────────────────┐
        │ Semantic / Keyword /   │                       │ Conversation Memory      │
        │ Hybrid Retrieval       │ ────────────────────► │ (SQL-backed sessions)    │
        └───────────────────────┘                       └────────────┬────────────┘
                                                                       │
                                                                       ▼
                                                          ┌─────────────────────────┐
                                                          │ RAG Prompt + Citations   │
                                                          │  → LLM → Final Answer    │
                                                          └─────────────────────────┘
```

Supporting services (Summarization, Comparison, Analytics) all read from the same ChromaDB index and SQLite metadata store, so no data is duplicated across features.

---

## 3. Technology Stack

| Layer | Component | Purpose |
|---|---|---|
| Backend Framework | **FastAPI** + Uvicorn | Async REST API, auto-generated OpenAPI/Swagger docs |
| Document Processing | **PyMuPDF (fitz)** | Page-accurate text extraction |
| Chunking | Custom `RecursiveCharacterTextSplitter` | Paragraph/sentence-aware chunking with overlap |
| Vector Database | **ChromaDB** (persistent, local) | Embedding storage + cosine similarity search |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`, local/free) or OpenAI | Semantic vector representations |
| Keyword Search | **rank-bm25** | Sparse lexical retrieval for hybrid search |
| LLM Engine | **OpenAI** (GPT-4o-mini) or **Ollama** (local Llama-3/Mistral) | Answer generation, summarization, comparison |
| Machine Learning | **TensorFlow / Keras**, scikit-learn | Document domain classification |
| Metadata Database | **SQLite** (SQLAlchemy ORM, Postgres-ready) | Document metadata, chat sessions, query logs |
| Testing | **pytest**, httpx | Unit tests for parsing, chunking, RAG orchestration, ML pipeline |

---

## 4. Project Structure

```
ai-research-assistant/
├── config/                      # Centralized settings (pydantic-settings)
├── data/
│   ├── raw_documents/           # Uploaded PDFs
│   ├── vector_db/                # ChromaDB persistence
│   └── dataset/                  # TensorFlow training data (sample CSV included)
├── models/                      # Trained tf_classifier.h5 + tokenizer.pickle (generated)
├── src/
│   ├── database/                 # SQLAlchemy models & session management
│   ├── document_processing/      # PDF parsing, chunking, ingestion pipeline
│   ├── ml/                       # Dataset prep, TF training, inference
│   ├── vector_store/              # ChromaDB manager, embeddings, BM25 hybrid search
│   ├── rag/                       # QA chain, summarizer, comparator, conversation memory
│   ├── analytics/                 # System usage metrics
│   ├── llm_provider.py            # Pluggable OpenAI / Ollama / offline-fallback LLM
│   └── logging_config.py
├── routes/                       # FastAPI routers (documents, search, analysis, analytics)
├── tests/                        # pytest unit tests
├── main.py                       # FastAPI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## 5. Setup Instructions

### 5.1 Prerequisites
- Python 3.10 or 3.11
- 8 GB RAM minimum (16 GB recommended for local embeddings + TensorFlow training)
- ~5 GB free disk space (mostly for TensorFlow + sentence-transformers + their model weights)

### 5.2 Installation

```bash
git clone <your-repo-url>
cd ai-research-assistant

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env: set OPENAI_API_KEY, or set LLM_PROVIDER=ollama for a fully local/free setup
```

### 5.3 Train the document classifier (one-time)

A sample labelled dataset covering all 7 categories is included at
`data/dataset/training_data.csv`. Train and persist the model:

```bash
python -m src.ml.train_classifier
```

This saves `models/tf_classifier.h5` and `models/tokenizer.pickle`. Newly
uploaded documents are auto-classified once these files exist; if they
don't exist yet, uploads still succeed — classification is simply skipped
(logged as a warning) until you train the model.

### 5.4 Run the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- A Postman collection is included at `postman_collection.json` — import it directly.

### 5.5 Run tests

```bash
pytest                          # fast unit tests (parsing, chunking, RAG orchestration, ML dataset)
RUN_SLOW_TESTS=1 pytest -m slow  # optional: full TensorFlow training smoke test
```

---

## 6. Environment Variables

See `.env.example` for the full list with inline documentation. Key variables:

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | `openai` or `ollama` | `openai` |
| `OPENAI_API_KEY` | Required if using OpenAI | — |
| `EMBEDDING_PROVIDER` | `huggingface` (local/free) or `openai` | `huggingface` |
| `VECTOR_DB_DIR` | ChromaDB persistence path | `./data/vector_db` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters | `1000` / `150` |
| `RETRIEVAL_TOP_K` | Chunks retrieved per query | `4` |
| `DATABASE_URL` | SQLAlchemy connection string | SQLite file |

**No API keys are committed.** `.env` is git-ignored; `.env.example` contains placeholders only.

---

## 7. API Documentation

Full interactive documentation is auto-generated by FastAPI at `/docs`. Summary of endpoints:

### Document Management
| Method | Endpoint | Description |
|---|---|---|
| POST | `/documents/upload` | Upload a PDF; triggers background ingestion pipeline |
| GET | `/documents` | List all documents with processing status |
| GET | `/documents/{doc_id}` | Get a single document's metadata |
| DELETE | `/documents/{doc_id}` | Delete a document (file + vectors + metadata) |
| POST | `/documents/{doc_id}/reprocess` | Re-run the ingestion pipeline for a document |

### Search & Q&A
| Method | Endpoint | Description |
|---|---|---|
| POST | `/search` | Semantic / keyword / hybrid retrieval (no LLM generation) |
| POST | `/ask` | RAG question answering with citations, confidence score, and conversation memory |

### Analysis
| Method | Endpoint | Description |
|---|---|---|
| POST | `/summarize` | Executive / Technical / Bullet-point / Key-takeaway summary for one document |
| POST | `/compare` | Compare 2+ documents across methodology, pros/cons, similarities, etc. |
| GET | `/classify/{doc_id}` | Get the TensorFlow-predicted category for a document |

### Analytics
| Method | Endpoint | Description |
|---|---|---|
| GET | `/analytics` | Total documents, chunks, category distribution, most-queried docs |

**Example: asking a question**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What chunking strategy does this system use?",
        "session_id": "demo-session-1",
        "mode": "hybrid"
      }'
```

Response includes `answer`, `citations` (document + page), `retrieved_context`, `confidence_score`, and `search_mode`.

---

## 8. Assumptions

- Only **PDF** documents are supported for upload (per assignment scope).
- A single **SQLite** file is sufficient for this assignment's scale; `DATABASE_URL` can be pointed at Postgres with no code changes (SQLAlchemy handles both).
- The bundled 105-row sample training dataset is for **demonstration/development only** — a production classifier should be trained on a much larger, professionally labelled corpus (e.g. arXiv abstracts per category).
- `session_id` for conversation memory is generated/managed by the **client** (e.g. a UUID stored per browser tab), not the server, since the API is stateless between processes.
- Background processing (`BackgroundTasks`) is sufficient for this assignment's scale; a production deployment would use a proper task queue (Celery/RQ) for retry semantics and horizontal scaling.

## 9. Design Decisions

- **Chunking strategy**: `RecursiveCharacterTextSplitter` splits on paragraph → sentence → word boundaries before ever resorting to a hard character cut, which keeps semantically related text together far better than fixed-size slicing. `chunk_size=1000` / `chunk_overlap=150` balances retrieval precision (small, focused chunks) against enough surrounding context for the LLM to reason correctly, with overlap preventing an idea from being silently cut in half at a chunk boundary.
- **Search modes**: Semantic search is the default for conceptual queries; keyword (BM25) is offered for exact-term lookups (error codes, class names, acronyms) where embeddings can under-weight exact matches; hybrid (weighted linear combination, `alpha=0.6` favoring semantic) is the recommended general-purpose default and is what `/ask` uses unless overridden.
- **Pluggable LLM/embeddings**: The system defaults to a fully local, free stack (HuggingFace embeddings + Ollama) so it's runnable without any paid API key, while supporting OpenAI as a drop-in swap via `.env` — no code changes needed.
- **Offline fallback LLM**: If no LLM is configured/reachable, `/ask` still returns retrieved context instead of crashing, so the retrieval pipeline remains demonstrable end-to-end even without credentials.
- **Lazy singleton initialization**: RAG/vector-store engines are constructed on first use rather than at module import time, so the API can boot even before heavy ML dependencies (embedding models) finish downloading.
- **Conversation memory via pronoun heuristic**: A lightweight pronoun/keyword check ("it", "its", "this document"...) combined with a per-session `last_active_doc_id` resolves ambiguous follow-ups without requiring a second LLM call just to disambiguate references — a reasonable trade-off between accuracy and latency/cost for this assignment's scope.

## 10. Limitations

- The bundled TensorFlow classifier is trained on a small sample dataset (105 rows); real-world accuracy will be modest until retrained on a larger corpus.
- Conversation memory's pronoun resolution is heuristic-based, not a full coreference-resolution model — it will mis-resolve unusual phrasing.
- Multi-document comparison and full-document summarization currently sample up to ~50 representative chunks per document rather than embedding the entire document verbatim into the prompt, to stay within LLM context limits on large PDFs.
- No authentication/authorization is implemented (listed as a bonus feature) — this is a single-tenant assignment deliverable.

## 11. Future Improvements

- Add authentication (JWT) and multi-user document isolation.
- Add a reranking model (e.g. cross-encoder) after hybrid retrieval for higher precision.
- Move background processing to Celery/RQ with retry and dead-letter handling.
- Add OCR (e.g. Tesseract) fallback for scanned/image-only PDFs.
- Add streaming responses for `/ask` (Server-Sent Events) for a more responsive UX.
- Containerize with Docker + docker-compose (API + ChromaDB + optional Ollama) and add a CI/CD pipeline.

---

## 12. Postman / Swagger

- Swagger UI (auto-generated, always in sync with the code): **http://localhost:8000/docs**
- A static Postman collection covering all endpoints is included: `postman_collection.json`
