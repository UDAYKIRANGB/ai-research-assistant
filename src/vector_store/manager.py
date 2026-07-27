"""
Vector database indexing & retrieval (ChromaDB).

Search Modes
------------
- SEMANTIC : dense-vector cosine similarity via embeddings. Best for
  conceptual / paraphrased queries where exact wording differs from the
  source text ("How does the system verify identity?" -> "authentication").
- KEYWORD  : sparse BM25 ranking over raw chunk text. Best for exact terms,
  acronyms, code identifiers, or names that embeddings can under-weight
  (e.g. "CVE-2023-1234", "RecursiveCharacterTextSplitter").
- HYBRID   : linear combination of normalized semantic + BM25 scores. Best
  default for general-purpose Q&A because it captures both conceptual
  matches and exact-term matches, trading a little speed for recall.
"""
from typing import Any, Dict, List

import chromadb
from rank_bm25 import BM25Okapi

from config.settings import settings
from src.logging_config import get_logger
from src.vector_store.embeddings import get_embedding_provider

logger = get_logger(__name__)

COLLECTION_NAME = "document_chunks"


class VectorStoreManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.vector_db_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = get_embedding_provider()
        self._bm25_index: BM25Okapi | None = None
        self._bm25_corpus_ids: List[str] = []

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index_chunks(self, chunks: List[Dict[str, Any]], file_name: str) -> int:
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "file_name": file_name,
                "page_number": c["page_number"],
            }
            for c in chunks
        ]

        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        self._bm25_index = None  # invalidate cached BM25 index
        logger.info("Indexed %d chunks for document '%s'", len(chunks), file_name)
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})
        self._bm25_index = None
        logger.info("Deleted vectors for doc_id=%s", doc_id)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def semantic_search(self, query: str, top_k: int = 4, doc_ids: List[str] | None = None) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed_query(query)
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
        return self._format_chroma_results(results)

    def keyword_search(self, query: str, top_k: int = 4, doc_ids: List[str] | None = None) -> List[Dict[str, Any]]:
        self._ensure_bm25_index(doc_ids)
        if self._bm25_index is None or not self._bm25_corpus_ids:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)
        ranked = sorted(zip(self._bm25_corpus_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for chunk_id, score in ranked:
            if score <= 0:
                continue
            record = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            if record["ids"]:
                results.append(
                    {
                        "chunk_id": chunk_id,
                        "text": record["documents"][0],
                        "metadata": record["metadatas"][0],
                        "score": float(score),
                    }
                )
        return results

    def hybrid_search(self, query: str, top_k: int = 4, doc_ids: List[str] | None = None,
                       alpha: float = 0.6) -> List[Dict[str, Any]]:
        """alpha weights semantic vs keyword score (alpha=1.0 -> pure semantic)."""
        semantic = self.semantic_search(query, top_k=top_k * 2, doc_ids=doc_ids)
        keyword = self.keyword_search(query, top_k=top_k * 2, doc_ids=doc_ids)

        def normalize(results: List[Dict[str, Any]]) -> Dict[str, float]:
            if not results:
                return {}
            scores = [r["score"] for r in results]
            lo, hi = min(scores), max(scores)
            span = (hi - lo) or 1.0
            return {r["chunk_id"]: (r["score"] - lo) / span for r in results}

        sem_scores = normalize(semantic)
        kw_scores = normalize(keyword)
        all_chunks = {r["chunk_id"]: r for r in semantic + keyword}

        combined = []
        for chunk_id, record in all_chunks.items():
            score = alpha * sem_scores.get(chunk_id, 0.0) + (1 - alpha) * kw_scores.get(chunk_id, 0.0)
            combined.append({**record, "score": score})

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _ensure_bm25_index(self, doc_ids: List[str] | None = None) -> None:
        if self._bm25_index is not None:
            return
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        record = self.collection.get(where=where, include=["documents"])
        self._bm25_corpus_ids = record["ids"]
        tokenized_corpus = [doc.lower().split() for doc in record["documents"]]
        if tokenized_corpus:
            self._bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self._bm25_index = None

    @staticmethod
    def _format_chroma_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        formatted = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            formatted.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": meta,
                    "score": 1 - dist,  # convert cosine distance -> similarity
                }
            )
        return formatted


_manager_singleton: VectorStoreManager | None = None


def get_vector_store_manager() -> VectorStoreManager:
    global _manager_singleton
    if _manager_singleton is None:
        _manager_singleton = VectorStoreManager()
    return _manager_singleton
