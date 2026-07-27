"""RAG Question Answering with strict context grounding, citations, and
conversation memory."""
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.llm_provider import get_llm
from src.logging_config import get_logger
from src.rag.memory import ConversationMemory
from src.vector_store.manager import get_vector_store_manager

logger = get_logger(__name__)

PROMPT_TEMPLATE = """You are an AI Research Assistant. Answer the user's question using ONLY the
provided document context below. Do not use outside knowledge.

If the context does not contain sufficient information to answer, reply
exactly: "I cannot determine the answer from the provided documents."

Conversation History:
{history}

Context:
{context}

Question: {question}

Provide a clear, direct answer followed by an explicit list of source
documents and page numbers referenced.
"""

NO_ANSWER_PHRASE = "I cannot determine the answer from the provided documents."


class RAGQuestionAnswering:
    def __init__(self):
        self.vector_store = get_vector_store_manager()
        self.llm = get_llm()

    def answer_question(
        self,
        db,
        session_id: str,
        query: str,
        doc_ids: Optional[List[str]] = None,
        search_mode: str = "hybrid",
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        top_k = top_k or settings.retrieval_top_k
        memory = ConversationMemory(db, session_id)
        resolved_doc_ids = memory.resolve_doc_ids(query, doc_ids)

        docs = self._retrieve(query, top_k, resolved_doc_ids, search_mode)

        if not docs:
            answer = NO_ANSWER_PHRASE
            confidence = 0.0
        else:
            context_str, citations = self._build_context(docs)
            history = memory.get_history_text()
            prompt = PROMPT_TEMPLATE.format(history=history, context=context_str, question=query)
            answer = self.llm.complete(prompt)
            confidence = self._estimate_confidence(docs)

            if docs:
                memory.set_active_doc(docs[0]["metadata"]["doc_id"])

        memory.add_turn("user", query)
        memory.add_turn("assistant", answer)

        citations = [
            {"document": d["metadata"]["file_name"], "page": d["metadata"]["page_number"]}
            for d in docs
        ]

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_context": [d["text"] for d in docs],
            "confidence_score": confidence,
            "search_mode": search_mode,
            "doc_ids_used": list({d["metadata"]["doc_id"] for d in docs}),
        }

    def _retrieve(self, query: str, top_k: int, doc_ids: Optional[List[str]], mode: str):
        if mode == "semantic":
            return self.vector_store.semantic_search(query, top_k, doc_ids)
        if mode == "keyword":
            return self.vector_store.keyword_search(query, top_k, doc_ids)
        return self.vector_store.hybrid_search(query, top_k, doc_ids)

    @staticmethod
    def _build_context(docs: List[Dict[str, Any]]):
        context_str = ""
        citations = []
        for d in docs:
            file_name = d["metadata"]["file_name"]
            page_no = d["metadata"]["page_number"]
            context_str += f"\n--- Source: {file_name} (Page {page_no}) ---\n{d['text']}\n"
            citations.append({"document": file_name, "page": page_no})
        return context_str, citations

    @staticmethod
    def _estimate_confidence(docs: List[Dict[str, Any]]) -> float:
        if not docs:
            return 0.0
        avg_score = sum(d["score"] for d in docs) / len(docs)
        return round(max(0.0, min(1.0, avg_score)), 3)
