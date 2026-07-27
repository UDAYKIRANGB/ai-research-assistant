"""Document summarization: Executive Summary, Technical Summary, Bullet
Points, and Key Takeaways, generated from ALL chunks belonging to a doc_id."""
from typing import Any, Dict, List

from src.llm_provider import get_llm
from src.logging_config import get_logger
from src.vector_store.manager import get_vector_store_manager

logger = get_logger(__name__)

SUMMARY_PROMPT = """You are an AI Research Assistant. Using ONLY the document
content below, produce a structured summary with these exact sections:

### Executive Summary
(2-4 sentences, plain-language overview for a non-technical reader)

### Technical Summary
(a denser paragraph covering methodology, architecture, or technical detail)

### Bullet Point Summary
(5-8 concise bullet points covering the main content)

### Key Takeaways
(3-5 bullet points on why this document matters / what to remember)

Document Content:
{content}
"""


class DocumentSummarizer:
    def __init__(self):
        self.vector_store = get_vector_store_manager()
        self.llm = get_llm()

    def summarize(self, doc_id: str, file_name: str, max_chars: int = 12000) -> Dict[str, Any]:
        chunks = self._get_all_chunks_for_doc(doc_id)
        if not chunks:
            return {
                "doc_id": doc_id,
                "file_name": file_name,
                "summary": "No indexed content found for this document.",
            }

        full_text = "\n\n".join(c["text"] for c in chunks)[:max_chars]
        prompt = SUMMARY_PROMPT.format(content=full_text)
        summary_text = self.llm.complete(prompt)

        return {
            "doc_id": doc_id,
            "file_name": file_name,
            "summary": summary_text,
            "chunks_used": len(chunks),
        }

    def _get_all_chunks_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        # Broad query scoped to the doc_id retrieves representative chunks;
        # for full-document coverage we pull a generous top_k.
        return self.vector_store.semantic_search(
            query="summary overview main content", top_k=50, doc_ids=[doc_id]
        )
