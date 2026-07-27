"""Multi-document comparison engine: methodologies, pros/cons, similarities,
differences, conclusions, and implementation approaches across 2+ docs."""
from typing import Any, Dict, List

from src.llm_provider import get_llm
from src.logging_config import get_logger
from src.vector_store.manager import get_vector_store_manager

logger = get_logger(__name__)

COMPARE_PROMPT = """You are an AI Research Assistant comparing multiple documents.
Using ONLY the content provided per document below, produce a structured
comparison with these sections:

### Methodologies
### Advantages & Disadvantages
### Similarities
### Differences
### Implementation Approaches
### Conclusions

If a document does not provide enough information for a section, say so
explicitly rather than inventing content.

{documents_block}
"""


class DocumentComparator:
    def __init__(self):
        self.vector_store = get_vector_store_manager()
        self.llm = get_llm()

    def compare(self, doc_ids: List[str], file_names: Dict[str, str], max_chars_per_doc: int = 6000) -> Dict[str, Any]:
        if len(doc_ids) < 2:
            raise ValueError("At least two documents are required for comparison.")

        documents_block = ""
        for doc_id in doc_ids:
            chunks = self.vector_store.semantic_search(
                query="key methodology results conclusion", top_k=20, doc_ids=[doc_id]
            )
            content = "\n".join(c["text"] for c in chunks)[:max_chars_per_doc]
            file_name = file_names.get(doc_id, doc_id)
            documents_block += f"\n\n=== Document: {file_name} (doc_id={doc_id}) ===\n{content}"

        prompt = COMPARE_PROMPT.format(documents_block=documents_block)
        comparison_text = self.llm.complete(prompt)

        return {
            "doc_ids": doc_ids,
            "comparison": comparison_text,
        }
