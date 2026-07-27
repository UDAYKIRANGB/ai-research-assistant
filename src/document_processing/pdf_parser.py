"""PDF text extraction with page-level metadata preservation (PyMuPDF / fitz)."""
from typing import Any, Dict, List

import fitz  # PyMuPDF

from src.logging_config import get_logger

logger = get_logger(__name__)


class PDFParser:
    """Extracts clean, page-indexed text from a PDF file."""

    def extract_text_with_metadata(self, pdf_path: str, doc_id: str) -> List[Dict[str, Any]]:
        """Return a list of {doc_id, page_number, text} dicts, one per non-empty page."""
        extracted_pages: List[Dict[str, Any]] = []
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                raw_text = page.get_text("text")
                cleaned = self._clean_text(raw_text)
                if cleaned:
                    extracted_pages.append(
                        {
                            "doc_id": doc_id,
                            "page_number": page_num + 1,
                            "text": cleaned,
                        }
                    )
        logger.info("Extracted %d non-empty pages from %s", len(extracted_pages), pdf_path)
        return extracted_pages

    def get_page_count(self, pdf_path: str) -> int:
        with fitz.open(pdf_path) as doc:
            return len(doc)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic cleaning: collapse whitespace, strip stray control characters."""
        if not text:
            return ""
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        cleaned = "\n".join(lines)
        # Collapse repeated spaces introduced by PDF text extraction artifacts.
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        return cleaned.strip()
