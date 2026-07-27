"""Unit tests for document parsing and chunking (no external services needed)."""
import fitz  # PyMuPDF
import pytest

from src.document_processing.chunker import DocumentChunker, RecursiveCharacterTextSplitter
from src.document_processing.pdf_parser import PDFParser


@pytest.fixture
def sample_pdf(tmp_path):
    """Generates a tiny in-memory PDF with two pages of known text."""
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    for i, text in enumerate(["Page one content about AI research.", "Page two content about vector databases."]):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


class TestPDFParser:
    def test_extract_text_with_metadata_returns_pages(self, sample_pdf):
        parser = PDFParser()
        pages = parser.extract_text_with_metadata(sample_pdf, doc_id="doc-1")

        assert len(pages) == 2
        assert pages[0]["doc_id"] == "doc-1"
        assert pages[0]["page_number"] == 1
        assert "AI research" in pages[0]["text"]
        assert pages[1]["page_number"] == 2

    def test_get_page_count(self, sample_pdf):
        parser = PDFParser()
        assert parser.get_page_count(sample_pdf) == 2

    def test_clean_text_collapses_whitespace(self):
        dirty = "Hello    world\n\n\nfoo   bar"
        cleaned = PDFParser._clean_text(dirty)
        assert "  " not in cleaned
        assert "Hello world" in cleaned


class TestRecursiveCharacterTextSplitter:
    def test_short_text_returns_single_chunk(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text("A short sentence.")
        assert chunks == ["A short sentence."]

    def test_long_text_is_split_into_multiple_chunks(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        long_text = ("Sentence number %d about research topics. " * 40) % tuple(range(40))
        chunks = splitter.split_text(long_text)

        assert len(chunks) > 1
        assert all(len(c) <= 130 for c in chunks)  # allows a little slack for overlap merging

    def test_overlap_preserves_boundary_context(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=15)
        text = "A" * 200
        chunks = splitter.split_text(text)
        # Consecutive chunks should share some overlapping characters.
        assert chunks[0][-15:] in chunks[1]

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)


class TestDocumentChunker:
    def test_create_chunks_preserves_page_metadata(self):
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        pages_data = [
            {"doc_id": "doc-1", "page_number": 1, "text": "First page. " * 10},
            {"doc_id": "doc-1", "page_number": 2, "text": "Second page. " * 10},
        ]
        chunks = chunker.create_chunks(pages_data)

        assert len(chunks) > 0
        assert all(c["doc_id"] == "doc-1" for c in chunks)
        page_numbers = {c["page_number"] for c in chunks}
        assert page_numbers == {1, 2}
        # chunk_ids should be unique and sequential
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))
