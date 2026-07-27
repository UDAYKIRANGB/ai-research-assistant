"""
Intelligent text chunking.

Chunking Strategy (see README "Design Decisions" for full justification):
    We use a RECURSIVE CHARACTER SPLIT strategy rather than naive fixed-size
    slicing. The splitter tries to break text at paragraph boundaries first,
    then sentences, then words, and only falls back to a hard character cut
    as a last resort. This keeps semantically related sentences together far
    more often than blind slicing, which improves embedding quality and
    retrieval relevance.

    Defaults: chunk_size=1000 chars, chunk_overlap=150 chars.
    - 1000 chars (~180-220 tokens) is small enough to keep each chunk
      topically focused (better retrieval precision) but large enough to
      preserve enough context for the LLM to reason over.
    - A 150-character overlap prevents a sentence/idea from being cut in half
      right at a chunk boundary, so its meaning isn't lost to whichever side
      the split lands on - both neighbouring chunks retain enough surrounding
      context to remain independently coherent.
"""
from typing import Any, Dict, List, Optional

from src.logging_config import get_logger

logger = get_logger(__name__)

# Ordered from "most preferred" to "least preferred" split point.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


class RecursiveCharacterTextSplitter:
    """A from-scratch, dependency-free re-implementation of the common
    'recursive character' chunking strategy (as popularized by LangChain).

    Algorithm
    ---------
    1. Try the highest-priority separator (e.g. "\\n\\n" for paragraphs).
       Split the text on it.
    2. Greedily pack consecutive pieces into a chunk as long as it stays
       under `chunk_size`. When adding the next piece would overflow,
       close the current chunk off and start a new one, seeding it with a
       trailing slice (`chunk_overlap` chars) of the previous chunk so
       context isn't lost at the boundary.
    3. Any individual piece that is *itself* longer than `chunk_size`
       (e.g. one huge paragraph with no line breaks) is recursively split
       again using the next separator down the priority list.
    4. If no separator works (single giant token / no whitespace at all),
       fall back to a hard character-count slice with overlap.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150,
                 separators: Optional[List[str]] = None):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def split_text(self, text: str) -> List[str]:
        chunks = self._split(text, self.separators)
        return [c.strip() for c in chunks if c.strip()]

    # ------------------------------------------------------------------ #
    def _split(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        separator, remaining_separators = separators[0], separators[1:]

        if separator == "":
            return self._hard_split(text)

        pieces = [p for p in text.split(separator) if p != ""]
        if len(pieces) <= 1:
            # This separator didn't actually break the text up - try the next one.
            if remaining_separators:
                return self._split(text, remaining_separators)
            return self._hard_split(text)

        # Recursively break down any individual piece that's still too big.
        normalized_pieces: List[str] = []
        for piece in pieces:
            if len(piece) > self.chunk_size:
                normalized_pieces.extend(
                    self._split(piece, remaining_separators) if remaining_separators else self._hard_split(piece)
                )
            else:
                normalized_pieces.append(piece)

        return self._pack_with_overlap(normalized_pieces, separator)

    def _pack_with_overlap(self, pieces: List[str], separator: str) -> List[str]:
        """Greedily merges small pieces into <=chunk_size chunks, carrying a
        bounded amount of trailing context forward as overlap."""
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        def join(parts: List[str]) -> str:
            return separator.join(parts)

        for piece in pieces:
            added_len = len(piece) + (len(separator) if current else 0)

            if current and current_len + added_len > self.chunk_size:
                chunks.append(join(current))

                # Seed the next chunk with a bounded overlap tail from the
                # chunk we just closed, so context carries across the boundary.
                overlap_parts: List[str] = []
                overlap_len = 0
                for part in reversed(current):
                    part_len = len(part) + (len(separator) if overlap_parts else 0)
                    if overlap_len + part_len > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_len += part_len

                current = overlap_parts
                current_len = overlap_len

                added_len = len(piece) + (len(separator) if current else 0)

            current.append(piece)
            current_len += added_len

        if current:
            chunks.append(join(current))

        return chunks

    def _hard_split(self, text: str) -> List[str]:
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += step
        return chunks


class DocumentChunker:
    """Splits page-level extracted text into overlapping chunks while
    preserving doc_id / page_number metadata for accurate citations."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)

    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        chunk_id = 0
        for page in pages_data:
            text_pieces = self.splitter.split_text(page["text"])
            for piece in text_pieces:
                chunks.append(
                    {
                        "chunk_id": f"{page['doc_id']}_c{chunk_id}",
                        "doc_id": page["doc_id"],
                        "page_number": page["page_number"],
                        "text": piece,
                    }
                )
                chunk_id += 1
        logger.info("Created %d chunks from %d pages", len(chunks), len(pages_data))
        return chunks
