"""Unit tests for conversation memory and the RAG QA chain.

Vector store and LLM calls are mocked so these tests run fast and offline -
they verify orchestration logic (retrieval scoping, citation building,
pronoun resolution, no-answer fallback), not the quality of a real model.
"""
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.base import Base
from src.rag.memory import ConversationMemory
from src.rag.qa_chain import NO_ANSWER_PHRASE, RAGQuestionAnswering


@pytest.fixture
def db_session():
    """In-memory SQLite session, isolated per test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestConversationMemory:
    def test_creates_session_on_first_use(self, db_session):
        memory = ConversationMemory(db_session, session_id="s1")
        assert memory.session.session_id == "s1"

    def test_add_turn_and_history(self, db_session):
        memory = ConversationMemory(db_session, session_id="s1")
        memory.add_turn("user", "Summarize Paper A.")
        memory.add_turn("assistant", "Paper A is about RAG systems.")

        history = memory.get_history_text()
        assert "Summarize Paper A." in history
        assert "Paper A is about RAG systems." in history

    def test_explicit_doc_ids_set_active_doc(self, db_session):
        memory = ConversationMemory(db_session, session_id="s1")
        resolved = memory.resolve_doc_ids("What is this about?", explicit_doc_ids=["doc-123"])

        assert resolved == ["doc-123"]
        assert memory.session.last_active_doc_id == "doc-123"

    def test_pronoun_followup_resolves_to_last_active_doc(self, db_session):
        memory = ConversationMemory(db_session, session_id="s1")
        memory.set_active_doc("doc-abc")

        resolved = memory.resolve_doc_ids("What are its limitations?", explicit_doc_ids=None)
        assert resolved == ["doc-abc"]

    def test_no_followup_without_prior_context_returns_none(self, db_session):
        memory = ConversationMemory(db_session, session_id="s1")
        resolved = memory.resolve_doc_ids("What is the capital of France?", explicit_doc_ids=None)
        assert resolved is None


class TestRAGQuestionAnswering:
    def _make_chunk(self, text="Relevant text.", doc_id="doc-1", file_name="paper.pdf", page=3, score=0.9):
        return {
            "chunk_id": f"{doc_id}_c0",
            "text": text,
            "metadata": {"doc_id": doc_id, "file_name": file_name, "page_number": page},
            "score": score,
        }

    @patch("src.rag.qa_chain.get_vector_store_manager")
    @patch("src.rag.qa_chain.get_llm")
    def test_answer_includes_citations_and_confidence(self, mock_get_llm, mock_get_vs, db_session):
        mock_vs = MagicMock()
        mock_vs.hybrid_search.return_value = [self._make_chunk()]
        mock_get_vs.return_value = mock_vs

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "RAG systems retrieve context before generating answers."
        mock_get_llm.return_value = mock_llm

        qa = RAGQuestionAnswering()
        result = qa.answer_question(db_session, session_id="s1", query="What is RAG?")

        assert result["answer"] == "RAG systems retrieve context before generating answers."
        assert result["citations"] == [{"document": "paper.pdf", "page": 3}]
        assert 0.0 <= result["confidence_score"] <= 1.0
        assert result["doc_ids_used"] == ["doc-1"]

    @patch("src.rag.qa_chain.get_vector_store_manager")
    @patch("src.rag.qa_chain.get_llm")
    def test_no_retrieved_chunks_returns_fallback_message(self, mock_get_llm, mock_get_vs, db_session):
        mock_vs = MagicMock()
        mock_vs.hybrid_search.return_value = []
        mock_get_vs.return_value = mock_vs
        mock_get_llm.return_value = MagicMock()

        qa = RAGQuestionAnswering()
        result = qa.answer_question(db_session, session_id="s1", query="Unrelated question?")

        assert result["answer"] == NO_ANSWER_PHRASE
        assert result["confidence_score"] == 0.0
        assert result["citations"] == []

    @patch("src.rag.qa_chain.get_vector_store_manager")
    @patch("src.rag.qa_chain.get_llm")
    def test_pronoun_followup_scopes_retrieval_to_last_doc(self, mock_get_llm, mock_get_vs, db_session):
        mock_vs = MagicMock()
        mock_vs.hybrid_search.return_value = [self._make_chunk(doc_id="doc-xyz")]
        mock_get_vs.return_value = mock_vs
        mock_get_llm.return_value = MagicMock(complete=MagicMock(return_value="Answer"))

        qa = RAGQuestionAnswering()
        # First turn establishes doc-xyz as the active document.
        qa.answer_question(db_session, session_id="s2", query="Summarize this.", doc_ids=["doc-xyz"])
        # Second turn uses a pronoun and should resolve back to doc-xyz automatically.
        qa.answer_question(db_session, session_id="s2", query="What are its limitations?")

        args, _ = mock_vs.hybrid_search.call_args
        assert args[2] == ["doc-xyz"]
