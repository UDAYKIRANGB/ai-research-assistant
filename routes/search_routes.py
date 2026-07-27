"""Search & Q&A endpoints: semantic/keyword/hybrid search, RAG-based
question answering with citations and conversation memory."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from routes.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse, SearchResultItem
from src.analytics.metrics import log_query
from src.database.base import get_db
from src.rag.qa_chain import RAGQuestionAnswering
from src.vector_store.manager import get_vector_store_manager

router = APIRouter(tags=["Search & Q&A"])

_qa_engine: RAGQuestionAnswering | None = None


def get_qa_engine() -> RAGQuestionAnswering:
    """Lazily constructs the RAG engine on first use rather than at module
    import time, so the API can start even before embedding/LLM
    dependencies are fully downloaded/configured."""
    global _qa_engine
    if _qa_engine is None:
        _qa_engine = RAGQuestionAnswering()
    return _qa_engine


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """Runs semantic, keyword, or hybrid retrieval across one or more documents."""
    manager = get_vector_store_manager()
    if request.mode == "semantic":
        results = manager.semantic_search(request.query, request.top_k, request.doc_ids)
    elif request.mode == "keyword":
        results = manager.keyword_search(request.query, request.top_k, request.doc_ids)
    else:
        results = manager.hybrid_search(request.query, request.top_k, request.doc_ids)

    items = [
        SearchResultItem(
            chunk_id=r["chunk_id"],
            text=r["text"],
            document=r["metadata"]["file_name"],
            page=r["metadata"]["page_number"],
            score=round(r["score"], 4),
        )
        for r in results
    ]
    return SearchResponse(query=request.query, mode=request.mode, results=items)


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, db: Session = Depends(get_db)):
    """RAG question answering: retrieves relevant chunks, generates a
    citation-grounded answer, and maintains conversation memory per session."""
    result = get_qa_engine().answer_question(
        db=db,
        session_id=request.session_id,
        query=request.question,
        doc_ids=request.doc_ids,
        search_mode=request.mode,
    )

    log_query(db, request.session_id, request.question, result.pop("doc_ids_used", []))

    return AskResponse(**result)
