"""Analysis endpoints: summarization, multi-document comparison, and
on-demand classification lookup."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from routes.schemas import (
    ClassifyResponse,
    CompareRequest,
    CompareResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from src.database.base import get_db
from src.database.models import Document
from src.rag.comparator import DocumentComparator
from src.rag.summarizer import DocumentSummarizer

router = APIRouter(tags=["Analysis"])

_summarizer: DocumentSummarizer | None = None
_comparator: DocumentComparator | None = None


def get_summarizer() -> DocumentSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = DocumentSummarizer()
    return _summarizer


def get_comparator() -> DocumentComparator:
    global _comparator
    if _comparator is None:
        _comparator = DocumentComparator()
    return _comparator


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest, db: Session = Depends(get_db)):
    doc = db.query(Document).filter_by(doc_id=request.doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    result = get_summarizer().summarize(doc_id=doc.doc_id, file_name=doc.file_name)
    return SummarizeResponse(**result)


@router.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest, db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.doc_id.in_(request.doc_ids)).all()
    if len(docs) != len(request.doc_ids):
        raise HTTPException(status_code=404, detail="One or more documents not found.")

    file_names = {d.doc_id: d.file_name for d in docs}
    result = get_comparator().compare(request.doc_ids, file_names)
    return CompareResponse(**result)


@router.get("/classify/{doc_id}", response_model=ClassifyResponse)
def get_classification(doc_id: str, db: Session = Depends(get_db)):
    """Returns the TensorFlow-predicted category for an already-processed document."""
    doc = db.query(Document).filter_by(doc_id=doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    confidence = float(doc.category_confidence) if doc.category_confidence else None
    return ClassifyResponse(doc_id=doc.doc_id, category=doc.category, confidence=confidence)
