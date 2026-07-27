"""Document management endpoints: upload, list, delete, reprocess."""
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from config.settings import settings
from routes.schemas import DocumentOut, UploadResponse
from src.database.base import get_db
from src.database.models import Document, ProcessingStatus
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.pipeline import process_pdf_pipeline, reprocess_document
from src.logging_config import get_logger
from src.vector_store.manager import get_vector_store_manager

router = APIRouter(prefix="/documents", tags=["Document Management"])
logger = get_logger(__name__)

os.makedirs(settings.upload_dir, exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Uploads a PDF document and triggers background processing
    (text extraction, TF classification, chunking, vector indexing)."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    file_path = os.path.join(settings.upload_dir, f"{doc_id}_{file.filename}")

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Quick page-count sanity check up front (fails fast on a corrupt file).
    try:
        total_pages = PDFParser().get_page_count(file_path)
    except Exception as exc:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

    doc = Document(
        doc_id=doc_id,
        file_name=file.filename,
        file_path=file_path,
        total_pages=total_pages,
        processing_status=ProcessingStatus.PENDING,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(process_pdf_pipeline, doc_id, file_path, file.filename)

    return UploadResponse(
        message="Document uploaded successfully. Processing started in the background.",
        metadata=DocumentOut(**doc.to_dict()),
    )


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    """Lists all uploaded documents with their processing status."""
    docs = db.query(Document).order_by(Document.upload_timestamp.desc()).all()
    return [DocumentOut(**d.to_dict()) for d in docs]


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter_by(doc_id=doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentOut(**doc.to_dict())


@router.delete("/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """Deletes a document's metadata, file, and vector index entries."""
    doc = db.query(Document).filter_by(doc_id=doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    get_vector_store_manager().delete_document(doc_id)

    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    db.delete(doc)
    db.commit()
    return {"message": f"Document {doc_id} deleted successfully."}


@router.post("/{doc_id}/reprocess")
def reprocess(doc_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Re-runs the ingestion pipeline for an already-uploaded document."""
    doc = db.query(Document).filter_by(doc_id=doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc.processing_status = ProcessingStatus.PENDING
    db.commit()

    background_tasks.add_task(reprocess_document, doc_id)
    return {"message": f"Reprocessing started for document {doc_id}."}
