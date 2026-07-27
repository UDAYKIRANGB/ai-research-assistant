"""End-to-end document processing pipeline, run as a FastAPI BackgroundTask
after a PDF is uploaded:

    PDF Upload -> Text Extraction -> TF Classification -> Chunking
               -> Embedding -> Vector Indexing -> Status = PROCESSED
"""
from config.settings import settings
from src.database.base import session_scope
from src.database.models import Document, ProcessingStatus
from src.document_processing.chunker import DocumentChunker
from src.document_processing.pdf_parser import PDFParser
from src.logging_config import get_logger
from src.ml.predictor import get_classifier
from src.vector_store.manager import get_vector_store_manager

logger = get_logger(__name__)


def process_pdf_pipeline(doc_id: str, file_path: str, file_name: str) -> None:
    parser = PDFParser()
    chunker = DocumentChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    vector_store = get_vector_store_manager()
    classifier = get_classifier()

    with session_scope() as db:
        doc = db.query(Document).filter_by(doc_id=doc_id).first()
        if not doc:
            logger.error("process_pdf_pipeline: doc_id %s not found in DB", doc_id)
            return
        doc.processing_status = ProcessingStatus.PROCESSING
        db.commit()

    try:
        pages = parser.extract_text_with_metadata(file_path, doc_id)
        full_text_sample = " ".join(p["text"] for p in pages)[:5000]

        chunks = chunker.create_chunks(pages)
        indexed_count = vector_store.index_chunks(chunks, file_name=file_name)

        category, confidence = (None, None)
        if classifier.is_ready() and full_text_sample.strip():
            prediction = classifier.predict(full_text_sample)
            if prediction:
                category, confidence = prediction

        with session_scope() as db:
            doc = db.query(Document).filter_by(doc_id=doc_id).first()
            doc.total_pages = len(pages)
            doc.total_chunks = indexed_count
            doc.category = category
            doc.category_confidence = f"{confidence:.3f}" if confidence is not None else None
            doc.processing_status = ProcessingStatus.PROCESSED
            db.commit()

        logger.info("Successfully processed document %s (%s): %d pages, %d chunks, category=%s",
                    doc_id, file_name, len(pages), indexed_count, category)

    except Exception as exc:
        logger.exception("Failed to process document %s: %s", doc_id, exc)
        with session_scope() as db:
            doc = db.query(Document).filter_by(doc_id=doc_id).first()
            if doc:
                doc.processing_status = ProcessingStatus.FAILED
                doc.error_message = str(exc)
                db.commit()


def reprocess_document(doc_id: str) -> None:
    """Re-runs the pipeline for an already-uploaded document (e.g. after a
    parser bug fix or a new classifier model is trained)."""
    with session_scope() as db:
        doc = db.query(Document).filter_by(doc_id=doc_id).first()
        if not doc:
            raise ValueError(f"Document {doc_id} not found")
        file_path, file_name = doc.file_path, doc.file_name

    # Remove stale vectors before re-indexing to avoid duplicate chunks.
    get_vector_store_manager().delete_document(doc_id)
    process_pdf_pipeline(doc_id, file_path, file_name)
