"""System analytics: document counts, chunk counts, category distribution,
top-queried documents, and total questions answered."""
from collections import Counter
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from src.database.models import Document, QueryLog


def get_system_analytics(db: Session) -> Dict[str, Any]:
    documents: List[Document] = db.query(Document).all()
    total_documents = len(documents)
    total_chunks = sum(d.total_chunks or 0 for d in documents)

    category_distribution = Counter(d.category for d in documents if d.category)

    query_logs: List[QueryLog] = db.query(QueryLog).all()
    total_questions_answered = len(query_logs)

    doc_query_counter: Counter = Counter()
    for log in query_logs:
        if log.referenced_doc_ids:
            for doc_id in log.referenced_doc_ids.split(","):
                doc_id = doc_id.strip()
                if doc_id:
                    doc_query_counter[doc_id] += 1

    doc_name_lookup = {d.doc_id: d.file_name for d in documents}
    top_queried_documents = [
        {
            "doc_id": doc_id,
            "file_name": doc_name_lookup.get(doc_id, "Unknown"),
            "query_count": count,
        }
        for doc_id, count in doc_query_counter.most_common(10)
    ]

    return {
        "total_documents": total_documents,
        "total_processed_chunks": total_chunks,
        "total_embeddings_generated": total_chunks,  # 1 embedding per chunk
        "category_distribution": dict(category_distribution),
        "total_questions_answered": total_questions_answered,
        "most_queried_documents": top_queried_documents,
    }


def log_query(db: Session, session_id: str, question: str, referenced_doc_ids: List[str]) -> None:
    db.add(
        QueryLog(
            session_id=session_id,
            question=question,
            referenced_doc_ids=",".join(referenced_doc_ids) if referenced_doc_ids else "",
        )
    )
    db.commit()
