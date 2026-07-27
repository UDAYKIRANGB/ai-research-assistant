"""SQLAlchemy ORM models: document metadata, chat sessions/messages, query logs."""
import enum
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from src.database.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class ProcessingStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class Document(Base):
    """Represents an uploaded PDF and its processing metadata."""
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True, default=_uuid)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    category = Column(String, nullable=True)          # predicted TensorFlow category
    category_confidence = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "upload_timestamp": self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "processing_status": self.processing_status.value if self.processing_status else None,
            "category": self.category,
            "category_confidence": self.category_confidence,
        }


class ChatSession(Base):
    """A conversation session used to maintain multi-turn memory."""
    __tablename__ = "chat_sessions"

    session_id = Column(String, primary_key=True, default=_uuid)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active_doc_id = Column(String, nullable=True)   # last document referenced ("it"/"its")

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """A single message (user or assistant) within a chat session."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"))
    role = Column(String, nullable=False)   # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class QueryLog(Base):
    """Every question asked, used for analytics (most-queried docs, totals, etc.)."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=True)
    question = Column(Text, nullable=False)
    referenced_doc_ids = Column(Text, nullable=True)   # comma-separated doc_ids cited
    created_at = Column(DateTime, default=datetime.utcnow)
