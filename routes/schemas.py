"""Shared Pydantic request/response models for the REST API."""
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    doc_id: str
    file_name: str
    upload_timestamp: Optional[str]
    total_pages: int
    total_chunks: int
    processing_status: str
    category: Optional[str] = None
    category_confidence: Optional[str] = None


class UploadResponse(BaseModel):
    message: str
    metadata: DocumentOut


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=4, ge=1, le=20)
    doc_ids: Optional[List[str]] = None
    mode: str = Field(default="hybrid", pattern="^(semantic|keyword|hybrid)$")


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    document: str
    page: int
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: List[SearchResultItem]


class AskRequest(BaseModel):
    question: str
    session_id: str = Field(..., description="Client-generated session identifier for conversation memory")
    doc_ids: Optional[List[str]] = None
    mode: str = Field(default="hybrid", pattern="^(semantic|keyword|hybrid)$")


class Citation(BaseModel):
    document: str
    page: int


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved_context: List[str]
    confidence_score: float
    search_mode: str


class SummarizeRequest(BaseModel):
    doc_id: str


class SummarizeResponse(BaseModel):
    doc_id: str
    file_name: str
    summary: str
    chunks_used: Optional[int] = None


class CompareRequest(BaseModel):
    doc_ids: List[str] = Field(..., min_length=2)


class CompareResponse(BaseModel):
    doc_ids: List[str]
    comparison: str


class ClassifyResponse(BaseModel):
    doc_id: str
    category: Optional[str]
    confidence: Optional[float]


class AnalyticsResponse(BaseModel):
    total_documents: int
    total_processed_chunks: int
    total_embeddings_generated: int
    category_distribution: dict
    total_questions_answered: int
    most_queried_documents: list
