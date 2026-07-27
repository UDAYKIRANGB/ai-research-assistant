"""FastAPI application entry point.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive API docs (Swagger UI): http://localhost:8000/docs
Alternative docs (ReDoc):          http://localhost:8000/redoc
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from routes import analysis_routes, analytics_routes, document_routes, search_routes
from src.database.base import init_db
from src.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AI Research & Knowledge Assistant",
    description=(
        "A production-oriented RAG backend for uploading, searching, "
        "summarizing, comparing, and classifying research documents."
    ),
    version="1.0.0",
)

# Permissive CORS for local development / API testing tools (Postman, Swagger UI).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document_routes.router)
app.include_router(search_routes.router)
app.include_router(analysis_routes.router)
app.include_router(analytics_routes.router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    logger.info("Application started | env=%s | llm_provider=%s | embedding_provider=%s",
                settings.app_env, settings.llm_provider, settings.embedding_provider)


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "AI Research & Knowledge Assistant",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=True)
