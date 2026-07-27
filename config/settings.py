"""
Centralized application configuration.

All environment-driven settings are declared here using pydantic-settings.
This is the single source of truth for configuration values across the app -
no module should read os.environ directly; import `settings` from here instead.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM Provider ---
    llm_provider: str = "openai"          # "openai" | "ollama"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # --- Embeddings ---
    embedding_provider: str = "huggingface"  # "huggingface" | "openai"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Storage paths ---
    upload_dir: str = "./data/raw_documents"
    vector_db_dir: str = "./data/vector_db"
    dataset_dir: str = "./data/dataset"
    tf_model_path: str = "./models/tf_classifier.h5"
    tokenizer_path: str = "./models/tokenizer.pickle"

    # --- Database ---
    database_url: str = "sqlite:///./data/app_metadata.db"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval ---
    retrieval_top_k: int = 4

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so the .env file is parsed only once."""
    return Settings()


settings = get_settings()
