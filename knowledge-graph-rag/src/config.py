# FILE: src/config.py
# CHANGES: Reduced default chunk size and overlap for more precise document retrieval.

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    neo4j_uri: str = Field(alias="NEO4J_URI")
    neo4j_user: str = Field(alias="NEO4J_USER")
    neo4j_password: str = Field(alias="NEO4J_PASSWORD")
    groq_api_key: str = Field(alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cuda", alias="EMBEDDING_DEVICE")
    embedding_cache_dir: str = Field(default="/root/.cache/huggingface", alias="EMBEDDING_CACHE_DIR")
    chunk_size: int = Field(default=300, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    ingestion_batch_size: int = Field(default=64, alias="INGESTION_BATCH_SIZE")
    vector_top_k: int = Field(default=10, alias="VECTOR_TOP_K")
    bm25_top_k: int = Field(default=10, alias="BM25_TOP_K")
    final_top_k: int = Field(default=5, alias="FINAL_TOP_K")
    session_file: str = Field(default="/app/session_data/.session_id", alias="SESSION_FILE")
    data_dir: str = Field(default="/app/data", alias="DATA_DIR")


import logging

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
