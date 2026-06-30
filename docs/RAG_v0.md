from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    bearer_token: str = Field(..., alias="HACKATHON_BEARER_TOKEN")
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    environment: str = Field("production", alias="ENVIRONMENT")

    embedding_model: str = "gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    retrieval_top_k: int = 20
    rerank_top_n: int = 8
    final_context_chunks: int = 5
    parent_chunk_size: int = 1500
    child_chunk_size: int = 256
    chunk_overlap: int = 32

    chroma_persist_dir: str = "./chroma_db"
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600
    semantic_cache_threshold: float = 0.04

    langsmith_api_key: str = ""
    langsmith_project: str = "policymind-ai"

    model_config = SettingsConfigDict(
        env_file=".env",
        populate_by_name=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()