from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: str = Field(default="development")
    app_name: str = Field(default="google-rag-agent")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    workers: int = Field(default=1)

    # ── Google Cloud / Vertex AI ─────────────────────────────────────────────
    google_cloud_project: str = Field(default="")
    google_cloud_location: str = Field(default="us-central1")
    google_application_credentials: str = Field(default="")

    # ── Gemini / Vertex AI Models ─────────────────────────────────────────────
    vertex_llm_model: str = Field(default="gemini-1.5-pro")
    vertex_embedding_model: str = Field(default="text-embedding-005")
    vertex_llm_temperature: float = Field(default=0.2)
    vertex_llm_max_tokens: int = Field(default=8192)

    # ── Google Drive ─────────────────────────────────────────────────────────
    gdrive_credentials_file: str = Field(default="credentials/gdrive_oauth_credentials.json")
    gdrive_credentials_json: str = Field(default="")  # base64-encoded credentials.json
    gdrive_token_file: str = Field(default="credentials/gdrive_token.json")
    gdrive_token_json: str = Field(default="")  # base64-encoded token (for server deployments)
    gdrive_scopes: str = Field(default="https://www.googleapis.com/auth/drive.readonly")
    gdrive_root_folder_id: str = Field(default="")

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_persist_directory: str = Field(default="./data/chroma")
    chroma_collection_name: str = Field(default="gdrive_documents")
    chroma_host: str = Field(default="")
    chroma_port: int = Field(default=8000)

    # ── RAG ──────────────────────────────────────────────────────────────────
    rag_chunk_size: int = Field(default=1000)
    rag_chunk_overlap: int = Field(default=200)
    rag_top_k: int = Field(default=5)
    rag_score_threshold: float = Field(default=0.7)

    # ── Google Chat ──────────────────────────────────────────────────────────
    google_chat_project_number: str = Field(default="")
    google_chat_service_account: str = Field(default="")

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8080")

    # ── Google AI Studio (fallback when Vertex AI Gemini is not enabled) ─────
    google_api_key: str = Field(default="")

    # ── Security ─────────────────────────────────────────────────────────────
    api_secret_key: str = Field(default="change-me")

    @property
    def use_google_ai_studio(self) -> bool:
        return bool(self.google_api_key)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def chroma_use_remote(self) -> bool:
        return bool(self.chroma_host)

    @property
    def gdrive_scopes_list(self) -> list[str]:
        return [s.strip() for s in self.gdrive_scopes.split(",") if s.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
