"""Vertex AI text embedding client — LangChain Embeddings-compatible."""

import vertexai
from langchain_google_vertexai import VertexAIEmbeddings

from app.config import get_settings
from app.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class VertexEmbeddings:
    """
    Wraps ``langchain_google_vertexai.VertexAIEmbeddings`` with lazy init.

    Implements the same ``embed_documents`` / ``embed_query`` interface so it
    can be swapped for any other LangChain Embeddings provider without changing
    call sites.
    """

    def __init__(self) -> None:
        self._model: VertexAIEmbeddings | None = None

    def _load_model(self) -> VertexAIEmbeddings:
        vertexai.init(
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
        model = VertexAIEmbeddings(
            model_name=settings.vertex_embedding_model,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
        logger.info(
            "vertex_embeddings_loaded",
            model=settings.vertex_embedding_model,
            project=settings.google_cloud_project,
        )
        return model

    @property
    def model(self) -> VertexAIEmbeddings:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    # ── LangChain Embeddings interface ────────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of document strings (runs synchronously)."""
        logger.debug("embedding_documents", count=len(texts))
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (runs synchronously)."""
        return self.model.embed_query(text)
