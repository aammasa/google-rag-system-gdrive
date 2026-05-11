"""
Embedding client — auto-selects backend based on available credentials.

  GOOGLE_API_KEY set  → Google AI Studio (text-embedding-004) — no ADC needed
  GOOGLE_API_KEY unset → Vertex AI (text-embedding-005) — requires ADC
"""

import vertexai
from langchain_google_vertexai import VertexAIEmbeddings

from app.config import get_settings
from app.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Google AI Studio embedding model (used when GOOGLE_API_KEY is set)
_GENAI_EMBEDDING_MODEL = "models/gemini-embedding-001"


class VertexEmbeddings:
    """
    Embedding client compatible with the LangChain Embeddings interface.

    Backend selection (same logic as the LLM in chat/agent.py):
      - GOOGLE_API_KEY present → GoogleGenerativeAIEmbeddings (AI Studio, no ADC)
      - GOOGLE_API_KEY absent  → VertexAIEmbeddings (Vertex AI, needs ADC)
    """

    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if settings.use_google_ai_studio:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            model = GoogleGenerativeAIEmbeddings(
                model=_GENAI_EMBEDDING_MODEL,
                google_api_key=settings.google_api_key,
            )
            logger.info(
                "embeddings_loaded",
                backend="google_ai_studio",
                model=_GENAI_EMBEDDING_MODEL,
            )
        else:
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
                "embeddings_loaded",
                backend="vertex_ai",
                model=settings.vertex_embedding_model,
            )
        return model

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        logger.debug("embedding_documents", count=len(texts))
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)
