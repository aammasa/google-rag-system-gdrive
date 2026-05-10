"""RAG retrieval — query → embed (thread) → similarity search → ranked chunks."""

import asyncio
from dataclasses import dataclass

from app.config import get_settings
from app.embeddings.vertex_embeddings import VertexEmbeddings
from app.logger import get_logger
from app.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    document_id: str
    content: str
    score: float
    metadata: dict


class Retriever:
    """
    Orchestrates the retrieval step:
      1. Embed the query via VertexEmbeddings (blocking → thread pool)
      2. Query ChromaStore for nearest neighbours
      3. Return ranked RetrievedChunk objects
    """

    def __init__(self, embeddings: VertexEmbeddings, vectorstore: ChromaStore) -> None:
        self.embeddings = embeddings
        self.vectorstore = vectorstore

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        _top_k = top_k or settings.rag_top_k
        _threshold = score_threshold if score_threshold is not None else settings.rag_score_threshold

        # Embedding is a blocking network call — run in a thread so we don't
        # block the asyncio event loop.
        embedding: list[float] = await asyncio.to_thread(
            self.embeddings.embed_query, query
        )

        raw = self.vectorstore.query(
            query_embedding=embedding,
            top_k=_top_k,
            score_threshold=_threshold,
        )

        chunks = [RetrievedChunk(**r) for r in raw]
        logger.info("retrieval_done", query=query[:80], results=len(chunks))
        return chunks
