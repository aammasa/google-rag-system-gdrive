"""Retrieval service — hybrid BM25 + vector search."""

from app.logger import get_logger
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.retriever import RetrievedChunk

logger = get_logger(__name__)


class RetrievalService:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        logger.info("retrieval_search", query=query[:80], top_k=top_k)
        return await self.retriever.retrieve(query, top_k, score_threshold)
