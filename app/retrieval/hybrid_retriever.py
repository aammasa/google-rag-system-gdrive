"""
Hybrid retriever — BM25 keyword search + vector search merged via RRF.

Reciprocal Rank Fusion (RRF) formula:
    score(d) = Σ 1 / (k + rank(d))   where k=60 (standard constant)

This gives each document a score based on its rank in both result lists,
naturally handling scale differences between BM25 and cosine similarity.
"""

import asyncio
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from app.embeddings.vertex_embeddings import VertexEmbeddings
from app.logger import get_logger
from app.retrieval.retriever import RetrievedChunk
from app.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)

_RRF_K = 60  # standard RRF constant — higher = less weight on top ranks


@dataclass
class HybridRetriever:
    """
    Two-stage retriever:
      1. Vector search  — semantic similarity via ChromaDB
      2. BM25 search    — keyword overlap against the same candidate pool
      3. RRF merge      — combine both ranked lists into one
    """

    embeddings: VertexEmbeddings
    vectorstore: ChromaStore
    bm25_candidates: int = 50   # fetch this many from vector search to build BM25 index

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,  # RRF scores are not cosine values; keep threshold low
    ) -> list[RetrievedChunk]:

        # 1. Vector search — fetch a larger candidate pool for BM25 to re-rank
        embedding: list[float] = await asyncio.to_thread(
            self.embeddings.embed_query, query
        )
        candidates = self.vectorstore.query(
            query_embedding=embedding,
            top_k=self.bm25_candidates,
            score_threshold=0.0,  # get all candidates; RRF handles filtering
        )

        if not candidates:
            return []

        # 2. BM25 search over the candidate pool
        tokenized_corpus = [c["content"].lower().split() for c in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.lower().split())

        # 3. Build ranked lists
        # Vector rank: candidates are already ordered by cosine similarity
        vector_ranks = {c["document_id"]: rank for rank, c in enumerate(candidates)}

        # BM25 rank: sort by BM25 score descending
        bm25_order = sorted(range(len(candidates)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranks = {candidates[i]["document_id"]: rank for rank, i in enumerate(bm25_order)}

        # 4. RRF fusion
        doc_ids = {c["document_id"] for c in candidates}
        rrf_scores: dict[str, float] = {}
        for doc_id in doc_ids:
            v_rank = vector_ranks.get(doc_id, len(candidates))
            b_rank = bm25_ranks.get(doc_id, len(candidates))
            rrf_scores[doc_id] = (1 / (_RRF_K + v_rank)) + (1 / (_RRF_K + b_rank))

        # 5. Re-rank candidates by RRF score and return top_k
        id_to_candidate = {c["document_id"]: c for c in candidates}
        ranked = sorted(doc_ids, key=lambda d: rrf_scores[d], reverse=True)

        results: list[RetrievedChunk] = []
        for doc_id in ranked[:top_k]:
            c = id_to_candidate[doc_id]
            results.append(
                RetrievedChunk(
                    document_id=c["document_id"],
                    content=c["content"],
                    score=round(rrf_scores[doc_id], 6),  # RRF score, not cosine
                    metadata=c["metadata"],
                )
            )

        logger.info(
            "hybrid_retrieval_done",
            query=query[:80],
            candidates=len(candidates),
            returned=len(results),
        )
        return results
