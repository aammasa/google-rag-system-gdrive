from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.dependencies import RetrievalServiceDep

router = APIRouter()


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    score_threshold: float = Field(0.7, ge=0.0, le=1.0)


class RetrievalResult(BaseModel):
    document_id: str
    content: str
    score: float
    metadata: dict


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalResult]


@router.post("/search", response_model=RetrievalResponse, summary="Semantic search")
async def semantic_search(
    payload: RetrievalRequest,
    service: RetrievalServiceDep,
) -> RetrievalResponse:
    """
    Embeds the query via Vertex AI and performs a similarity search
    against the ChromaDB vector store.
    """
    chunks = await service.search(
        query=payload.query,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
    )
    results = [
        RetrievalResult(
            document_id=c.document_id,
            content=c.content,
            score=c.score,
            metadata=c.metadata,
        )
        for c in chunks
    ]
    return RetrievalResponse(query=payload.query, results=results)
