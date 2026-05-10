from fastapi import APIRouter

from app.api.v1 import chat, health, ingestion, retrieval

api_router = APIRouter()

# Unversioned health check (required by Cloud Run / load balancers)
api_router.include_router(health.router, tags=["health"])

# Versioned API routes
api_router.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["ingestion"])
api_router.include_router(retrieval.router, prefix="/api/v1/retrieval", tags=["retrieval"])
api_router.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
