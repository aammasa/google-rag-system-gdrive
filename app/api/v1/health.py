from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter()
settings = get_settings()


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """Liveness probe — returns 200 when the service is running."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.get("/ready", response_model=HealthResponse, summary="Readiness check")
async def readiness_check() -> HealthResponse:
    """
    Readiness probe — will return 503 once downstream dependency checks
    (ChromaDB, Vertex AI) are wired up and any of them are unhealthy.
    """
    # TODO: ping ChromaDB, check Vertex AI credentials
    return HealthResponse(
        status="ready",
        version=settings.app_version,
        environment=settings.app_env,
    )
